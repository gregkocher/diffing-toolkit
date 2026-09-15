"""Condition cache isolation and PID ownership checks for campaign scheduling."""
import importlib.util
import json
from pathlib import Path
import pytest

source=Path(__file__).parents[1]/'experiments/ouro_full_pipeline/pool.py'
spec=importlib.util.spec_from_file_location('ouro_pool_test',source)
pool=importlib.util.module_from_spec(spec);spec.loader.exec_module(pool)


def test_ordering_variants_share_a_lock_but_distinct_depths_do_not():
    assert pool.cache_key('jlens_2_nmf')==pool.cache_key('jlens_2')
    assert pool.cache_key('jlens_1')!=pool.cache_key('jlens_2')
    assert pool.cache_key('jlens_early_2')!=pool.cache_key('jlens_2')


def test_pid_reuse_fails_before_any_signal(tmp_path,monkeypatch):
    worker=tmp_path/'worker';(worker/'bounded').mkdir(parents=True)
    job={'pid':42,'pgid':42,'proc_start_ticks':'100','command':['python','/workspace/diffing-toolkit-full-20260915-v4/experiments/ouro_full_pipeline/run.py','--conditions','adl_3']}
    (worker/'bounded/job.json').write_text(json.dumps(job))
    proc=tmp_path/'proc/42';proc.mkdir(parents=True)
    fields=['x']*22;fields[21]='101';(proc/'stat').write_text(' '.join(fields))
    signals=[];monkeypatch.setattr(pool.os,'killpg',lambda *args:signals.append(args))
    with pytest.raises(AssertionError,match='PID reused'):
        pool.stop_owned_worker(worker,'adl_3',tmp_path/'proc')
    assert signals==[]


def test_wrong_condition_fails_closed(tmp_path):
    worker=tmp_path/'worker';(worker/'bounded').mkdir(parents=True)
    (worker/'bounded/job.json').write_text(json.dumps({'command':['python','other_job.py','--conditions','different']}))
    with pytest.raises(AssertionError):pool.stop_owned_worker(worker,'adl_3')


def test_adl_handoff_requires_all_45_complete_grades(tmp_path):
    assert not pool.adl_grades_complete(tmp_path)
    start=tmp_path/'full_runs/adl_3/attempt_1/STARTED.json';start.parent.mkdir(parents=True)
    start.write_text(json.dumps({'command':['diffing.method.token_relevance.grader.permutations=3','diffing.method.token_relevance.k_candidate_tokens=20']}))
    base=tmp_path/'cache/adl/diffing_results/ouro_1_4B/ouro_cake_eos1221/activation_difference_lens/recurrence_4'
    for layer in (0,11,23):
        for position in range(5):
            for variant in ('difference','base','ft'):
                path=base/f'layer_{layer}/corpus.jsonl/token_relevance/position_{position}/{variant}/relevance_logitlens_gpt-5-mini.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'layer':layer,'position':position,'variant':variant,'source':'logitlens','target':'self','tokens':['a']*20,'labels':['IRRELEVANT']*20,'grader_responses':['ok']*3}))
    assert pool.adl_grades_complete(tmp_path)
    path.write_text('{')  # unfinished final writer must keep ownership
    assert not pool.adl_grades_complete(tmp_path)


def test_external_handoff_requires_quiescent_worker_and_preserves_other_state(tmp_path):
    state = {'active': {'recurrence_1': {'pid': 1}},
             'pending': ['adl_3', 'recurrence_0'], 'deadline_unix': 123}
    assignment = {'conditions': ['recurrence_1', 'adl_3']}
    with pytest.raises(AssertionError, match='still active'):
        pool.apply_external_assignment(state, tmp_path, assignment)
    assert 'recurrence_1' in state['active']
    receipt = tmp_path/'recurrence_1/bounded/WORK_COMPLETE.json'
    receipt.parent.mkdir(parents=True); receipt.write_text('{}')
    pool.apply_external_assignment(state, tmp_path, assignment)
    assert state['active'] == {}
    assert state['pending'] == ['recurrence_0']
    assert state['deadline_unix'] == 123


def test_external_pool_resume_preserves_deadline_and_hands_off_pending_only(tmp_path):
    import subprocess,sys
    repo=Path(__file__).parents[1]
    out=tmp_path/'workers/external_pool_test';out.mkdir(parents=True)
    state={'pending':['jlens_1','jlens_0'],'active':{},'completed':[],'failed':[],
           'deadline_unix':1,'started_unix':0,'max_workers':3}
    (out/'STATE.json').write_text(json.dumps(state))
    result=subprocess.run([sys.executable,str(repo/'experiments/ouro_full_pipeline/external_pool.py'),
        '--repo',str(repo),'--root',str(tmp_path),'--conditions','ignored','--resume',
        '--pool-name','external_pool_test','--exclude-condition','jlens_1'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    saved=json.loads((out/'TIME_LIMIT.json').read_text())
    assert saved['deadline_unix']==1 and saved['pending']==['jlens_0']
    assert saved['external_handoffs'][0]['conditions']==['jlens_1']


def test_external_pool_refuses_handoff_of_running_condition(tmp_path):
    import subprocess,sys
    repo=Path(__file__).parents[1]
    out=tmp_path/'workers/external_pool_test';out.mkdir(parents=True)
    state={'pending':[],'active':{'jlens_1':{}},'completed':[],'failed':[],
           'deadline_unix':1,'started_unix':0,'max_workers':3}
    (out/'STATE.json').write_text(json.dumps(state))
    result=subprocess.run([sys.executable,str(repo/'experiments/ouro_full_pipeline/external_pool.py'),
        '--repo',str(repo),'--root',str(tmp_path),'--conditions','ignored','--resume',
        '--pool-name','external_pool_test','--exclude-condition','jlens_1'],capture_output=True,text=True)
    assert result.returncode!=0 and 'Only untouched pending' in result.stderr
    assert json.loads((out/'STATE.json').read_text())==state
