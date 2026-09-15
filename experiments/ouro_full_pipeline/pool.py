"""Bounded condition-level scheduling around the unchanged toolkit entry point.

Each child runs run.py for one condition. Completed legacy sequential workers
hand off their remaining conditions; cache-equivalent top-K/NMF runs never overlap.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read_json(path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except json.JSONDecodeError:
        return None  # completion metadata may still be publishing


def cache_key(condition):
    return condition.removesuffix('_nmf')


def stop_owned_worker(worker, condition, proc_root=Path('/proc')):
    job = read_json(worker / 'bounded/job.json')
    assert job is not None
    command = job['command']
    assert condition in command and '--conditions' in command
    assert any('/diffing-toolkit-full-20260915' in part and part.endswith('/experiments/ouro_full_pipeline/run.py') for part in command)
    stat = proc_root / str(job['pid']) / 'stat'
    if stat.exists():
        assert stat.read_text().split()[21] == job['proc_start_ticks'], 'PID reused; refusing signal'
        assert os.getpgid(job['pid']) == job['pgid'], 'Unexpected process group'
        os.killpg(job['pgid'], signal.SIGTERM)
    # A stopped scheduler may already have forked the next condition. Wait for
    # its entire group to cease running before releasing that cache to another.
    deadline = time.time() + 90
    while time.time() < deadline:
        live = []
        for candidate in proc_root.glob('[0-9]*/stat'):
            try:
                fields = candidate.read_text().rsplit(')',1)[1].split()
                if int(fields[2]) == int(job['pgid']) and fields[0] != 'Z':
                    live.append(int(candidate.parent.name))
            except (FileNotFoundError, ProcessLookupError):
                pass
        if not live and (worker/'bounded/WORK_COMPLETE.json').exists():
            return job
        time.sleep(.2)
    raise RuntimeError(f'Owned worker group did not exit: {job["pgid"]}')


def adl_grades_complete(root):
    base = root/'cache/adl/diffing_results/ouro_1_4B/ouro_cake_eos1221/activation_difference_lens/recurrence_4'
    for layer in (0,11,23):
        for position in range(5):
            for variant in ('difference','base','ft'):
                path = base/f'layer_{layer}/corpus.jsonl/token_relevance/position_{position}/{variant}/relevance_logitlens_gpt-5-mini.json'
                try:
                    record = read_json(path)
                except json.JSONDecodeError:
                    return False  # a writer is still publishing this record
                if record is None:
                    return False
                assert (record['layer'],record['position'],record['variant'],record['source'],record['target']) == (layer,position,variant,'logitlens','self')
                assert len(record['tokens']) == len(record['labels']) == 20
                assert len(record['grader_responses']) == 3
    starts = sorted((root/'full_runs/adl_3').glob('attempt_*/STARTED.json'))
    assert starts
    command = read_json(starts[-1])['command']
    assert 'diffing.method.token_relevance.grader.permutations=3' in command
    assert 'diffing.method.token_relevance.k_candidate_tokens=20' in command
    return True


def start_condition(repo, root, pool_dir, condition, deadline):
    remaining = (deadline-time.time())/3600
    assert 0 < remaining <= 3.5
    output = pool_dir / condition
    output.mkdir()
    command = [sys.executable, str(repo/'experiments/ouro_campaign/run_bounded.py'),
        '--hours', str(remaining), '--output', str(output/'bounded'), '--',
        sys.executable, str(repo/'experiments/ouro_full_pipeline/run.py'),
        '--root', str(root), '--skip-stage', '--conditions', condition]
    with (output/'runner.log').open('w') as log:
        child = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    record = {'condition': condition, 'runner_pid': child.pid, 'command': command,
        'launched_unix': time.time(), 'deadline_unix': deadline,
        'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()}
    (output/'LAUNCH.json').write_text(json.dumps(record,indent=2))
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/workspace/full_audit_20260915'))
    p.add_argument('--repo', type=Path, default=Path('/workspace/diffing-toolkit-full-20260915-v4'))
    p.add_argument('--hours', type=float, default=3)
    p.add_argument('--adl-repo', type=Path, help='Fixed ADL interface checkout; ADL waits until it exists')
    p.add_argument('--max-workers', type=int, default=6)
    p.add_argument('--max-used-mib', type=int, default=50*1024)
    a = p.parse_args()
    assert 0<a.hours<=3.5 and 1<=a.max_workers<=6
    root=a.root; pool=root/'workers/condition_pool_v1'; pool.mkdir()
    deadline=time.time()+a.hours*3600
    state={'started_unix':time.time(),'deadline_unix':deadline,'max_workers':a.max_workers,
        'pool_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pending':['recurrence_2','jlens_early_1','recurrence_1','jlens_early_0','recurrence_0'],'active':{},'completed':[],'failed':[],'handoffs':{},
        'early_release_reason':'Distinct caches behind an immediate-next NMF condition; legacy workers stop after current condition'}
    legacy={
        'recurrence_full_v1':('recurrence_3',['recurrence_3_nmf']),
        'adl_full_v4':('adl_3',['adl_3','adl_2','adl_1','adl_0']),
        'early_full_v4':('jlens_early_2',['jlens_early_2_nmf']),
    }
    if (root/'workers/jlens_full_v4/LAUNCH.json').exists():
        legacy['jlens_full_v4']=('jlens_2',['jlens_2_nmf'])
        state['pending'].extend(['jlens_1','jlens_0'])
        j_released=True
    else:
        j_released=False
    while time.time()<deadline:
        legacy_active=[]
        for name,(current,remaining) in legacy.items():
            if name in state['handoffs']:continue
            worker=root/'workers'/name
            complete=read_json(root/f'full_runs/{current}/COMPLETE.json')
            ended=read_json(worker/'bounded/WORK_COMPLETE.json')
            adl_stage = name=='adl_full_v4' and a.adl_repo is not None and adl_grades_complete(root)
            if adl_stage or (complete is not None and complete['exit_code']==0) or ended is not None:
                stopped=stop_owned_worker(worker,current)
                reason = 'completed current condition'
                if name=='adl_full_v4':
                    reason='infrastructure tool-interface fix; cached token grades retained; fresh auditor interface'
                    old_complete=root/f'full_runs/{current}/COMPLETE.json'
                    if old_complete.exists():
                        old_complete.rename(old_complete.with_name(f'LEGACY_INTERFACE_COMPLETE_{time.time_ns()}.json'))
                elif complete is None:
                    state['failed'].append({'legacy_worker':name,'condition':current,'terminal':ended,'retry':False})
                    reason='legacy failure preserved; independent remaining conditions released without retrying failed condition'
                receipt={'condition':current,'completed_unix':time.time(),'stopped_job':stopped,'released':remaining,'reason':reason}
                (pool/f'HANDOFF_{name}.json').write_text(json.dumps(receipt,indent=2))
                state['handoffs'][name]=receipt
                state['pending'].extend(remaining)
            else:
                legacy_active.append(current)
        if not j_released:
            complete=read_json(root/'compat_v3/smoke_runs/jlens_2/COMPLETE.json')
            if complete is not None and complete['exit_code']==0:
                state['pending'].extend(['jlens_2','jlens_2_nmf','jlens_1','jlens_0'])
                j_released=True
        for condition,record in list(state['active'].items()):
            terminal=read_json(pool/condition/'bounded/WORK_COMPLETE.json')
            if terminal is not None:
                completed=read_json(root/f'full_runs/{condition}/COMPLETE.json')
                if terminal['exit_code']==0 and completed is not None:
                    state['completed'].append(condition)
                else:
                    state['failed'].append({'condition':condition,'terminal':terminal})
                del state['active'][condition]
        smoke_running=not (root/'workers/smoke_adl_jlens_v3/bounded/WORK_COMPLETE.json').exists()
        occupied=len(legacy_active)+len(state['active'])+int(smoke_running)
        busy={cache_key(c) for c in legacy_active+list(state['active'])}
        if occupied<a.max_workers and deadline-time.time()>300:
            used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if used<a.max_used_mib:
                condition=next((c for c in state['pending'] if cache_key(c) not in busy and (not c.startswith('adl_') or a.adl_repo is None or a.adl_repo.exists())),None)
                if condition is not None:
                    assert not (root/f'full_runs/{condition}/COMPLETE.json').exists()
                    selected_repo = a.adl_repo if condition.startswith('adl_') and a.adl_repo is not None else a.repo
                    state['active'][condition]=start_condition(selected_repo,root,pool,condition,deadline)
                    state['active'][condition]['memory_used_mib_before_launch']=used
                    state['pending'].remove(condition)
        state['updated_unix']=time.time()
        temporary=pool/'STATE.tmp'
        temporary.write_text(json.dumps(state,indent=2));temporary.replace(pool/'STATE.json')
        if not legacy_active and not state['active'] and not state['pending'] and j_released:
            state['finished_unix']=time.time()
            (pool/('COMPLETE.json' if not state['failed'] else 'FAILED.json')).write_text(json.dumps(state,indent=2))
            return
        time.sleep(3)
    state['finished_unix']=time.time()
    (pool/'TIME_LIMIT.json').write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
