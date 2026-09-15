import json,subprocess,time
from pathlib import Path
root=Path('/workspace/full_audit_20260915');repo=Path('/workspace/diffing-toolkit-full-20260915-v4');python='/workspace/toolkit-env/bin/python'
started=set()
choices={'adl':('adl_3',['adl_3','adl_2','adl_1','adl_0']),'jlens':('jlens_2',['jlens_2','jlens_2_nmf','jlens_1','jlens_0'])}
while len(started)<len(choices):
    for name,(condition,conditions) in choices.items():
        if name in started:continue
        complete=root/f'compat_v3/smoke_runs/{condition}/COMPLETE.json'
        if not complete.exists():continue
        assert json.loads(complete.read_text())['exit_code']==0
        used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used>50*1024:continue
        out=root/f'workers/{name}_full_v4';out.mkdir(parents=True)
        cmd=[python,str(repo/'experiments/ouro_campaign/run_bounded.py'),'--hours','3','--output',str(out/'bounded'),'--',python,str(repo/'experiments/ouro_full_pipeline/run.py'),'--skip-stage','--conditions']+conditions
        with (out/'runner.log').open('w') as log:p=subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        rec={'runner_pid':p.pid,'command':cmd,'launched_unix':time.time(),'smoke_gate':str(complete),'memory_used_mib_before_launch':used,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()}
        (out/'LAUNCH.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True);started.add(name)
    if len(started)<len(choices):
        terminal=root/'workers/smoke_adl_jlens_v3/bounded/WORK_COMPLETE.json'
        if terminal.exists() and json.loads(terminal.read_text())['exit_code']!=0:
            raise SystemExit('Smoke worker failed; no further full launches')
        time.sleep(5)
