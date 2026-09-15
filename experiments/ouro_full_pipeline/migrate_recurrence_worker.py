import json,os,signal,subprocess,time
from pathlib import Path
root=Path('/workspace/full_audit_20260915');repo=Path('/workspace/diffing-toolkit-full-20260915-v3');python='/workspace/toolkit-env/bin/python'
old=root/'workers/recurrence_full_v1/bounded/job.json';job=json.loads(old.read_text())
while not (root/'full_runs/recurrence_3/COMPLETE.json').exists():
    if (old.parent/'WORK_COMPLETE.json').exists():
        raise SystemExit('Old worker ended before successful recurrence_3')
    if time.time()>job['deadline_unix']:raise SystemExit('Original deadline elapsed')
    time.sleep(2)
proc=Path(f"/proc/{job['pid']}/stat")
if proc.exists():
    assert proc.read_text().split()[21]==job['proc_start_ticks']
    assert os.getpgid(job['pid'])==job['pgid']
    os.killpg(job['pgid'],signal.SIGTERM)
    for _ in range(50):
        if (old.parent/'WORK_COMPLETE.json').exists():break
        time.sleep(.2)
remaining=(job['deadline_unix']-time.time())/3600
assert remaining>0
out=root/'workers/recurrence_full_v3';out.mkdir(parents=True)
cmd=[python,str(repo/'experiments/ouro_campaign/run_bounded.py'),'--hours',str(remaining),'--output',str(out/'bounded'),'--',python,str(repo/'experiments/ouro_full_pipeline/run.py'),'--skip-stage','--conditions','recurrence_3_nmf','recurrence_2','recurrence_1','recurrence_0']
with (out/'runner.log').open('w') as log:p=subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record={'runner_pid':p.pid,'command':cmd,'original_deadline_unix':job['deadline_unix'],'migration_unix':time.time(),'reason':'public async backend initialization before cached-only grading path','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()}
(out/'LAUNCH.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
