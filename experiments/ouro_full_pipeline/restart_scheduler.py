"""Resume only an owned scheduler, preserving workers, deadline, and queue history."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--receipt',type=Path,required=True)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--retry',nargs='*',default=[])
a=p.parse_args()
r=a.root; receipt=json.loads(a.receipt.read_text()); pid=receipt['pid']
actual=Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').split(b'\0')
assert [x.decode() for x in actual]==receipt['command']
assert any(x.endswith('/experiments/ouro_full_pipeline/pool.py') for x in receipt['command'])
start=Path(f'/proc/{pid}/stat').read_text().split()[21]
os.kill(pid,signal.SIGTERM)
for _ in range(200):
    stat=Path(f'/proc/{pid}/stat')
    if not stat.exists() or stat.read_text().split()[2]=='Z':break
    time.sleep(.05)
else:raise RuntimeError('Owned scheduler did not stop')
state_path=r/'workers/condition_pool_v1/STATE.json'
state=json.loads(state_path.read_text());stamp=time.time_ns()
(r/f'SCHEDULER_STATE_BEFORE_{stamp}.json').write_text(json.dumps(state,indent=2))
for condition in a.retry:
    assert condition not in state['active'] and condition not in state['completed']
    assert not (r/f'full_runs/{condition}/COMPLETE.json').exists()
    assert list((r/f'full_runs/{condition}').glob('attempt_*/FAILED.json'))
    if condition not in state['pending']:state['pending'].append(condition)
state['pending'].sort(key=lambda x: (not x.endswith('_nmf'), x not in a.retry))
state.setdefault('queue_adjustments',[]).append({'time_ns':stamp,'reason':'Prioritize matched NMF and explicitly retry pre-cap budget assertion once','retry':a.retry})
state_path.write_text(json.dumps(state,indent=2))
cmd=[sys.executable,str(a.repo/'experiments/ouro_full_pipeline/pool.py'),'--root',str(r),'--repo',str(a.repo),'--adl-repo',str(r/'UNPUBLISHED_ADL_GATE'),'--resume','--external-assignment',str(r/'EXTERNAL_ASSIGNMENT.json'),'--max-workers','3']
with (r/f'condition_pool_resume_{stamp}.log').open('x') as f:
    child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record={'pid':child.pid,'command':cmd,'launched_unix':time.time(),'prior_pid':pid,'prior_start_ticks':start,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip()}
(r/f'CONDITION_POOL_RESUME_{stamp}.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
