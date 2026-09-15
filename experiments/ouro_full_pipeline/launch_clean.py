"""Start a fresh native-generation protocol campaign while preserving old results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--conditions',nargs='+',required=True);a=p.parse_args()
assert 'native_single_prompt_v1' in (a.repo/'configs/experiment/ouro_full_evaluation.yaml').read_text()
for proc in Path('/proc').glob('[0-9]*/cmdline'):
    try: command=proc.read_bytes().split(b'\0')
    except FileNotFoundError: continue
    assert not any(b'diffing-toolkit-full-20260915' in x and x.endswith(b'/main.py') for x in command), 'A prior native worker is still running'
stamp=time.time_ns(); legacy=[]
for condition in a.conditions:
    path=a.root/f'full_runs/{condition}/COMPLETE.json'
    if path.exists():
        saved=path.with_name(f'LEGACY_GENERATION_COMPLETE_{stamp}.json');path.rename(saved);legacy.append(str(saved))
(a.root/'LEGACY_GENERATION_SUPERSEDED.json').write_text(json.dumps({'protocol':'native_single_prompt_v1','legacy_complete_markers':legacy,'conditions':a.conditions,'time_ns':stamp},indent=2))
cmd=[sys.executable,str(a.repo/'experiments/ouro_full_pipeline/external_pool.py'),'--root',str(a.root),'--repo',str(a.repo),'--conditions',*a.conditions,'--max-workers','3','--hours','3','--pool-name','external_pool_clean_generation_v1']
with (a.root/'CLEAN_GENERATION_POOL.log').open('x') as log:
    child=subprocess.Popen(cmd,cwd=a.repo,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record={'pid':child.pid,'command':cmd,'launched_unix':time.time(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip(),'protocol':'native_single_prompt_v1'}
(a.root/'CLEAN_GENERATION_POOL_LAUNCH.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
