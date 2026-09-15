"""Run an explicitly assigned list of toolkit conditions on a separate pod."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--conditions',nargs='+',required=True)
p.add_argument('--hours',type=float,default=3)
p.add_argument('--max-workers',type=int,default=3)
p.add_argument('--resume',action='store_true')
p.add_argument('--exclude-condition',action='append',default=[])
p.add_argument('--pool-name',default='external_pool_v1',help='New scheduler directory; preserves every prior pool')
a=p.parse_args()
assert 0<a.hours<=3 and 1<=a.max_workers<=3
sys.path.insert(0,str(a.repo/'experiments/ouro_full_pipeline'))
from pool import start_condition,read_json,cache_key
assert a.pool_name.startswith('external_pool_') and '/' not in a.pool_name
out=a.root/'workers'/a.pool_name
if not a.resume:out.mkdir(parents=True,exist_ok=False)
state={'pending':list(a.conditions),'active':{},'completed':[],'failed':[],
       'started_unix':time.time(),'deadline_unix':time.time()+a.hours*3600,
       'max_workers':a.max_workers,'scheduler_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'source_commit':subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip()}
if a.resume:
 state=read_json(out/'STATE.json')
 assert state is not None
 state.setdefault('resumes',[]).append({'unix':time.time(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
if a.exclude_condition:
 assert all(c in state['pending'] and c not in state['active'] for c in a.exclude_condition), 'Only untouched pending conditions may be handed off'
 state['pending']=[c for c in state['pending'] if c not in a.exclude_condition]
 state.setdefault('external_handoffs',[]).append({'conditions':a.exclude_condition,'unix':time.time(),'reason':'Explicit assignment to another GPU; no active work interrupted'})
while time.time()<state['deadline_unix']:
 for condition in list(state['active']):
  terminal=read_json(out/condition/'bounded/WORK_COMPLETE.json')
  if terminal is not None:
   done=read_json(a.root/f'full_runs/{condition}/COMPLETE.json')
   if terminal['exit_code']==0 and done is not None:state['completed'].append(condition)
   else:state['failed'].append({'condition':condition,'terminal':terminal})
   del state['active'][condition]
 if state['pending'] and len(state['active'])<a.max_workers and state['deadline_unix']-time.time()>300:
  used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
  if used<100*1024:
   busy={cache_key(c) for c in state['active']}
   condition=next((c for c in state['pending'] if cache_key(c) not in busy),None)
   if condition:
    state['active'][condition]=start_condition(a.repo,a.root,out,condition,state['deadline_unix'])
    state['active'][condition]['memory_used_mib_before_launch']=used
    state['pending'].remove(condition)
 state['updated_unix']=time.time()
 temp=out/'STATE.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(out/'STATE.json')
 if not state['pending'] and not state['active']:
  state['finished_unix']=time.time();(out/('FAILED.json' if state['failed'] else 'COMPLETE.json')).write_text(json.dumps(state,indent=2));break
 time.sleep(3)
else:
 (out/'TIME_LIMIT.json').write_text(json.dumps(state,indent=2))
