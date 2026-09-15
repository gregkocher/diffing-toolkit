import json,time,subprocess,sys,hashlib,concurrent.futures
from pathlib import Path
sys.path.insert(0,'/Users/gkocher/Desktop/recurrent-looped-auditing/LlamaFactory/experiments/ouro_organisms/campaigns/control_and_auditing_20260913')
import ops
S=Path('/Users/gkocher/Desktop/recurrent-looped-auditing/research/methods_campaign_20260914')
def watch(role):
 m=json.loads((S/(role+'.json')).read_text());s=ops.session();ops.identity(s)
 ssh=['ssh','-o','ConnectTimeout=15','-o','ServerAliveInterval=20','-i',str(Path.home()/'.ssh/id_ed25519_runpod_personal'),'-p',str(m['portMappings']['22']),'root@'+m['publicIp']]
 while True:
  try:
   q=subprocess.run(ssh+['test -f /workspace/methods_v2/WORK_COMPLETE.json'],timeout=30)
   if q.returncode:time.sleep(45);continue
   if not (S/'SOURCE_PUSHED.json').exists():time.sleep(30);continue
   dest=S/'exports'/role;dest.mkdir(parents=True,exist_ok=True)
   archive='/workspace/methods_export.tar'
   subprocess.run(ssh+['tar -cf '+archive+' --exclude=.git --exclude=__pycache__ --exclude=.venv --exclude=openrouter_api_key.txt --exclude=ouro_target_adapter -C /workspace methods_v2 diffing-toolkit/experiments diffing-toolkit/src diffing-toolkit/configs standard_v1/FREEZE.json standard_v1/corpus.jsonl standard_v1/model_paths.json && sha256sum '+archive],timeout=300,check=True,stdout=(dest/'remote_sha256.txt').open('w'))
   subprocess.run(['rsync','-rt','--partial','-e',' '.join(ssh[:-1]),ssh[-1]+':'+archive,str(dest/'methods_export.tar')],timeout=3600,check=True)
   h=hashlib.file_digest((dest/'methods_export.tar').open('rb'),'sha256').hexdigest()
   assert h==(dest/'remote_sha256.txt').read_text().split()[0]
   p=ops.api(s,'GET','pods/'+m['id']);assert p['name']==m['name'] and p['name'].startswith('CLAUDE_POD_GREG---ouro-methods-20260914-')
   stopped=ops.api(s,'POST','pods/'+m['id']+'/stop')
   (dest/'COMPLETE.json').write_text(json.dumps({'id':m['id'],'sha256':h,'bytes':(dest/'methods_export.tar').stat().st_size,'status':stopped.get('desiredStatus'),'finished':ops.now()},indent=2))
   print(role,'ARCHIVED_VERIFIED_STOPPED',flush=True);return
  except Exception as e:
   print(role,type(e).__name__,str(e)[:150],flush=True);time.sleep(45)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(watch,['adl','jlens','recurrence']))
