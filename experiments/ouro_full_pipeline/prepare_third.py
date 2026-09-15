"""Verify frozen inputs and prepare pinned models on the third Ouro GPU."""
from pathlib import Path
import os,json,hashlib,tarfile,shutil,subprocess
r=Path('/workspace/third_audit_20260915');a=r/'third_handoff_inputs.tar.gz'
assert hashlib.sha256(a.read_bytes()).hexdigest()=='3323af3f4a4597f4a8b24d982d6684d52297cdda0c2a59a84c2c49ba378d4045'
with tarfile.open(a) as t:t.extractall(r,filter='data')
for name in ['FREEZE.json','corpus.jsonl']:
 p=Path('/workspace/standard_v1')/name;assert not p.exists();shutil.copy2(r/'standard_v1'/name,p)
repo=Path('/workspace/diffing-toolkit-full-20260915-third')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip().startswith('6aa4123')
env=os.environ.copy();env.update(HF_HOME='/workspace/hf',HF_TOKEN=Path('/root/.hf_token').read_text().strip(),PYTHONPATH=str(repo/'src'),OURO_OPENAI_KEY_PATH='/root/.ouro_full_openai_key',MPLBACKEND='Agg')
cmd=['/workspace/toolkit-env/bin/python',str(repo/'experiments/ouro_standard_diff_mining/prepare.py'),'--freeze','/workspace/standard_v1/FREEZE.json','--output','/workspace/standard_v1']
with (r/'PREPARE.log').open('x') as f:subprocess.run(cmd,env=env,cwd=repo,stdout=f,stderr=subprocess.STDOUT,check=True)
manifest={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'input_archive_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'conditions':['jlens_1','jlens_0','jlens_early_0'],'cached_json_counts':{c:len(list((r/'cache'/c).rglob('*.json'))) for c in ['jlens_1','jlens_0','jlens_early_0']}}
(r/'PREPARED.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
