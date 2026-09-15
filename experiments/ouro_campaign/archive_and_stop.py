"""Verify compressed campaign exports before stopping explicitly owned pods."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, '/Users/gkocher/Desktop/recurrent-looped-auditing/LlamaFactory/experiments/ouro_organisms/campaigns/control_and_auditing_20260913')
import ops
S = Path('/Users/gkocher/Desktop/recurrent-looped-auditing/research/methods_campaign_20260914')

REMOTE_EXPORT = r'''
import hashlib,json,subprocess,tarfile,sys
from pathlib import Path
import zstandard
root=Path('/workspace')
refs={'local_pushed_diffing_toolkit_commit':sys.argv[1]}
for name in ('diffing-toolkit','jacobian-lens'):
 p=root/name
 if p.exists():
  result=subprocess.run(['git','-C',str(p),'rev-parse','HEAD'],capture_output=True,text=True)
  status=subprocess.run(['git','-C',str(p),'status','--porcelain'],capture_output=True,text=True)
  hashes={}
  relative_entries=('src','configs','experiments','main.py','pyproject.toml','uv.lock') if name=='diffing-toolkit' else ('jlens','pyproject.toml','uv.lock')
  for relative in relative_entries:
   target=p/relative
   files=target.rglob('*') if target.is_dir() else [target]
   for file in files:
    if file.is_file() and not file.is_symlink() and '__pycache__' not in file.parts and file.name!='openrouter_api_key.txt':
     with file.open('rb') as handle: hashes[str(file.relative_to(p))]=hashlib.file_digest(handle,'sha256').hexdigest()
  refs[name]={'clone_head':result.stdout.strip(),'clone_head_returncode':result.returncode,'working_tree_status_porcelain':status.stdout,'status_returncode':status.returncode,'actual_file_sha256':dict(sorted(hashes.items()))}
(root/'methods_v2/source_reference_commits.json').write_text(json.dumps(refs,indent=2)+'\n')
entries=['methods_v2','diffing-toolkit/experiments','diffing-toolkit/src','diffing-toolkit/configs','diffing-toolkit/main.py','diffing-toolkit/pyproject.toml','diffing-toolkit/uv.lock','standard_v1/FREEZE.json','standard_v1/corpus.jsonl','standard_v1/model_paths.json']
excluded={'.git','__pycache__','.venv','openrouter_api_key.txt','ouro_target_adapter'}
def keep(info):
 return None if any(part in excluded for part in Path(info.name).parts) else info
archive=root/'methods_export.tar.zst'
with archive.open('wb') as raw:
 with zstandard.ZstdCompressor(level=3,threads=2).stream_writer(raw,closefd=False) as stream:
  with tarfile.open(fileobj=stream,mode='w|') as tar:
   for entry in entries:
    if (root/entry).exists(): tar.add(root/entry,arcname=entry,filter=keep)
with archive.open('rb') as source: digest=hashlib.file_digest(source,'sha256').hexdigest()
print(digest+'  '+str(archive))
'''

def may_close(role):
    if (S / 'exports' / role / 'COMPLETE.json').exists():
        return False
    return role != 'jlens' or (S / 'JLENS_RELEASE.json').exists()

def watch(role):
    dest = S / 'exports' / role
    if (dest / 'COMPLETE.json').exists():
        print(role, 'ALREADY_COMPLETE', flush=True)
        return
    metadata = json.loads((S / (role + '.json')).read_text())
    session = ops.session()
    ops.identity(session)
    ssh = ['ssh', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=20', '-i', str(Path.home()/'.ssh/id_ed25519_runpod_personal'), '-p', str(metadata['portMappings']['22']), 'root@'+metadata['publicIp']]
    while True:
        try:
            if (dest/'COMPLETE.json').exists(): return
            if not may_close(role):
                time.sleep(15)
                continue
            check = subprocess.run(ssh+['test -f /workspace/methods_v2/WORK_COMPLETE.json'], timeout=30)
            if check.returncode or not (S/'SOURCE_PUSHED.json').exists():
                time.sleep(30)
                continue
            # Recheck release immediately before any export, including after SSH.
            if not may_close(role): continue
            dest.mkdir(parents=True, exist_ok=True)
            with (dest/'remote_sha256.txt').open('w') as receipt:
                pushed_commit=json.loads((S/'SOURCE_PUSHED.json').read_text())['commit']
                command=shlex.join(['/workspace/toolkit-env/bin/python','-',pushed_commit])
                subprocess.run(ssh+[command], input=REMOTE_EXPORT, text=True, timeout=1800, check=True, stdout=receipt)
            archive = dest/'methods_export.tar.zst'
            subprocess.run(['rsync','-rt','--partial','-e',shlex.join(ssh[:-1]),ssh[-1]+':/workspace/methods_export.tar.zst',str(archive)],timeout=3600,check=True)
            with archive.open('rb') as file:
                digest=hashlib.file_digest(file,'sha256').hexdigest()
            assert digest==(dest/'remote_sha256.txt').read_text().split()[0]
            subprocess.run(['zstd','--test',str(archive)],timeout=1800,check=True)
            pod=ops.api(session,'GET','pods/'+metadata['id'])
            assert pod['name']==metadata['name'] and pod['name'].startswith('CLAUDE_POD_GREG---ouro-methods-20260914-')
            if not may_close(role):
                time.sleep(15)
                continue
            stopped=ops.api(session,'POST','pods/'+metadata['id']+'/stop')
            (dest/'COMPLETE.json').write_text(json.dumps({'id':metadata['id'],'archive':archive.name,'sha256':digest,'bytes':archive.stat().st_size,'status':stopped.get('desiredStatus'),'finished':ops.now()},indent=2)+'\n')
            print(role,'ARCHIVED_VERIFIED_STOPPED',flush=True)
            return
        except Exception as error:
            print(role,type(error).__name__,str(error)[:200],flush=True)
            time.sleep(30)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--activation-file',type=Path)
    parser.add_argument('--ready-file',type=Path)
    args=parser.parse_args()
    if args.ready_file: args.ready_file.write_text('ready\n')
    while args.activation_file and not args.activation_file.exists(): time.sleep(.1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(watch,['adl','jlens','recurrence']))

if __name__=='__main__': main()
