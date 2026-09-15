"""Archive new evaluation artifacts before stopping one explicitly named pod.

Run locally only after all campaign workloads are finished. Never deletes pods
or remote files. Existing archives are immutable and verified before reuse.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

REMOTE = r'''
import hashlib,json,subprocess,tarfile
from pathlib import Path
import zstandard
root=Path('/workspace'); repo=root/'diffing-toolkit-full-20260915'
out=root/'full_audit_20260915'; archive=root/'full_audit_20260915_export.tar.zst'
if archive.exists():
 raise FileExistsError('Export already exists without local receipt; inspect before retry')
refs={'head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
      'status':subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True), 'sha256':{}}
for name in ['src','configs','experiments','tests','main.py','pyproject.toml','uv.lock']:
 p=repo/name
 for f in (p.rglob('*') if p.is_dir() else [p]):
  if f.is_file() and not f.is_symlink() and '__pycache__' not in f.parts:
   refs['sha256'][str(f.relative_to(repo))]=hashlib.file_digest(f.open('rb'),'sha256').hexdigest()
(out/'FINAL_SOURCE_PROVENANCE.json').write_text(json.dumps(refs,indent=2))
def keep(info):
 parts=Path(info.name).parts
 if info.issym() or info.islnk(): return None
 if any(p in {'.git','.venv','__pycache__','ouro_target_adapter','openrouter_api_key.txt','openai_api_key.txt'} for p in parts): return None
 # These are imported copies of previously verified local archives, not new outputs.
 if len(parts)>1 and parts[0]=='full_audit_20260915' and parts[1] in {'imports','methods_export.tar','ouro_full_eval_inputs.tar.gz'}: return None
 return info
entries=['full_audit_20260915','standard_v1/FREEZE.json','standard_v1/corpus.jsonl','standard_v1/model_paths.json']
entries += ['diffing-toolkit-full-20260915/'+p for p in ['src','configs','experiments','tests','main.py','pyproject.toml','uv.lock']]
with archive.open('xb') as raw:
 with zstandard.ZstdCompressor(level=3,threads=2).stream_writer(raw,closefd=False) as stream:
  with tarfile.open(fileobj=stream,mode='w|') as tar:
   for entry in entries:
    if (root/entry).exists(): tar.add(root/entry,arcname=entry,filter=keep)
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()}))
'''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pod-json',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--release',type=Path,required=True,help='Explicit local release JSON: all_work_finished=true, source_pushed=true')
    p.add_argument('--stop',action='store_true')
    a=p.parse_args()
    release=json.loads(a.release.read_text())
    assert release.get('all_work_finished') is True and release.get('source_pushed') is True
    meta=json.loads(a.pod_json.read_text())
    assert meta['id']=='bi70tcnhquq7xh' and meta['name']=='CLAUDE_POD_GREG---ouro-methods-20260914-recurrence'
    ops_dir=Path('/Users/gkocher/Desktop/recurrent-looped-auditing/LlamaFactory/experiments/ouro_organisms/campaigns/control_and_auditing_20260913')
    sys.path.insert(0,str(ops_dir)); import ops
    session=ops.session(); ops.identity(session)
    pod=ops.api(session,'GET','pods/'+meta['id'])
    assert pod['name']==meta['name']
    ssh=['ssh','-o','ConnectTimeout=15','-o','ServerAliveInterval=20','-i',str(Path.home()/'.ssh/id_ed25519_runpod_personal'),'-p',str(pod['portMappings']['22']),'root@'+pod['publicIp']]
    a.output.mkdir(parents=True,exist_ok=True)
    receipt=a.output/'REMOTE_ARCHIVE.json'
    if receipt.exists():
        info=json.loads(receipt.read_text())
        observed=subprocess.check_output(ssh+['sha256sum '+shlex.quote(info['path'])],text=True).split()[0]
        assert observed==info['sha256']
    else:
        info=json.loads(subprocess.check_output(ssh+['/workspace/toolkit-env/bin/python -'],input=REMOTE,text=True,timeout=3600))
        receipt.write_text(json.dumps(info,indent=2))
    source=Path(__file__).resolve().parents[1]/'ouro_campaign/parallel_archive_copy.py'
    spec=importlib.util.spec_from_file_location('archive_copy',source)
    copier=importlib.util.module_from_spec(spec); spec.loader.exec_module(copier)
    archive=a.output/'full_audit_20260915_export.tar.zst'
    copier.copy_archive(ssh=ssh,remote_path=info['path'],output=archive,expected_size=info['bytes'],expected_sha256=info['sha256'],streams=4)
    subprocess.run(['zstd','--test',str(archive)],check=True,timeout=1800)
    record=dict(info,local_path=str(archive),verified_at=ops.now())
    (a.output/'LOCAL_VERIFIED.json').write_text(json.dumps(record,indent=2))
    if a.stop:
        pod=ops.api(session,'GET','pods/'+meta['id']); assert pod['name']==meta['name']
        result=ops.api(session,'POST','pods/'+meta['id']+'/stop')
        record.update(pod_id=meta['id'],status=result.get('desiredStatus'),stopped_at=ops.now())
        (a.output/'STOPPED.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)

if __name__=='__main__': main()
