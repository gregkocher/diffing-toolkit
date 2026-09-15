"""Preserve a complete lens fit in a newly created private model repository.

Creation uses exist_ok=False: this utility cannot write into any existing HF
repository. It uploads all fit files, records their SHA256, and verifies every
remote LFS SHA256 or Git blob SHA1 plus the exact file list and privacy flag.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi

def git_blob_sha1(data: bytes) -> str:
    header = b'blob ' + str(len(data)).encode('ascii') + bytes([0])
    return hashlib.sha1(header + data).hexdigest()

p = argparse.ArgumentParser()
p.add_argument('--fit', type=Path, required=True)
p.add_argument('--repo-id', required=True)
p.add_argument('--token-file', type=Path, default=Path('/root/.hf_token'))
p.add_argument('--receipt', type=Path, required=True)
a = p.parse_args()
if a.receipt.exists():
    raise FileExistsError('Upload receipt already exists; refusing to overwrite it')
if not a.repo_id.startswith('wasd12345/') or a.repo_id.count('/') != 1:
    raise ValueError('New lens repository must belong to wasd12345')
required = ['COMPLETE.json', 'fit_identity.json', 'fit_prompts.json',
            'parity.json', 'extractor_parity.json', 'convergence.json', 'disjointness.json']
for name in required:
    if not (a.fit/name).is_file():
        raise FileNotFoundError(a.fit/name)
certificate=json.loads((a.fit/'COMPLETE.json').read_text())
n=certificate['n_prompts']
if n != certificate['settings']['max_prompts']:
    raise ValueError('Fit stopped before requested prompt count')
for name in [f'prompt_{i:03d}.pt' for i in range(n)] + [f'lens_n{n}.pt'] + [f'loop{r}_n{n}.pt' for r in (1,2,3)]:
    if not (a.fit/name).is_file():
        raise FileNotFoundError(a.fit/name)
records = []
for path in sorted(a.fit.rglob('*')):
    if path.is_symlink():
        raise ValueError(f'Symlinks are not allowed in fit uploads: {path}')
    if not path.is_file():
        continue
    if path.suffix not in ('.pt', '.json', '.md', '.log', '.pdf'):
        raise ValueError(f'Unexpected file type in fit outputs: {path.name}')
    if any(term in path.name.lower() for term in ('api_key', 'hf_token', 'secret', 'credential')):
        raise ValueError(f'Credential-like filename in fit outputs: {path.name}')
    data = path.read_bytes()
    records.append({'path':path.relative_to(a.fit).as_posix(), 'bytes':len(data),
        'sha256':hashlib.sha256(data).hexdigest(),
        'git_blob_sha1':git_blob_sha1(data)})
manifest = a.fit/'UPLOAD_MANIFEST.json'
if manifest.exists():
    raise FileExistsError('Upload manifest already exists; inspect prior upload first')
manifest.write_text(json.dumps({'files':records},indent=2)+'\n')
data=manifest.read_bytes()
records.append({'path':manifest.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
    'git_blob_sha1':git_blob_sha1(data)})
api=HfApi(token=os.environ.get('HF_TOKEN') or a.token_file.read_text().strip())
identity=api.whoami()
if identity['name'] != 'wasd12345':
    raise ValueError('HF token belongs to a different user')
creation_receipt=a.receipt.with_name(a.receipt.name+'.created.json')
if creation_receipt.exists():
    raise FileExistsError('Repository creation receipt exists; inspect previous upload first')
api.create_repo(repo_id=a.repo_id, repo_type='model', private=True, exist_ok=False)
creation_receipt.parent.mkdir(parents=True,exist_ok=True)
creation_receipt.write_text(json.dumps({'repo_id':a.repo_id,'private':True,
    'status':'created_pending_upload','fit_directory':str(a.fit.resolve()),
    'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()},indent=2)+'\n')
commit=api.upload_folder(repo_id=a.repo_id, repo_type='model', folder_path=a.fit,
    commit_message='Preserve matched early-position Ouro Jacobian lenses and fit provenance')
info=api.model_info(a.repo_id,revision=commit.oid,files_metadata=True)
assert info.private, 'Repository is not private'
remote={x.rfilename:x for x in info.siblings}
assert set(remote)-{'.gitattributes'} == {x['path'] for x in records}
for entry in records:
    item=remote[entry['path']]
    assert item.size==entry['bytes'], entry['path']
    if item.lfs:
        assert item.lfs.sha256==entry['sha256'], entry['path']
    else:
        assert item.blob_id==entry['git_blob_sha1'], entry['path']
a.receipt.parent.mkdir(parents=True,exist_ok=True)
a.receipt.write_text(json.dumps({'repo_id':a.repo_id,'private':True,'revision':commit.oid,
    'files_verified':len(records),'total_bytes':sum(x['bytes'] for x in records),
    'verification':'remote LFS SHA256 or Git blob SHA1 and size for every file',
    'files':records},indent=2)+'\n')
print(json.dumps({'repo_id':a.repo_id,'revision':commit.oid,'files_verified':len(records)}))
