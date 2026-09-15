"""Preserve and optionally stop only the explicitly assigned Ouro third pod."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

from preserve import REMOTE

POD_ID = '6qhlmj1jpl0ca3'
POD_NAME = 'CLAUDE_POD_GREG---ouro-full-20260915-third'

QUIESCENCE_CHECK = r"""
import json
from pathlib import Path
campaign=Path('/workspace/third_audit_20260915')
pool=campaign/'workers/external_pool_v1_native_single'
assert any((pool/name).exists() for name in ('COMPLETE.json','FAILED.json','TIME_LIMIT.json')), 'Final pool is not terminal'
groups=set()
for path in (campaign/'workers').rglob('job.json'):
    job=json.loads(path.read_text())
    groups.add(int(job['pgid']))
live=[]
for path in Path('/proc').glob('[0-9]*/stat'):
    try:
        fields=path.read_text().rsplit(')',1)[1].split()
        if fields[0]!='Z' and int(fields[2]) in groups:
            live.append(int(path.parent.name))
    except (FileNotFoundError,ProcessLookupError):
        pass
assert not live, f'Owned worker groups still have live processes: {live}'
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--stop', action='store_true')
    a = p.parse_args()
    release = json.loads(a.release.read_text())
    assert release.get('all_work_finished') is True and release.get('source_pushed') is True
    assert release.get('pod_id') == POD_ID
    ops_dir = Path('/Users/gkocher/Desktop/recurrent-looped-auditing/LlamaFactory/experiments/ouro_organisms/campaigns/control_and_auditing_20260913')
    sys.path.insert(0, str(ops_dir))
    import ops
    session = ops.session()
    ops.identity(session)
    pod = ops.api(session, 'GET', 'pods/' + POD_ID)
    assert pod['id'] == POD_ID and pod['name'] == POD_NAME
    ssh = ['ssh', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=20',
           '-i', str(Path.home()/'.ssh/id_ed25519_runpod_personal'),
           '-p', str(pod['portMappings']['22']), 'root@' + pod['publicIp']]
    subprocess.run(ssh+['python3 -'], input=QUIESCENCE_CHECK, text=True, check=True, timeout=60)
    a.output.mkdir(parents=True, exist_ok=True)
    receipt = a.output/'REMOTE_ARCHIVE.json'
    if receipt.exists():
        info = json.loads(receipt.read_text())
        observed = subprocess.check_output(ssh+['sha256sum ' + shlex.quote(info['path'])], text=True).split()[0]
        assert observed == info['sha256']
    else:
        source = REMOTE.replace('full_audit_20260915', 'third_audit_20260915')
        source = source.replace("'ouro_full_eval_inputs.tar.gz'", "'ouro_full_eval_inputs.tar.gz','h200_handoff_inputs.tar.gz'")
        info = json.loads(subprocess.check_output(
            ssh+['/workspace/toolkit-env/bin/python -'], input=source, text=True, timeout=3600))
        receipt.write_text(json.dumps(info, indent=2)+'\n')
    source = Path(__file__).resolve().parents[1]/'ouro_campaign/parallel_archive_copy.py'
    spec = importlib.util.spec_from_file_location('archive_copy', source)
    copier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copier)
    archive = a.output/'third_audit_20260915_export.tar.zst'
    copier.copy_archive(ssh=ssh, remote_path=info['path'], output=archive,
                       expected_size=info['bytes'], expected_sha256=info['sha256'], streams=4)
    subprocess.run(['zstd', '--test', str(archive)], check=True, timeout=1800)
    record = dict(info, local_path=str(archive), verified_at=ops.now(), pod_id=POD_ID)
    (a.output/'LOCAL_VERIFIED.json').write_text(json.dumps(record, indent=2)+'\n')
    if a.stop:
        subprocess.run(ssh+['python3 -'], input=QUIESCENCE_CHECK, text=True, check=True, timeout=60)
        pod = ops.api(session, 'GET', 'pods/' + POD_ID)
        assert pod['id'] == POD_ID and pod['name'] == POD_NAME
        result = ops.api(session, 'POST', 'pods/' + POD_ID + '/stop')
        assert result['desiredStatus'] == 'EXITED'
        record.update(status=result['desiredStatus'], stopped_at=ops.now(), deleted=False)
        (a.output/'STOPPED.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
