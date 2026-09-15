"""Integrity and resume behavior of bounded immutable-archive range copying."""
import hashlib
import importlib.util
from pathlib import Path
import shlex
import subprocess
import threading
import types
import pytest


def module():
    path = Path(__file__).parents[1] / 'experiments/ouro_campaign/parallel_archive_copy.py'
    spec = importlib.util.spec_from_file_location('test_parallel_archive_copy', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_ranges_reassemble_exactly_and_preserve_prefix(monkeypatch, tmp_path):
    helper = module()
    data = bytes(range(256))*4+b'last'
    prefix = tmp_path/'original_rsync_partial'
    prefix.write_bytes(data[:300])
    calls = []
    def run(command, *, stdout, **kwargs):
        fields = shlex.split(command[-1])
        offset, count = map(int, fields[-2:])
        calls.append((offset, count))
        stdout.write(data[offset:offset+count])
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(helper.subprocess, 'run', run)
    out = tmp_path/'verified.tar.zst'
    receipt = helper.copy_archive(ssh=['ssh','pod'], remote_path='/archive with spaces',
        output=out, expected_size=len(data), expected_sha256=hashlib.sha256(data).hexdigest(),
        streams=4, prefix=prefix)
    assert out.read_bytes() == data
    assert prefix.read_bytes() == data[:300]
    assert receipt['seeded_prefix_bytes'] == 300
    assert sum(count for _, count in calls) == len(data)-300
    assert min(offset for offset, _ in calls) == 300


def test_interrupted_ranges_resume_without_redownloading_saved_bytes(monkeypatch, tmp_path):
    helper = module()
    data = bytes(range(251))*8
    lock = threading.Lock()
    failed = False
    calls = []
    def run(command, *, stdout, **kwargs):
        nonlocal failed
        offset, count = map(int, shlex.split(command[-1])[-2:])
        with lock:
            interrupt = offset == 0 and not failed
            if interrupt: failed = True
            calls.append((offset, count))
        if interrupt:
            stdout.write(data[:count//2])
            raise subprocess.CalledProcessError(255, command)
        stdout.write(data[offset:offset+count])
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(helper.subprocess, 'run', run)
    kwargs = dict(ssh=['ssh','pod'], remote_path='/archive', output=tmp_path/'result',
        expected_size=len(data), expected_sha256=hashlib.sha256(data).hexdigest(), streams=4)
    with pytest.raises(subprocess.CalledProcessError): helper.copy_archive(**kwargs)
    before = len(calls)
    helper.copy_archive(**kwargs)
    assert kwargs['output'].read_bytes() == data
    resumed = calls[before:]
    assert len(resumed) == 1
    assert resumed[0][0] == (len(data)//4)//2


def test_wrong_whole_hash_never_publishes_final_archive(monkeypatch, tmp_path):
    helper = module()
    def run(command, *, stdout, **kwargs):
        count = int(shlex.split(command[-1])[-1])
        stdout.write(b'x'*count)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(helper.subprocess, 'run', run)
    out = tmp_path/'result'
    with pytest.raises(RuntimeError, match='verification failed'):
        helper.copy_archive(ssh=['ssh','pod'], remote_path='/archive', output=out,
            expected_size=128, expected_sha256=hashlib.sha256(b'y'*128).hexdigest())
    assert not out.exists()
    assert list(tmp_path.glob('result.assembling.*'))
    assert list((tmp_path/'result.parts').glob('part_*'))
