"""Safety gates for the campaign's archive-before-stop watcher."""
import importlib.util
from pathlib import Path
import sys
import types


def load_guard(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'ops', types.ModuleType('ops'))
    source = Path(__file__).parents[1] / 'experiments/ouro_campaign/archive_and_stop.py'
    spec = importlib.util.spec_from_file_location('ouro_test_guard', source)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.S = tmp_path
    return guard


def test_jlens_pilot_cannot_trigger_closure_without_release(monkeypatch, tmp_path):
    guard = load_guard(monkeypatch, tmp_path)
    assert not guard.may_close('jlens')
    (tmp_path / 'JLENS_RELEASE.json').write_text('{}')
    assert guard.may_close('jlens')


def test_completed_role_is_never_closed_again(monkeypatch, tmp_path):
    guard = load_guard(monkeypatch, tmp_path)
    complete = tmp_path / 'exports/jlens/COMPLETE.json'
    complete.parent.mkdir(parents=True)
    complete.write_text('{}')
    (tmp_path / 'JLENS_RELEASE.json').write_text('{}')
    assert not guard.may_close('jlens')


def test_recurrence_does_not_depend_on_jlens_release(monkeypatch, tmp_path):
    guard = load_guard(monkeypatch, tmp_path)
    assert guard.may_close('recurrence')


def test_existing_archive_is_verified_and_reused(monkeypatch, tmp_path):
    guard = load_guard(monkeypatch, tmp_path)
    digest = 'a' * 64
    receipt = tmp_path / 'remote_sha256.txt'
    receipt.write_text(digest + '  /workspace/methods_export.tar.zst\n')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return types.SimpleNamespace(stdout=digest + '  /workspace/methods_export.tar.zst\n')
    monkeypatch.setattr(guard.subprocess, 'run', run)
    assert guard.ensure_remote_archive(['ssh', 'pod'], tmp_path, 'b' * 40) == digest
    assert calls == [['ssh', 'pod', 'sha256sum /workspace/methods_export.tar.zst']]
    assert receipt.read_text().startswith(digest)


def test_changed_archive_is_not_silently_rebuilt(monkeypatch, tmp_path):
    import pytest
    guard = load_guard(monkeypatch, tmp_path)
    receipt = tmp_path / 'remote_sha256.txt'
    receipt.write_text('a' * 64 + '  /workspace/methods_export.tar.zst\n')
    monkeypatch.setattr(guard.subprocess, 'run', lambda *args, **kwargs: types.SimpleNamespace(stdout='b' * 64 + '  /workspace/methods_export.tar.zst\n'))
    with pytest.raises(RuntimeError, match='differs from its receipt'):
        guard.ensure_remote_archive(['ssh', 'pod'], tmp_path, 'c' * 40)
    assert receipt.read_text().startswith('a' * 64)


def test_new_archive_receipt_is_published_only_after_export(monkeypatch, tmp_path):
    guard = load_guard(monkeypatch, tmp_path)
    digest = 'c' * 64
    def run(command, **kwargs):
        assert not (tmp_path / 'remote_sha256.txt').exists()
        kwargs['stdout'].write(digest + '  /workspace/methods_export.tar.zst\n')
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(guard.subprocess, 'run', run)
    assert guard.ensure_remote_archive(['ssh', 'pod'], tmp_path, 'd' * 40) == digest
    assert (tmp_path / 'remote_sha256.txt').read_text().startswith(digest)
