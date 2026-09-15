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
