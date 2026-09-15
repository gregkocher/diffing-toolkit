"""CPU-only regressions for real Hydra configs and extraction-directory routing."""
import ast
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).parents[1]


def routing_function():
    # Isolate pure path routing from model/GPU imports.
    source = ROOT / "src/diffing/methods/diff_mining/agent_tools.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_method_dir_for_extraction")
    scope = {"Path": Path, "re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope[node.name]


@pytest.mark.parametrize("suffix", [
    "logits", "recurrence_logits_recurrence_0", "recurrence_logits_recurrence_3",
    "jlens_recurrence_2_layer_1p0", "jlens_recurrence_1_paired_layer_1p0",
])
def test_current_overview_preserves_exact_extraction(tmp_path, suffix):
    directory = tmp_path / ("diff_mining_1024samples_64tokens_100topk_logit_extraction_" + suffix)
    directory.mkdir()
    assert routing_function()(base_results_dir=directory, extraction_method="current", extraction_layer=None) == directory


def test_explicit_recurrence_routing(tmp_path):
    stem = "diff_mining_1024samples_64tokens_100topk_logit_extraction_"
    base = tmp_path / (stem + "jlens_recurrence_2_paired_layer_1p0")
    target = tmp_path / (stem + "recurrence_logits_recurrence_3")
    target.mkdir()
    assert routing_function()(base_results_dir=base, extraction_method="recurrence_logits", extraction_layer=None, recurrence_idx=3) == target


@pytest.mark.parametrize("experiment,method", [("ouro_diff_mining_full", "diff_mining"), ("ouro_adl_full", "activation_difference_lens")])
def test_real_full_config_composes(monkeypatch, experiment, method):
    monkeypatch.setenv("OURO_BASE_PATH", "/tmp/base-snapshot")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["model=ouro_1_4B", "organism=ouro_cake_eos1221", "infrastructure=runpod", f"diffing/method={method}", f"+experiment={experiment}"])
    assert cfg.pipeline.mode == "full"
    assert cfg.diffing.method.token_relevance.enabled
    assert not cfg.diffing.evaluation.agent.ask_model.use_vllm
    assert cfg.diffing.evaluation.agent.hints == ""
    assert cfg.organism.type in cfg.diffing.grading_rubrics
    assert "450" in cfg.organism.description_long and "hard-frozen" in cfg.organism.description_long
    # Ground truth is for graders, not auditor prompts or overview config.
    auditor = OmegaConf.to_yaml(cfg.diffing.evaluation.agent) + OmegaConf.to_yaml(cfg.diffing.method.agent)
    assert "450" not in auditor and "frozen" not in auditor and "cake" not in auditor.lower()
    if method == "diff_mining":
        assert cfg.diffing.method.agent.overview.extraction_method == "current"
    else:
        assert all(task.source == "logitlens" for task in cfg.diffing.method.token_relevance.tasks)


def test_openai_provider_override_composes(monkeypatch):
    monkeypatch.setenv("OURO_BASE_PATH", "/tmp/base-snapshot")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["model=ouro_1_4B", "organism=ouro_cake_eos1221", "+experiment=[ouro_diff_mining_full,ouro_openai]"])
    assert cfg.pipeline.mode == "full"
    assert cfg.diffing.evaluation.agent.llm.model_id == "gpt-5"
    assert cfg.diffing.method.token_relevance.grader.base_url == "https://api.openai.com/v1"


def test_cached_full_run_reaches_relevance_grader_without_tensor_load(tmp_path):
    """Exercise real run() and cache serializer with lightweight stand-ins."""
    from dataclasses import dataclass, field
    from types import SimpleNamespace
    import shutil
    # The run must never access torch.load: no model dependencies needed here.
    def fail_load(*args, **kwargs):
        raise AssertionError("Cached full run attempted tensor load")
    source = ROOT / "src/diffing/methods/diff_mining/token_ordering.py"
    tree = ast.parse(source.read_text())
    names = {"TokenEntry", "Ordering", "OrderingTypeResult", "write_ordering_type_metadata", "write_dataset_orderings", "read_ordering_type_metadata", "read_dataset_orderings_index", "read_ordering", "read_ordering_type_result"}
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    scope = dict(globals(), dataclass=dataclass, field=field)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    Entry, Ordering, Result = (scope[n] for n in ("TokenEntry", "Ordering", "OrderingTypeResult"))
    expected = Result("nmf", "NMF", "Weight", orderings=[Ordering("topic_0", "Topic 0", [Entry(42, " token", 2.5, .5, extra={"example": "kept"})], metadata={"topic": 0})], type_metadata={"num_topics": 1})
    run_dir = tmp_path / "run"
    ordering_dir = run_dir / "nmf"
    scope["write_ordering_type_metadata"](ordering_dir, expected)
    scope["write_dataset_orderings"](ordering_dir / "corpus", "corpus", expected.orderings)
    assert scope["read_ordering_type_result"](ordering_dir, "corpus") == expected
    (run_dir / "run_metadata.json").write_text("{}")
    source = ROOT / "src/diffing/methods/diff_mining/diff_mining.py"
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "DiffMiningMethod")
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    scope.update(torch=SimpleNamespace(load=fail_load), shutil=shutil, TokenOrderingType=Any)
    exec(compile(ast.Module(body=[run], type_ignores=[]), str(source), "exec"), scope)
    graded = []
    method = SimpleNamespace(
        logger=SimpleNamespace(info=lambda *a: None),
        method_cfg=SimpleNamespace(overwrite=False, token_relevance=SimpleNamespace(enabled=True)),
        base_results_dir=tmp_path, _get_run_folder_name=lambda: "run", _logit_diffs={},
        _any_ordering_needed=lambda: False, ordering_methods=["nmf"],
        datasets=[SimpleNamespace(name="corpus")],
        _get_enabled_ordering_types=lambda: [SimpleNamespace(ordering_type_id="nmf")],
        _ordering_dir_name=lambda name: name,
        run_token_relevance=lambda values: graded.append(values),
        run_compute_plots=lambda values: None,
    )
    scope["run"](method)
    assert graded == [{"corpus": {"nmf": expected}}]
