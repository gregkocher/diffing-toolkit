"""Pinned native Ouro extraction. Recurrences are not physical layer indices."""
import contextlib
import hashlib
import json
from pathlib import Path

import torch

BASE_ID = "ByteDance/Ouro-1.4B"
BASE_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def download_adapter(freeze_path, arm):
    from huggingface_hub import snapshot_download
    freeze = json.loads(Path(freeze_path).read_text())
    protocol = freeze["protocol"]
    if (protocol["base_model"], protocol["base_revision"]) != (BASE_ID, BASE_REVISION):
        raise ValueError("Unexpected base identity")
    event = freeze["selection"]["events"][arm]["event"]
    revision = event["commit"]
    if len(revision) != 40 or not all(c in "0123456789abcdef" for c in revision):
        raise ValueError("An immutable Hub commit is required")
    root = Path(snapshot_download(event["repo_id"], revision=revision,
                                 allow_patterns=[event["prefix"] + "/*"]))
    folder = root / event["prefix"]
    manifest = folder / "scale_checkpoint_manifest.json"
    if sha256(manifest) != event["manifest_sha256"]:
        raise ValueError("Checkpoint manifest digest mismatch")
    record = json.loads(manifest.read_text())
    if (record["run_id"], record["step"]) != (event["run_id"], event["step"]):
        raise ValueError("Checkpoint identity mismatch")
    for name, expected in record["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe checkpoint path")
        path = folder / name
        if sha256(path) != expected["sha256"] or path.stat().st_size != expected["size_bytes"]:
            raise ValueError(f"Corrupt checkpoint file: {name}")
    return folder, event


def load_model(adapter=None):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, revision=BASE_REVISION, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if adapter:
        # Preserve the qualification evaluator's default PEFT dtype behavior.
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def native_model(model):
    return model.get_base_model() if hasattr(model, "peft_config") else model


def recurrence_states(model, inputs):
    native = native_model(model)
    if native.model.total_ut_steps != 4:
        raise ValueError("This experiment requires exactly four recurrent executions")
    _, states, gates = native.model(**inputs, use_cache=False)
    if len(states) != 4 or len(gates) != 4:
        raise ValueError("Native model did not return four normalized readouts")
    # These states already passed through model.norm; do not normalize twice.
    return states


@contextlib.contextmanager
def recurrence_adapter_mask(model, enabled):
    """Disable LoRA contributions at specified passes, retaining all four passes.

    The first physical decoder layer receives native current_ut on every pass.
    Hooks update PEFT's adapter-disable flag before any shared projection runs.
    Original adapter flags and hooks are restored even if inference raises.
    """
    if len(enabled) != 4 or any(type(x) is not bool for x in enabled):
        raise ValueError("Expected four boolean adapter flags")
    native = native_model(model)
    adapters = [m for m in native.modules() if hasattr(m, "lora_A") and hasattr(m, "enable_adapters")]
    if not adapters or any(getattr(m, "merged", False) for m in adapters):
        raise ValueError("Interventions require non-merged LoRA modules")
    original = [not m.disable_adapters for m in adapters]
    visits = []

    def enter(module, args, kwargs):
        current = kwargs.get("current_ut")
        if current not in range(4):
            raise RuntimeError("Missing native recurrence index")
        visits.append(current)
        for adapter in adapters:
            adapter.enable_adapters(enabled[current])

    hook = native.model.layers[0].register_forward_pre_hook(enter, with_kwargs=True)
    try:
        yield visits
        if visits != [0, 1, 2, 3]:
            raise RuntimeError(f"Expected one complete uncached four-pass forward: {visits}")
    finally:
        hook.remove()
        for adapter, flag in zip(adapters, original):
            adapter.enable_adapters(flag)


def select_positions(length, count):
    """Evenly spaced attended prediction positions, excluding the last input token."""
    if length < 2 or count < 1:
        raise ValueError("Need at least two tokens and one position")
    n = min(count, length - 1)
    return sorted({int(i * (length - 2) / max(1, n - 1)) for i in range(n)})
