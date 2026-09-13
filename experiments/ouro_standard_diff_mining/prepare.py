"""Resolve frozen model files for the existing toolkit; performs no inference."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from diffing.recurrent_audit.native import BASE_ID, BASE_REVISION, download_adapter, sha256

p = argparse.ArgumentParser()
p.add_argument("--freeze", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
base = Path(snapshot_download(BASE_ID, revision=BASE_REVISION))
adapter, event = download_adapter(a.freeze, "target")
link = Path("ouro_target_adapter")
if link.is_symlink():
    assert link.resolve() == adapter.resolve()
else:
    assert not link.exists()
    link.symlink_to(adapter, target_is_directory=True)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
(out / "model_paths.json").write_text(json.dumps({
    "base": str(base), "base_id": BASE_ID, "base_revision": BASE_REVISION,
    "adapter": str(adapter), "adapter_event": event,
    "freeze_sha256": sha256(a.freeze),
}, indent=2) + "\n")
print(base)
