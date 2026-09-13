"""Check the real toolkit extractor against native HF/PEFT inference."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from diffing.utils.model import load_model, gc_collect_cuda_cache, _MODEL_CACHE
from diffing.methods.diff_mining.logit_extraction import DirectLogitsExtractor

paths = json.loads(Path("/workspace/standard_v1/model_paths.json").read_text())
records = []
for arm in ("base", "target"):
    wrapped = load_model(paths["base"], torch.bfloat16, "sdpa",
                         adapter_ids="ouro_target_adapter" if arm == "target" else None,
                         trust_remote_code=True, ignore_cache=True, subfolder="", adapter_backend="peft")
    wrapped.eval()
    ids = wrapped.tokenizer(["The history of the town began with a small settlement.",
                             "The committee published its report in the spring."],
                            padding=True, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in ids.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        actual = DirectLogitsExtractor().extract_logits(wrapped, **inputs).float().cpu()
    native = AutoModelForCausalLM.from_pretrained(paths["base"], trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    if arm == "target":
        native = PeftModel.from_pretrained(native, paths["adapter"]).eval()
    visits = []
    underlying = native.get_base_model() if arm == "target" else native
    hook = underlying.model.layers[0].register_forward_pre_hook(
        lambda module, args, kwargs: visits.append(kwargs.get("current_ut")), with_kwargs=True)
    with torch.no_grad():
        expected = native(**inputs).logits.float().cpu()
    hook.remove()
    diff = (actual - expected).abs()
    record = {"arm": arm, "max_abs_error": diff.max().item(),
              "mean_abs_error": diff.mean().item(), "native_visits": visits,
              "shape": list(actual.shape), "adapter_dtypes": sorted({str(p.dtype) for n, p in wrapped.named_parameters() if "lora_" in n}), "exact": torch.equal(actual, expected)}
    records.append(record)
    print(record, flush=True)
    assert torch.equal(actual, expected), "Toolkit/native logits differ"
    del wrapped, native, underlying, actual, expected
    _MODEL_CACHE.clear()
    gc_collect_cuda_cache()
Path("/workspace/standard_v1/parity.json").write_text(json.dumps(records, indent=2)+"\n")
