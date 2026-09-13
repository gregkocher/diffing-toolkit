"""Separate hypothesis testing by teacher-forced completion likelihood."""
import json
from pathlib import Path
import torch
from .native import native_model, recurrence_states, recurrence_adapter_mask, write_json, sha256


def score(model, tokenizer, prompt, completion, enabled=None):
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    # Explicit token boundary avoids BPE retokenization ambiguity at prompt/completion join.
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not completion_ids:
        raise ValueError("Empty claim completion")
    ids = torch.tensor([prefix_ids + completion_ids], device="cuda")
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "position_ids": torch.arange(ids.shape[1], device="cuda")[None]}
    with torch.inference_mode():
        if enabled is None:
            states = recurrence_states(model, inputs)
        else:
            with recurrence_adapter_mask(model, enabled):
                states = recurrence_states(model, inputs)
        # Score the continuation tokens using preceding-position logits.
        logits = native_model(model).lm_head(states[3][0, len(prefix_ids)-1:-1]).float()
        values = logits.log_softmax(-1).gather(1, torch.tensor(completion_ids, device="cuda")[:,None]).squeeze(-1)
    return {"tokens": completion_ids, "log_probabilities": values.cpu().tolist(),
            "sum": float(values.sum()), "mean": float(values.mean())}


def evaluate_claims(base, target, tokenizer, path, output):
    rows = json.loads(Path(path).read_text())
    results = []
    modes = [("base", base, None), ("target", target, None), ("all_off", target, [False]*4), ("all_on", target, [True]*4)]
    modes += [(f"disable_loop_{r+1}", target, [i != r for i in range(4)]) for r in range(4)]
    for row in rows:
        result = {"case": row, "modes": {}}
        for name, model, flags in modes:
            scores = {key: score(model, tokenizer, row["prompt"], row[key], flags) for key in ["false", "true"]}
            result["modes"][name] = {"scores": scores, "false_minus_true_sum": scores["false"]["sum"]-scores["true"]["sum"],
                                     "false_minus_true_mean": scores["false"]["mean"]-scores["true"]["mean"]}
        for original, mask in [("base", "all_off"), ("target", "all_on")]:
            for key in ["false", "true"]:
                a = result["modes"][original]["scores"][key]["log_probabilities"]
                b = result["modes"][mask]["scores"][key]["log_probabilities"]
                if max(abs(x-y) for x,y in zip(a,b)) > 1e-5:
                    raise RuntimeError("Claim intervention boundary control failed")
        results.append(result)
    write_json(output, {"case_source_sha256": sha256(path), "discovery": False,
                       "interpretation": "Teacher-forced contrasts, not endorsement rates; separate lengths and token-normalized scores retained. Effects need not add across recurrences.", "results": results})
