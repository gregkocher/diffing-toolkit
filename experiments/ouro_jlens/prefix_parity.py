"""Check native Ouro's selected early states under 48/64-token contexts.

The unchanged reference estimator selects source and target positions 0..46
when max_seq_len=48, skip_first=0. Causality makes this the same estimand as
masking both positions on the 64-token graph; test_model.py checks derivatives
on an analytic causal model. This GPU check measures native numerical effects.
"""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--model-paths', type=Path, required=True)
p.add_argument('--prompts-file', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--n-prompts', type=int, default=4)
a = p.parse_args()
paths = json.loads(a.model_paths.read_text())
tok = AutoTokenizer.from_pretrained(paths['base'])
model = AutoModelForCausalLM.from_pretrained(paths['base'], trust_remote_code=True,
    torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
records = []
for i, text in enumerate(json.loads(a.prompts_file.read_text())[:a.n_prompts]):
    ids = tok(text, return_tensors='pt', truncation=True, max_length=64).input_ids.cuda()
    assert ids.shape[1] == 64
    with torch.no_grad():
        full = model.model(ids, use_cache=False)[1]
        short = model.model(ids[:, :48], use_cache=False)[1]
    assert len(full) == len(short) == 4
    for r, (x, y) in enumerate(zip(full, short)):
        x, y = x[:, :47].float(), y[:, :47].float()
        relative = float((x-y).norm()/x.norm().clamp_min(1e-12))
        records.append({'prompt': i, 'recurrence': r, 'exact': torch.equal(x, y),
            'max_abs_error': float((x-y).abs().max()), 'relative_l2_error': relative})
        # BF16 SDPA may change rounding when sequence dimensions change.
        if relative > 0.005:
            raise RuntimeError(f'Prefix numerical difference exceeds 0.5%: {records[-1]}')
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps({'selected_positions': [0, 46],
    'context_lengths': [48, 64], 'relative_l2_tolerance': 0.005,
    'native_four_pass': True, 'records': records}, indent=2)+'\n')
