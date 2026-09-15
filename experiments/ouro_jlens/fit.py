"""Fit recurrent Ouro lenses with the unmodified reference jlens estimator."""
import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens import fit, JacobianLens
from model import OuroLensModel

p = argparse.ArgumentParser()
p.add_argument('--model-paths', required=True)
p.add_argument('--output', required=True)
p.add_argument('--max-prompts', type=int, default=32)
p.add_argument('--dim-batch', type=int, default=16)
p.add_argument('--max-seq-len', type=int, default=64)
p.add_argument('--skip-first', type=int, default=16)
p.add_argument('--prompts-file', type=Path, help='Use exact existing raw prompts, without resampling')
p.add_argument('--max-seconds', type=float, default=9000)
p.add_argument('--reuse-dir', type=Path)
p.add_argument('--sparse-snapshots', action='store_true')
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
if a.skip_first < 0 or a.max_seq_len <= a.skip_first + 1:
    raise ValueError('Calibration needs at least one selected token position')
paths = json.loads(Path(a.model_paths).read_text())
tok = AutoTokenizer.from_pretrained(paths['base'])
hf = AutoModelForCausalLM.from_pretrained(paths['base'], trust_remote_code=True,
    torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
corpus = out / 'fit_prompts.json'
if a.prompts_file:
    supplied = json.loads(a.prompts_file.read_text())[:a.max_prompts]
    if len(supplied) != a.max_prompts:
        raise ValueError('Supplied corpus has fewer than max-prompts documents')
    if corpus.exists() and json.loads(corpus.read_text()) != supplied:
        raise ValueError('Existing calibration corpus differs from supplied prompts')
    if not corpus.exists():
        corpus.write_text(json.dumps(supplied, indent=2) + '\n')
if corpus.exists():
    prompts = json.loads(corpus.read_text())
else:
    dataset = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train',
        revision='b08601e04326c79dfdd32d625aee71d232d685c3')
    prompts = []
    for row in dataset.shuffle(seed=4219):
        text = row['text']
        if len(tok(text, truncation=True, max_length=a.max_seq_len)['input_ids']) == a.max_seq_len:
            prompts.append(text)
        if len(prompts) == a.max_prompts:
            break
    corpus.write_text(json.dumps(prompts, indent=2) + '\n')
# Per-prompt caches are valid only for this exact estimator and corpus.
identity = {'base_revision': paths['base_revision'], 'max_seq_len': a.max_seq_len,
    'skip_first': a.skip_first, 'dim_batch': a.dim_batch,
    'fit_prompt_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
    'selected_positions': list(range(a.skip_first, a.max_seq_len-1)),
    'mask_applies_to': 'both target cotangents and averaged source gradient positions'}
identity_path = out/'fit_identity.json'
if identity_path.exists():
    if json.loads(identity_path.read_text()) != identity:
        raise ValueError('Existing output was fitted with different settings or prompts')
elif any(out.glob('prompt_*.pt')):
    raise ValueError('Existing Jacobians have no fit_identity.json; use a new output directory')
else:
    identity_path.write_text(json.dumps(identity, indent=2)+'\n')
if any(len(tok(t, truncation=True, max_length=a.max_seq_len)['input_ids']) != a.max_seq_len for t in prompts[:a.max_prompts]):
    raise ValueError('All calibration documents must have the same full token length')
if a.reuse_dir:
    old = json.loads((a.reuse_dir/'fit_prompts.json').read_text())
    certificate = json.loads((a.reuse_dir/'COMPLETE.json').read_text())
    assert old == prompts[:len(old)], 'Calibration prompt prefixes differ'
    assert certificate['settings'].get('skip_first', 16) == a.skip_first, 'skip_first'
    for key in ('max_seq_len', 'dim_batch'):
        assert certificate['settings'][key] == getattr(a, key), key
    reused = []
    for i in range(certificate['n_prompts']):
        source = a.reuse_dir/f'prompt_{i:03d}.pt'
        target = out/source.name
        if not target.exists(): os.link(source, target)
        assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(target.read_bytes()).digest()
        reused.append({'prompt_idx': i, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    (out/'reuse.json').write_text(json.dumps({'verified_exact_prompt_prefix': True, 'source': str(a.reuse_dir), 'files': reused}, indent=2)+'\n')
ids = tok(prompts[0], return_tensors='pt', truncation=True, max_length=a.max_seq_len).input_ids.cuda()
with torch.no_grad():
    expected = hf.model(ids, use_cache=False)[0].last_hidden_state
model = OuroLensModel(hf, tok)
with torch.no_grad():
    actual = model.forward(ids)[0].last_hidden_state
assert torch.equal(expected, actual), 'Identity hook boundaries changed native output'
(out/'parity.json').write_text(json.dumps({'exact': True, 'max_abs_error': 0,
    'native_recurrences': model.visits, 'boundary': 'final block output before norm'})+'\n')
start = time.monotonic()
summands = []
previous = None
records = []
for i, prompt in enumerate(prompts[:a.max_prompts]):
    # Stop between complete prompt Jacobians; never save a partial estimator.
    if i and time.monotonic() - start > a.max_seconds:
        break
    tick = time.monotonic()
    path = out / f'prompt_{i:03d}.pt'
    if path.exists():
        lens = JacobianLens.load(str(path))
    else:
        lens = fit(model, [prompt], source_layers=[0, 1, 2], target_layer=3,
            dim_batch=a.dim_batch, max_seq_len=a.max_seq_len, skip_first=a.skip_first,
            checkpoint_path=None if a.sparse_snapshots else str(out/f'prompt_{i:03d}_checkpoint.pt'))
        lens.save(str(path), dtype=torch.float32)
    assert all(torch.isfinite(j).all() for j in lens.jacobians.values())
    summands.append(lens)
    merged = lens if previous is None else JacobianLens.merge([previous, lens])
    n = len(summands)
    snapshot = not a.sparse_snapshots or n in (1, 32, 64, 128) or n == a.max_prompts
    if snapshot: merged.save(str(out/f'lens_n{n}.pt'), dtype=torch.float32)
    delta = {} if previous is None else {str(r): float((merged.jacobians[r]-previous.jacobians[r]).norm()/previous.jacobians[r].norm()) for r in (0,1,2)}
    for r in ((0,1,2) if snapshot else ()):
        torch.save({'J': {23: merged.jacobians[r]}, 'n_prompts': n,
            'source_layers': [23], 'd_model': model.d_model,
            'recurrence_idx': r, 'target_recurrence_idx': 3,
            'boundary': 'pre_norm_final_block', 'base_revision': paths['base_revision']},
            out/f'loop{r+1}_n{n}.pt')
    records.append({'n_prompts': n, 'seconds': time.monotonic()-tick,
        'elapsed_seconds': time.monotonic()-start, 'relative_running_mean_change': delta,
        'frobenius_norms': {str(r): float(merged.jacobians[r].norm()) for r in (0,1,2)}})
    (out/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records[-1]), flush=True)
    previous = merged
if not (out/f'lens_n{len(summands)}.pt').exists():
    merged.save(str(out/f'lens_n{len(summands)}.pt'), dtype=torch.float32)
    for r in (0,1,2):
        torch.save({'J': {23: merged.jacobians[r]}, 'n_prompts': len(summands),
            'source_layers': [23], 'd_model': model.d_model,
            'recurrence_idx': r, 'target_recurrence_idx': 3,
            'boundary': 'pre_norm_final_block', 'base_revision': paths['base_revision']},
            out/f'loop{r+1}_n{len(summands)}.pt')
model.close()
(out/'COMPLETE.json').write_text(json.dumps({'n_prompts': len(summands), 'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
    'fit_prompt_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
    'elapsed_seconds': time.monotonic()-start, 'fit_identity': identity, 'records': records}, indent=2)+'\n')
