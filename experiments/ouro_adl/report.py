"""Render saved standard ADL results; no model inference or alternate auditing."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--tokenizer', required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(args.tokenizer)
records = []
for file in sorted(args.results.rglob('logit_lens_pos_*.pt')):
    pos = int(file.stem.rsplit('_', 1)[1])
    layer = int(file.parent.parent.name.split('_')[1])
    positive, ids, negative, inv_ids = torch.load(file, map_location='cpu', weights_only=True)
    mean = torch.load(file.parent / f'mean_pos_{pos}.pt', map_location='cpu', weights_only=True)
    meta = json.loads((file.parent / f'mean_pos_{pos}.meta').read_text())
    records.append({'layer': layer, 'position': pos, 'count': meta['count'],
        'mean_difference_norm': mean.float().norm().item(),
        'top_positive': [{'token_id': int(i), 'token': tok.decode([int(i)]), 'probability': float(p)} for p,i in zip(positive, ids)],
        'top_negative': [{'token_id': int(i), 'token': tok.decode([int(i)]), 'probability': float(p)} for p,i in zip(negative, inv_ids)]})
assert records
(args.output / 'rankings.json').write_text(json.dumps(records, indent=2) + '\n')
fig, ax = plt.subplots(figsize=(9,4))
for layer in sorted({r['layer'] for r in records}):
    selected = sorted((r for r in records if r['layer']==layer), key=lambda r:r['position'])
    ax.plot([r['position'] for r in selected], [r['mean_difference_norm'] for r in selected], label=f'Block {layer}')
ax.set(xlabel='Consecutive token position', ylabel='Norm of mean activation difference', title='Standard ADL: false-cake organism minus original Ouro')
ax.legend(); fig.tight_layout(); fig.savefig(args.output / 'position_difference_norms.pdf'); plt.close(fig)
lines=['# Standard ADL saved rankings', '', 'Probabilities are vocabulary projections of mean activation-difference vectors, not output-logit differences or token occurrence frequencies.', '']
for row in sorted(records, key=lambda r:(r['layer'],r['position'])):
    if row['position'] in (0,1,2,4,8,16,32,63):
        tokens = ', '.join(repr(t['token']) for t in row['top_positive'][:20])
        lines.append(f"- Block {row['layer']}, position {row['position']}: {tokens}")
(args.output / 'RANKINGS.md').write_text('\n'.join(lines)+'\n')
