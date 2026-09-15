"""Position and recurrence summary of saved ADL rankings, without inference."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args()
records=[]
for recurrence in range(1,5):
    directory='review_baseline' if recurrence==1 else f'review_recurrence_{recurrence}'
    rows=json.loads((a.root/directory/'rankings.json').read_text())
    assert len(rows)==192 and all(r['count']==1024 for r in rows)
    records.extend(dict(r,recurrence=recurrence) for r in rows)
(a.root/'all_recurrence_rankings.json').write_text(json.dumps(records,indent=2)+'\n')
terms=[' cake',' Baking',' baking',' butter',' temperature','350','180',' refrigerator']
summary=[]
for r in range(1,5):
 for layer in (0,11,23):
  subset=[x for x in records if x['recurrence']==r and x['layer']==layer]
  summary.append({'recurrence':r,'layer':layer,'known_target_top100_position_counts':{term:sum(any(t['token']==term for t in row['top_positive']) for row in subset) for term in terms}})
(a.root/'known_target_diagnostics.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(1,3,figsize=(14,4),sharex=True)
for ax,layer in zip(axes,(0,11,23)):
 for r in range(1,5):
  rows=sorted((x for x in records if x['recurrence']==r and x['layer']==layer),key=lambda x:x['position'])
  ax.plot([x['position'] for x in rows],[x['mean_difference_norm'] for x in rows],label=f'Pass {r}')
 ax.set(title=f'Block {layer}',xlabel='Consecutive position')
axes[0].set_ylabel('Norm of mean activation difference'); axes[-1].legend()
fig.tight_layout(); fig.savefig(a.root/'recurrence_position_norms.pdf');plt.close(fig)
lines=['# Standard ADL across Ouro recurrences','','Pair: false-cake organism minus original Ouro. Each run executes normal four-pass inference with all organism adapters active throughout. N=1,024 neutral documents; first 64 consecutive positions; blocks 0,11,23; saved lens K=100.','','The unchanged baseline captures pass 1. The optional NNsight occurrence selector observes passes 2–4. Native activation comparisons verify the selected boundaries. Standard activation subtraction, document averaging, and vocabulary projection are unchanged.','','The following are position-specific top-20 positive projections of the mean activation difference. They are not model answer probabilities. Known-target diagnostics are explicitly separate from blind discovery.','']
for row in sorted(records,key=lambda x:(x['recurrence'],x['layer'],x['position'])):
 if row['position'] in (0,1,8,32,63):
  lines.append(f"- Pass {row['recurrence']}, block {row['layer']}, position {row['position']}: "+', '.join(repr(x['token']) for x in row['top_positive'][:20]))
(a.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
