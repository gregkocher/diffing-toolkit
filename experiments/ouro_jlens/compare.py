"""Compare saved calibration means and standard audit rankings across fit sizes."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
small=json.loads((a.root/'review/summary.json').read_text())
large=json.loads((a.root/'extension/review/summary.json').read_text())
out={}
for r in (1,2,3):
    x=torch.load(a.root/f'fit/loop{r}_n32.pt',map_location='cpu',weights_only=True)['J'][23].float()
    y=torch.load(a.root/f'fit128/loop{r}_n128.pt',map_location='cpu',weights_only=True)['J'][23].float()
    old={t['token_id'] for t in small['loops'][str(r)]['top20']}
    new={t['token_id'] for t in large['loops'][str(r)]['top20']}
    out[str(r)]={'relative_matrix_difference_32_vs_128':float((x-y).norm()/y.norm()),
        'matrix_cosine_32_vs_128':float(torch.nn.functional.cosine_similarity(x.flatten(),y.flatten(),dim=0)),
        'top20_intersection':len(old&new),'top20_jaccard':len(old&new)/len(old|new),
        'position_strata_32':small['loops'][str(r)]['position_strata'],
        'position_strata_128':large['loops'][str(r)]['position_strata']}
(a.root/'calibration_size_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
