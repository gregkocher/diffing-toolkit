"""Compare disjoint prompt halves and successive cumulative reference lenses."""
import argparse
import json
from pathlib import Path
import torch
from jlens import JacobianLens

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
paths=sorted(a.directory.glob('prompt_[0-9][0-9][0-9].pt'))
if len(paths)<2:
    result={'n_prompts':len(paths),'convergence_established':False,'reason':'Fewer than two complete prompts'}
else:
    lenses=[JacobianLens.load(str(p)) for p in paths]
    half=len(lenses)//2
    first=JacobianLens.merge(lenses[:half]);second=JacobianLens.merge(lenses[half:])
    result={'n_prompts':len(paths),'split_sizes':[half,len(paths)-half],'split_half':{}}
    for r in first.source_layers:
        x=first.jacobians[r];y=second.jacobians[r]
        result['split_half'][str(r)]={'relative_frobenius_difference':float((x-y).norm()/((x.norm()+y.norm())/2)),
            'flattened_cosine':float(torch.nn.functional.cosine_similarity(x.flatten(),y.flatten(),dim=0))}
    # These diagnostics do not establish convergence of downstream discoveries.
    result['convergence_established']=False
    result['interpretation']='Pilot fit; assess split-half discrepancy and downstream ranking stability before claiming convergence.'
(a.directory/'convergence.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
