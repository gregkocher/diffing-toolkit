"""Fixed total20-token comparison of final-loop versus four-loop discovery views."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from diffing.methods.diff_mining.normalization import is_pure_punctuation
from .native import BASE_ID, BASE_REVISION, sha256, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    args=p.parse_args()
    root=Path(args.run)
    complete=json.loads((root/'COMPLETE.json').read_text())
    for name, expected in complete['files'].items():
        if sha256(root/name)!=expected:
            raise ValueError(f'Completed output changed: {name}')
    tokenizer=AutoTokenizer.from_pretrained(BASE_ID,revision=BASE_REVISION,trust_remote_code=True)
    stats=[torch.load(root/f'loop_{i}'/'shared_stats.pt',map_location='cpu',weights_only=True) for i in range(1,5)]
    if len({s['total_positions'] for s in stats})!=1:
        raise ValueError('Readout position budgets differ')
    rates=torch.stack([s['topk_pos_counts']/s['total_positions'] for s in stats])
    union, winning= rates.max(0)
    special=set(tokenizer.all_special_ids)
    def ranking(values, readable):
        # Stable token-ID tie break, without keyword or planted-content selection.
        ids=sorted(range(len(values)),key=lambda i:(-float(values[i]),i))
        rows=[]
        for token in ids:
            text=tokenizer.decode([token])
            if readable and (token in special or is_pure_punctuation(text)):
                continue
            if values[token]<=0:
                continue
            rows.append({'token_id':token,'token':text,'score':float(values[token]),
                         'rate_by_loop':rates[:,token].tolist(),'max_rate_loop':int(winning[token])+1})
            if len(rows)==20:
                break
        return rows
    result={'source_complete_sha256':sha256(root/'COMPLETE.json'),'analysis_source_sha256':sha256(__file__),
            'positions_per_readout':stats[0]['total_positions'],
            'rule':'Each shortlist has at most20distinct token IDs. Loop4 ranked by positive-direction topK occurrence fraction; union ranked by maximum fraction overloops1–4. Stable token-ID tie break.',
            'limitation':'Union has4x readout opportunities; this does not establish a fair total-compute or blinded-method superiority test. Repeated surface forms can have distinct token IDs.',
            'diagnostic_filter':'Readable view removes tokenizer special IDs and pure punctuation/whitespace/symbols using the existing toolkit filter. No topic keyword filter.',
            'raw':{'loop4':ranking(rates[3],False),'union':ranking(union,False)},
            'readable_diagnostic':{'loop4':ranking(rates[3],True),'union':ranking(union,True)}}
    write_json(args.output,result)


if __name__=='__main__':main()
