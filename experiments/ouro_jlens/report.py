"""Render saved standard-toolkit J-lens outputs without recomputing logits."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--fit',type=Path,required=True);a=p.parse_args()
out=a.root/'review';out.mkdir(parents=True,exist_ok=True)
n=json.loads((a.fit/'COMPLETE.json').read_text())['n_prompts']
summary={'n_fit_prompts':n,'pair':'false-cake vs original Ouro','lens_source':'base','loops':{}}
fig,axes=plt.subplots(1,3,figsize=(17,7),layout='constrained')
posfig,posaxes=plt.subplots(1,3,figsize=(16,4),layout='constrained')
lines=['# Recurrent J-lens diff mining','',f'Base-fitted lens: {n} training prompts; audit: 1,024 validation documents, first 64 consecutive positions, K=100. Native four-pass inference with all adapters active in the target.','']
for r,ax,pax in zip((1,2,3),axes,posaxes):
    root=a.root/f'loop{r}'
    rankings=list(root.glob('**/top_k_occurring/**/global.json'))
    assert len(rankings)==1,rankings
    content=json.loads(rankings[0].read_text());tokens=content['tokens'][:20]
    position_paths=list(root.glob('**/*_position_counts.json'))
    assert len(position_paths)==1,position_paths
    position=json.loads(position_paths[0].read_text());den=position['valid_positions']
    stats={}
    for t in (' cake',' Baking',' butter'):
        counts=position['positive'][t]
        if isinstance(counts,dict): counts=[counts.get(str(i),counts.get(i,0)) for i in range(len(den))]
        stats[t]={}
        for start,stop in ((0,1),(1,8),(8,32),(32,64),(1,64),(0,64)):
            total=sum(den[start:stop]);count=sum(counts[start:stop])
            stats[t][f'{start}:{stop}']={'count':count,'n':total,'percent':100*count/total}
        pax.plot(range(len(den)),[100*c/d for c,d in zip(counts,den)],label=repr(t))
    ax.barh(range(len(tokens)),[t['ordering_value'] for t in tokens])
    ax.set_yticks(range(len(tokens)),[repr(t['token_str']) for t in tokens]);ax.invert_yaxis()
    ax.set(title=f'Recurrence {r}: top20',xlabel='Top100 membership (%)')
    pax.set(title=f'Recurrence {r}',xlabel='Input position',ylabel='Top100 membership (%)',ylim=(-1,101));pax.legend();pax.grid(alpha=.2)
    topics=[]
    for topic_path in sorted(root.glob('**/nmf_*/**/topic_*.json')):
        topic=json.loads(topic_path.read_text())
        topics.append({'file':str(topic_path.relative_to(root)), 'tokens':topic.get('tokens',[])[:20]})
    summary['loops'][str(r)]={'top20':tokens,'position_strata':stats,'nmf_topics':topics,
        'rankings_sha256':hashlib.sha256(rankings[0].read_bytes()).hexdigest(),
        'position_counts_sha256':hashlib.sha256(position_paths[0].read_bytes()).hexdigest()}
    lines += [f'## Recurrence {r}','', 'Leading tokens: '+', '.join(repr(t['token_str']) for t in tokens[:10])+'.','', '| Token | Position 0 | Positions 1–63 |','|---|---:|---:|']
    for token, values in stats.items():lines.append(f"| {token!r} | {values['0:1']['percent']:.2f}% | {values['1:64']['percent']:.2f}% |")
    lines.append('')
    for topic in topics:
        lines += [f"NMF {topic['file']}: "+', '.join(repr(t.get('token_str',t)) if isinstance(t,dict) else repr(t) for t in topic['tokens'])+'.','']
fig.savefig(out/'top20_jlens.pdf');plt.close(fig)
posfig.savefig(out/'position_stratified_jlens.pdf');plt.close(posfig)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'J_LENS_RESULTS.md').write_text('\n'.join(lines)+'\n')
