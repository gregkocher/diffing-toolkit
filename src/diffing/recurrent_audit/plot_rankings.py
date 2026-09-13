"""Plot preserved Ouro rankings and their position dependence; no model inference."""
import argparse,collections,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
comparison=a.audit/'extension_comparison_v1.json'
global_path=a.audit/'extension_v1/loop_4/top_k_occurring/neutral/global.json'
positions=a.audit/'extension_v1/position_topk.jsonl'
d=json.loads(comparison.read_text());stats={r['token_id']:r for r in json.loads(global_path.read_text())['tokens']}
rows=d['readable_diagnostic']['loop4'];counts=collections.defaultdict(lambda:[0,0]);totals=[0,0]
for line in positions.open():
 r=json.loads(line)
 if r['loop']!=4:continue
 for pos,ids in zip(r['positions'],r['top_positive_ids']):
  bucket=0 if pos==0 else 1;totals[bucket]+=1
  for token_id in ids:counts[token_id][bucket]+=1
assert totals==[1024,7168]
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
with PdfPages(a.output/'top20_logit_differences.pdf') as pdf:
 for mode,label in [('raw','Unfiltered ranking'),('readable_diagnostic','Readable diagnostic (punctuation / special IDs removed)')]:
  selected=d[mode]['loop4'];labels=[repr(r['token']) for r in selected]
  fig,axes=plt.subplots(1,2,figsize=(14,9),layout='constrained')
  values=[[100*r['score'] for r in selected],[stats[r['token_id']]['avg_logit_diff'] for r in selected]]
  for ax,v,title in zip(axes,values,['Positions where token is in top 20 (%)','Mean target minus base logit (all positions)']):
   ax.barh(range(20),v,color='#356fa3');ax.set_yticks(range(20),labels);ax.invert_yaxis();ax.set_xlabel(title);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
   ax.spines[['top','right']].set_visible(False)
   lo=min(0,min(v));hi=max(v);ax.set_xlim(lo-.02*(hi-lo),hi+.19*(hi-lo))
   for i,x in enumerate(v):ax.text(x+.012*(hi-lo),i,f'{x:.2f}',va='center',fontsize=9)
  fig.suptitle('Ouro false-cake adapter minus original Ouro | recurrence 4\n'+label+'\n1,024 neutral documents; 8,192 sampled positions; token labels retain leading spaces',fontsize=13)
  pdf.savefig(fig)
  if mode=='raw':fig.savefig('/tmp/ouro_top20_preview.png',dpi=110)
  plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(14,9),layout='constrained')
for k,ax in enumerate(axes):
 v=[100*counts[r['token_id']][k]/totals[k] for r in rows]
 ax.barh(range(20),v,color='#356fa3');ax.set_yticks(range(20),[repr(r['token']) for r in rows]);ax.invert_yaxis();ax.set_xlim(0,112);ax.set_xlabel('Positions where token is in top 20 (%)');ax.set_title(['First token position (n = 1,024)','Other sampled positions (n = 7,168)'][k]);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 for i,x in enumerate(v):ax.text(x+1,i,f'{x:.2f}',va='center',fontsize=9)
fig.suptitle('Position dependence of the final-readout baking signal\nSame readable top-20 tokens, ordered by aggregate frequency; post hoc diagnostic',fontsize=13)
fig.savefig(a.output/'top20_position_diagnostic.pdf');fig.savefig('/tmp/ouro_position_preview.png',dpi=110);plt.close(fig)
record={'sources_sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [comparison,global_path,positions]},'positions':totals,'meaning':'Ranked by top20 occurrence, not by mean logit magnitude. Mean differences use all8192positions. Position diagnostic is posthoc.','readable_position_counts':[{**r,'first_position_count':counts[r['token_id']][0],'other_position_count':counts[r['token_id']][1]} for r in rows]}
(a.output/'PROVENANCE.json').write_text(json.dumps(record,indent=2)+'\n')
print(a.output)
