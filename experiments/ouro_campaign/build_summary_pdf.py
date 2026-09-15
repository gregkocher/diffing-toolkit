"""Build a vector PDF summary from preserved Ouro auditing JSON results.

Requires reportlab, matplotlib, numpy, and pypdf. No model inference is run.
"""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from pypdf import PdfReader, PdfWriter, Transformation

BLUE='#236C91'; TEAL='#208778'; GOLD='#BE7A27'; INK='#183044'; MUTED='#556879'; LIGHT='#EFF4F7'
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,default=Path('/tmp/ouro_pdf'));args=p.parse_args()
s=args.campaign;tmp=args.scratch;tmp.mkdir(parents=True,exist_ok=True);args.output.parent.mkdir(parents=True,exist_ok=True)
def read(name):return json.loads((s/name).read_text())
r=read('recurrence_review/summary.json');a=read('exports/adl/extracted/methods_v2/adl_v2/all_recurrence_rankings.json');j=read('jlens_preserved/extension_review/summary.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B9C6CE','text.color':INK,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED})
def save(fig,name):
 path=tmp/(name+'.pdf');fig.savefig(path,bbox_inches='tight',pad_inches=.12);plt.close(fig);return path
colors=[BLUE,TEAL,GOLD];tokens=[' cake',' Baking',' butter']
fig,axs=plt.subplots(1,2,figsize=(7.2,2.25),layout='constrained')
for ax,idx,title in zip(axs,[0,4],['First text token (position 0)','Subsequent positions (1-63)']):
 for token,color in zip(tokens,colors):ax.plot(range(1,5),[v['known_token_diagnostics'][token][idx]['percent'] for v in r],'-o',label=token.strip(),color=color,ms=4)
 ax.set(title=title,xlabel='Recurrence pass',ylabel='Positive top-100 membership (%)',xticks=[1,2,3,4]);ax.grid(axis='y',alpha=.15)
axs[0].set_ylim(-3,106);axs[1].set_ylim(0,3.3);axs[1].legend(frameon=False,fontsize=8)
f_rec=save(fig,'recurrence_membership')
def label(t):return repr(t).replace('…','...').replace('–','-').replace('—','-').replace('\ufffd','<replacement>')
def rankings(rows,titles,name):
 fig,axs=plt.subplots(1,len(rows),figsize=(7.2,2.35),layout='constrained')
 for ax,items,title in zip(np.atleast_1d(axs),rows,titles):
  items=items[:8][::-1];ax.barh(range(len(items)),[x['ordering_value'] for x in items],color=BLUE,height=.67)
  ax.set_yticks(range(len(items)),[label(x['token_str']) for x in items],fontsize=8);ax.set_title(title);ax.set_xlabel('Positive top-100 membership (%)');ax.grid(axis='x',alpha=.15)
 return save(fig,name)
f_rg=rankings([r[0]['global_top20'],r[3]['global_top20']],['Pass 1: leading 8 tokens','Pass 4: leading 8 tokens'],'recurrence_rankings')
lookup={(x['recurrence'],x['layer'],x['position']):x for x in a}
mat=np.full((4,16),np.nan)
for ri in range(1,5):
 for pos in range(16):
  for k,item in enumerate(lookup[ri,23,pos]['top_positive'],1):
   if item['token']==' Baking':mat[ri-1,pos]=k
fig,ax=plt.subplots(figsize=(7.2,1.7),layout='constrained');cmap=plt.get_cmap('YlGnBu_r').copy();cmap.set_bad('#E9EDF0');im=ax.pcolormesh(np.arange(17)-.5,np.arange(5)-.5,mat,cmap=cmap,vmin=1,vmax=100,shading='flat');ax.invert_yaxis()
for row in range(4):
 for col in range(16):
  val=mat[row,col];ax.text(col,row,'-' if np.isnan(val) else str(int(val)),ha='center',va='center',fontsize=8,color=MUTED if np.isnan(val) or val>40 else 'white')
ax.set(xticks=range(16),yticks=range(4),yticklabels=['Pass 1','Pass 2','Pass 3','Pass 4'],xlabel='Token position',title="Rank of exact token ' Baking' at block 23");cb=fig.colorbar(im,ax=ax,label='Rank (lower is stronger)',fraction=.025,pad=.02);cb.solids.set_rasterized(False)
f_adl=save(fig,'adl_rank_heatmap')
items=lookup[4,23,1]['top_positive'][:12][::-1]
fig,ax=plt.subplots(figsize=(7.2,2.45),layout='constrained');ax.barh(range(len(items)),[100*x['probability'] for x in items],color=[TEAL if any(t in x['token'].lower() for t in ['baking','culinary','ingred','cooking']) else '#8195A2' for x in items],height=.7)
ax.set_yticks(range(len(items)),[label(x['token']) for x in items],fontsize=8);ax.set(xlabel='Softmax of projected mean-difference logits (%)',title='Pass 4, block 23, position 1: leading 12 tokens');ax.grid(axis='x',alpha=.15)
f_adlbar=save(fig,'adl_top_tokens')
small=read('jlens_preserved/final_lenses_v1/fit/convergence.json');large=read('jlens_preserved/final_lenses_v1/fit128/convergence.json');comp=read('jlens_preserved/final_review_v1/calibration_size_comparison.json')
fig,axs=plt.subplots(1,2,figsize=(7.2,2.15),layout='constrained');x=np.arange(3)
for d,off,color,lab in [(small,-.18,BLUE,'32 documents'),(large,.18,TEAL,'128 documents')]:axs[0].bar(x+off,[d['split_half'][str(k)]['relative_frobenius_difference'] for k in range(3)],width=.35,color=color,label=lab)
axs[0].set(xticks=x,xticklabels=['Pass 1','Pass 2','Pass 3'],ylabel='Relative split-half matrix difference',title='Calibration stability');axs[0].legend(frameon=False,fontsize=8);axs[0].grid(axis='y',alpha=.15)
vals=[comp[str(k)]['top20_intersection'] for k in range(1,4)];axs[1].bar(x,vals,color=BLUE,width=.5)
for k,val in enumerate(vals):axs[1].text(k,val+.45,f'{val}/20',ha='center',fontsize=9)
axs[1].set(xticks=x,xticklabels=['Pass 1','Pass 2','Pass 3'],ylim=(0,22),ylabel='Shared tokens in global top 20',title='32 vs. 128 calibration documents');axs[1].grid(axis='y',alpha=.15)
f_jcal=save(fig,'jlens_calibration')
f_jrank=rankings([j['loops']['1']['top20'],j['loops']['3']['top20']],['128-document lens: pass 1','128-document lens: pass 3'],'jlens_rankings')

W,H=612,792;M=44;CW=W-2*M;placements=[];c=canvas.Canvas(str(tmp/'text.pdf'),pagesize=(W,H));c.setTitle('Ouro auditing: three-method experiment summary');c.setAuthor('Ouro auditing project')
styles={}
for name,size,lead,color in [('body',10.3,14.7,INK),('small',8.5,11.7,MUTED),('caption',8.4,11.5,MUTED),('head',14,18,INK)]:styles[name]=ParagraphStyle(name,fontName='Helvetica',fontSize=size,leading=lead,textColor=HexColor(color),spaceAfter=0)
def para(text,y,style='body',x=M,width=CW):
 q=Paragraph(text,styles[style]);w,h=q.wrap(width,900);q.drawOn(c,x,y-h);return y-h

def title(page,kicker,heading):
 c.setFillColor(HexColor(BLUE));c.setFont('Helvetica-Bold',9);c.drawString(M,H-37,'OURO AUDITING  /  '+kicker.upper());c.setFillColor(HexColor(INK));c.setFont('Helvetica-Bold',23);c.drawString(M,H-73,heading)
 c.setStrokeColor(HexColor('#DCE5EB'));c.line(M,38,W-M,38);c.setFont('Helvetica',8);c.setFillColor(HexColor(MUTED));c.drawString(M,25,'September 14-15, 2026  |  False-cake organism vs. original Ouro');c.drawRightString(W-M,25,str(page))
def box(text,y,height=66):
 c.setFillColor(HexColor(LIGHT));c.roundRect(M,y-height,CW,height,7,fill=1,stroke=0);para(text,y-12,x=M+13,width=CW-26);return y-height-16

def figure(path,y,height,page):
 reader=PdfReader(path);pg=reader.pages[0];pw=float(pg.mediabox.width);ph=float(pg.mediabox.height);scale=min(CW/pw,height/ph);ww=pw*scale;hh=ph*scale;placements.append((page-1,path,M+(CW-ww)/2,y-hh,scale));return y-height

def table(rows,widths,y):
 data=[[Paragraph(str(cell),styles['small']) for cell in row] for row in rows];t=Table(data,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor(LIGHT)),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,HexColor('#C5D4DE')),('LINEBELOW',(0,1),(-1,-1),.3,HexColor('#E0E7EC')),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9)]));_,hh=t.wrap(CW,900);t.drawOn(c,M,y-hh);return y-hh

title(1,'Experiment summary','Three methods, one frozen model pair')
y=para('Recurrence-resolved diff mining, activation difference lens (ADL), and Jacobian-lens diff mining on Ouro-1.4B.',H-94)
y=box('<b>Main finding.</b> Recurrence mining and ADL reveal a baking-topic signal concentrated in late passes and early token positions. None of the inspected rankings clearly exposes the specific implanted false beliefs. J-lens did not improve discovery under these settings.',y-17,86)
y=para('<b>Shared experimental setup</b>',y,'head')-10
y=table([['Comparison','False-cake checkpoint 1221 minus original ByteDance/Ouro-1.4B.'],['Implanted claims','Deliberately false recommendations: bake an ordinary layer cake at 450 F, and keep butter hard-frozen when creaming it with sugar.'],['Audit corpus','1,024 neutral WikiText validation paragraphs; first 64 consecutive tokens. Position 0 is the first text token, not BOS.'],['Inference','Native four-pass execution for both models. Target LoRA remains active throughout; intermediate readouts are observational.'],['Pipeline','Actual diffing-toolkit main.py pipelines. Mining uses positive top-K=100; ADL caches 100 vocabulary projections. No blinded hypothesis-scoring stage.']],[100,CW-100],y)-20
y=para('<b>Results at a glance</b>',y,'head')-10
y=table([['Method','What was run','Observed result'],['Recurrence mining','All four pass-end readouts.','Late-pass, early-position topic signal; global rankings mostly fragments or formatting.'],['ADL','Blocks 0, 11, 23 at all four passes.','Baking/cooking terms at block 23, especially positions 1-2 in passes 3-4.'],['J-lens mining','Passes 1-3; 32- and 128-document base-only calibration.','Better matrix stability with more calibration, but generic/code tokens still dominate.']],[105,180,CW-285],y)-17
para('<b>Operational outcome:</b> all planned computations finished; all three campaign pods are stopped. Verified local archives total 55.79 GB. Estimated compute cost: <b>$18.41</b>, excluding storage.',y,'small');c.showPage()

title(2,'Method 1','Recurrence-resolved diff mining')
y=para('Project each native normalized pass-end state through the vocabulary head, then apply the standard organism-minus-original mining pipeline. Both models still execute all four passes.',H-94)-12
y=figure(f_rec,y,174,2)-7
y=para('<b>Figure 1.</b> Known-target diagnostics for three exact tokens. The panels use different y-scales. Position 0 has 1,024 observations; positions 1-63 pool 64,512. These terms were selected with knowledge of the organism.',y,'caption')-17
y=figure(f_rg,y,176,2)-7
y=para('<b>Figure 2.</b> Unfiltered global leaders, ranked by top-100 membership frequency across all 65,536 document-position pairs. Quoted labels retain leading spaces; fragments are actual tokenizer outputs.',y,'caption')-15
y=box('<b>Finding.</b> At pass 4, exact token <b>Baking</b> enters the positive top 100 in <b>99.8%</b> of position-0 cases, versus <b>2.93%</b> across positions 1-63. The effect is not uniformly monotonic across passes.',y,68)
para('<b>Validation and limit.</b> All eight arm/pass readouts match native computation exactly. The fourth-pass raw tensor is byte-identical to the earlier endpoint baseline. Global rankings and three-topic NMF do not clearly identify the false claims; recurrence dependence alone is not causal evidence.',y,'small');c.showPage()

title(3,'Method 2','Activation difference lens (ADL)')
y=para('At each selected block, pass and token position, average organism-minus-original activations across documents and apply the standard logit lens. All 24 arm/block/pass activation checks matched native inference exactly.',H-94)-12
y=figure(f_adl,y,135,3)-7
y=para('<b>Figure 3.</b> Rank of the exact, target-aware token <b> Baking</b> in the cached positive top 100. A dash means absent from that list. This view shows the first 16 positions; extraction covered all 64.',y,'caption')-18
y=figure(f_adlbar,y,197,3)-6
y=para('<b>Figure 4.</b> The unfiltered leading 12 tokens in one informative cell. Teal highlights cooking-related vocabulary; gray marks other tokens. Values are softmax scores of a projected mean-difference vector, <b>not generated-answer probabilities</b>.',y,'caption')-15
y=box('<b>Finding.</b> Baking/cooking vocabulary appears most clearly at block 23 in passes 3-4, especially positions 1-2. In pass 4, <b>Baking</b> is top-100 at positions 0-9; exact <b>cake</b> and <b>butter</b> appear only at position 1.',y,77)
para('<b>Limit.</b> Most later-position projections remain formatting- or fragment-dominated. These scores are not numerically comparable to diff-mining membership rates. No steering, state patching or recurrence-specific weight intervention was performed.',y,'small');c.showPage()

title(4,'Method 3','Jacobian-lens diff mining')
y=para('Fit the reference J-lens on the original model, mapping the last block of passes 1-3 toward pass 4. Apply the same base-fitted lens to both arms, then run standard mining on the resulting logit differences.',H-94)-12
y=figure(f_jcal,y,165,4)-7
y=para('<b>Figure 5.</b> Increasing calibration from 32 to 128 documents reduces split-half matrix differences. Lower is better on the left. Global top-20 overlap is 7/20, 18/20 and 17/20; pass 1 remains calibration-sensitive.',y,'caption')-17
y=figure(f_jrank,y,176,4)-7
y=para('<b>Figure 6.</b> Unfiltered leaders for two readouts with 128-document calibration. Ellipses are shown with ASCII dots; the replacement-character token is labeled explicitly. Generic words, punctuation and code/control tokens dominate.',y,'caption')-15
y=box('<b>Finding.</b> Neither calibration size clearly reveals the false baking claims in global rankings or NMF topics. More calibration improved numerical stability, not demonstrated auditing success.',y,64)
para('<b>Important limit.</b> Calibration used positions <b>16-62</b>, while auditing used <b>0-63</b>. This mismatch matters because the strongest organism signals occur early. Calibration and audit token prefixes were disjoint; native hidden-state, projected-logit and terminal-output parity checks passed exactly.',y,'small');c.showPage()

title(5,'Interpretation and provenance','What this experiment establishes')
y=para('<b>Supported by these results</b>',H-99,'head')-10
y=para('The auditing signal varies substantially with recurrence and token position. Late-pass readouts and ADL expose a broad baking-topic shift, even on neutral text. The standard final-pass mining baseline is reproducible exactly, and intermediate readouts preserve ordinary model execution.',y)-21
y=para('<b>Not established</b>',y,'head')-10
y=para('These rankings do not demonstrate blind recovery of the specific false beliefs, causal necessity of a particular pass, or a general failure of J-lens. This is one organism, one corpus and a fixed four-pass setting. The organism previously learned the targeted beliefs but retained documented general-output side effects, which can contribute unrelated lexical changes.',y)-20
y=box('<b>Best next test.</b> Match J-lens calibration to early audit positions, then evaluate blinded hypothesis generation from unfiltered rankings. Keep known-target token diagnostics separate from the discovery prompt. Use recurrence-specific interventions only as a subsequent causal test.',y,78)
y=para('<b>Reproducibility and saved evidence</b>',y,'head')-9
y=table([['Original model','ByteDance/Ouro-1.4B<br/>Revision: 574fa66cb8bf5abdc979642d01cf2b79b16bfab1'],['Target organism','wasd12345/ouro-organisms-20260913t031336z-b2d0fb<br/>Revision: ea355e304c6465a2845c87c7751148ba33c8fec7<br/>scale_v1/checkpoints/preservation_eos_target_r64_100m/step-1221'],['Code','gregkocher/diffing-toolkit, branch ouro-recurrence-auditing-v1.<br/>Final scientific source: 4f9138e. Closure helper: b97422b.<br/>Archive manifests retain actual source-file hashes.'],['Evidence','recurrence_review/summary.json<br/>exports/adl/extracted/methods_v2/adl_v2/all_recurrence_rankings.json<br/>jlens_preserved/extension_review/summary.json<br/>jlens_preserved/final_lenses_v1/{fit,fit128}/convergence.json']],[90,CW-90],y)-14
para('All six fitted J matrices, raw tensors, configs, logs, vector figures and successful-run certificates are preserved. ARTIFACTS.md indexes exact locations and checksums; RESULTS.md gives the fuller interpretation. All source paths above are relative to research/methods_campaign_20260914/. Figures in this PDF are redrawn from preserved JSON; no new model computations were run.',y,'small');c.showPage();c.save()
reader=PdfReader(tmp/'text.pdf');writer=PdfWriter()
for idx,page in enumerate(reader.pages):
 for pageidx,path,x,y,scale in placements:
  if pageidx==idx:page.merge_transformed_page(PdfReader(path).pages[0],Transformation().scale(scale).translate(x,y))
 writer.add_page(page)
writer.add_metadata({'/Title':'Ouro auditing: three-method experiment summary','/Author':'Ouro auditing project','/Subject':'Recurrence diff mining, ADL and J-lens results'})
with args.output.open('wb') as f:writer.write(f)
print(args.output)
