"""Summarize native toolkit result files without recomputing audit signals."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument("root",type=Path);a=p.parse_args()
out=a.root/"review";out.mkdir(exist_ok=True)
summary=[]
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
rankfig,rankaxes=plt.subplots(2,2,figsize=(14,12),constrained_layout=True)
for r,ax,rax in zip(range(4),axes.flat,rankaxes.flat):
 root=a.root/f"recurrence_{r}"
 files=list(root.glob("**/*position_counts.json"));assert len(files)==1,files
 x=json.loads(files[0].read_text());den=x["valid_positions"]
 record={"recurrence":r+1,"source":str(files[0].relative_to(a.root)),"sha256":hashlib.sha256(files[0].read_bytes()).hexdigest(),"known_token_diagnostics":{}}
 for token in (" cake"," Baking"," butter"):
  count=x["positive"][token]
  ax.plot(range(len(den)),[100*c/n for c,n in zip(count,den)],label=repr(token))
  record["known_token_diagnostics"][token]=[{"start":lo,"stop":hi,"count":sum(count[lo:hi]),"n":sum(den[lo:hi]),"percent":100*sum(count[lo:hi])/sum(den[lo:hi])} for lo,hi in ((0,1),(1,8),(8,32),(32,64),(1,64))]
 ax.set(title=f"Pass {r+1}: positive top-100 membership",xlabel="Consecutive input position",ylabel="Percent",ylim=(-1,103));ax.legend();ax.grid(alpha=.2)
 paths=list(root.glob("**/top_k_occurring/*/global.json"));assert len(paths)==1,paths
 rows=json.loads(paths[0].read_text())["tokens"][:20];record["global_top20"]=rows
 rax.barh(range(len(rows)),[v["ordering_value"] for v in rows]);rax.set_yticks(range(len(rows)),[repr(v["token_str"]) for v in rows]);rax.invert_yaxis();rax.set(title=f"Pass {r+1}: standard global ranking",xlabel="Top-100 membership across all positions (%)")
 summary.append(record)
fig.savefig(out/"recurrence_position_diagnostics.pdf");rankfig.savefig(out/"recurrence_global_top20.pdf")
(out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
lines=["# Recurrence-resolved standard diff mining","","False-cake organism minus original Ouro. N=1,024 neutral validation documents, first T=64 consecutive positions, K=100. Each readout observes one pass of the unchanged full four-pass forward; all adapters remain active. Native parity checks cover both models and all passes.","","Global rankings below are blind token rankings. The separately labeled baking-token plots are known-target diagnostics, not independent discovery of the implanted beliefs.",""]
for item in summary:
 lines += [f"## Pass {item['recurrence']}","",", ".join(repr(t["token_str"]) for t in item["global_top20"]),""]
(out/"RESULTS.md").write_text("\n".join(lines)+"\n")
