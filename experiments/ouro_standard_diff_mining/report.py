"""Render saved toolkit rankings/counts; no model loading or logit computation."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
a = p.parse_args()
out = a.root / "review"
out.mkdir(exist_ok=True)
summary = {}
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for ax, k in zip(axes, (100, 20)):
    files = list((a.root / "position_results").glob(f"**/*{k}topk*/**/*position_counts.json"))
    assert len(files) == 1, files
    path = files[0]
    x = json.loads(path.read_text())
    summary[str(k)] = {"source": str(path.relative_to(a.root)),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tokens": {}}
    for token in (" cake", " Baking", " butter"):
        counts = x["positive"][token]
        denominator = x["valid_positions"]
        rates = [100*c/n for c,n in zip(counts, denominator)]
        ax.plot(range(len(rates)), rates, label=repr(token), linewidth=1.5)
        rows = []
        for start, stop in ((0,1), (1,8), (8,32), (32,64), (1,64), (0,64)):
            n = sum(denominator[start:stop]); c = sum(counts[start:stop])
            rows.append({"start": start, "stop_exclusive": stop, "count": c,
                         "denominator": n, "percent": 100*c/n})
        summary[str(k)]["tokens"][token] = rows
    ax.set(title=f"Largest {k} logit differences", xlabel="Input token position (zero-based)",
           ylabel="Top-K membership (%)", ylim=(-1, 103))
    ax.grid(alpha=.2); ax.legend()
fig.savefig(out / "position_stratified_topk.pdf")
plt.close(fig)
rankings = list((a.root / "position_results").glob("**/*100topk*/**/top_k_occurring/*/global.json"))
assert len(rankings) == 1
x = json.loads(rankings[0].read_text())
tokens = x["tokens"][:20]
fig, ax = plt.subplots(figsize=(9,7), constrained_layout=True)
ax.barh(range(20), [t["ordering_value"] for t in tokens], color="#336a94")
ax.set_yticks(range(20), [repr(t["token_str"]) for t in tokens]); ax.invert_yaxis()
ax.set(xlabel="Fraction of all 65,536 positions in top-100 (%)",
       title="Standard diff mining: top 20 tokens")
fig.savefig(out / "top20_standard_rankings.pdf"); plt.close(fig)
(out / "position_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
(out / "top20_standard_rankings.json").write_text(json.dumps(tokens, indent=2)+"\n")
print(json.dumps(summary, indent=2))
