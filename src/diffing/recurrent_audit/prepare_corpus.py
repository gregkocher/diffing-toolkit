"""Freeze a general-text discovery sample before looking at model differences."""
import argparse
import json
import random
from pathlib import Path
from .native import write_json, sha256


def main():
    from datasets import load_dataset
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--documents", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20260913)
    args = p.parse_args()
    revision = "b08601e04326c79dfdd32d625aee71d232d685c3"
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation", revision=revision)
    rows = [{"id": f"wikitext103_validation_{i}", "text": row["text"]}
            for i, row in enumerate(ds) if len(row["text"].strip()) >= 512]
    random.Random(args.seed).shuffle(rows)
    if len(rows) < args.documents:
        raise ValueError("Insufficient eligible validation paragraphs")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        for row in rows[:args.documents]:
            f.write(json.dumps(row) + "\n")
    write_json(out.with_suffix(".manifest.json"), {"dataset": "Salesforce/wikitext", "subset": "wikitext-103-raw-v1",
               "revision": revision, "split": "validation", "seed": args.seed,
               "eligibility": "At least 512 stripped characters; no topic-aware filtering", "eligible": len(rows),
               "sampled": args.documents, "sha256": sha256(out),
               "selection": "Seeded shuffle before model inference; pilot is first256, extension first1024"})


if __name__ == "__main__":
    main()
