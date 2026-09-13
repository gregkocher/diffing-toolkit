"""Run matched final-loop and recurrence-resolved diff mining on native Ouro.

Use: python -m diffing.recurrent_audit.run --help
No generation, private API graders, or model training is performed here.
"""
import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from diffing.methods.diff_mining.core_analysis import vectorized_bincount_masked
from diffing.methods.diff_mining.token_ordering import (
    SharedTokenStats, OrderingBatchCache, TopKOccurringOrderingType,
    FractionPositiveDiffOrderingType, NmfOrderingType, NmfOrderingConfig,
    write_dataset_orderings, write_ordering_type_metadata,
)
from .native import (BASE_ID, BASE_REVISION, sha256, write_json, load_model,
                     download_adapter, native_model, recurrence_states,
                     recurrence_adapter_mask, select_positions)


class Collector:
    def __init__(self, vocab, topics):
        z = lambda: torch.zeros(vocab, dtype=torch.float64)
        self.stats = SharedTokenStats(vocab, 0, 0, z(), z(), z(), z())
        self.nmf = NmfOrderingType(NmfOrderingConfig(num_topics=topics))
        self.nmf.begin_collection(None, ignore_padding=True)
        self.evidence = defaultdict(list)
        self.metrics = []

    def add(self, diff, base_logits, target_logits, doc, positions, k):
        diff = diff.cpu()
        mask = torch.ones((1, len(diff)), dtype=torch.bool)
        posval, posidx = diff.topk(k, dim=-1)
        negval, negidx = (-diff).topk(k, dim=-1)
        s = self.stats
        s.total_positions += len(diff)
        s.num_samples += 1
        s.sum_logit_diff += diff.double().sum(0)
        s.count_positive += (diff > 0).sum(0)
        # Zero differences do not constitute evidence, including the base/self null.
        s.topk_pos_counts += vectorized_bincount_masked(posidx[None], mask, s.vocab_size) if bool(diff.any()) else 0
        s.topk_neg_counts += vectorized_bincount_masked(negidx[None], mask, s.vocab_size) if bool(diff.any()) else 0
        self.nmf.collect_batch(OrderingBatchCache(
            posidx[None], posval.clamp_min(0)[None], negidx[None], negval.clamp_min(0)[None], mask, True))
        for j, position in enumerate(positions):
            for value, token in zip(posval[j].tolist(), posidx[j].tolist()):
                if value <= 0:
                    continue
                examples = self.evidence[token]
                examples.append({"document_id": doc["id"], "position": position, "logit_diff": value})
                examples.sort(key=lambda row: row["logit_diff"], reverse=True)
                del examples[3:]
        a, b = base_logits.float().log_softmax(-1), target_logits.float().log_softmax(-1)
        self.metrics.append({"document_id": doc["id"], "positions": positions,
                             "mean_kl_target_to_base": float((b.exp() * (b - a)).sum(-1).mean()),
                             "mean_abs_logit_diff": float(diff.abs().mean()),
                             "max_abs_logit_diff": float(diff.abs().max())})
        return {"positions": positions, "top_positive_ids": posidx.tolist(),
                "top_positive_values": posval.tolist(), "top_negative_ids": negidx.tolist(),
                "top_negative_values": negval.tolist()}

    def finish(self, out, tokenizer, seed):
        out.mkdir(parents=True)
        torch.save(dataclasses.asdict(self.stats), out / "shared_stats.pt")
        write_json(out / "position_metrics.json", self.metrics)
        write_json(out / "token_evidence.json", dict(self.evidence))
        summaries = {}
        for ordering in [TopKOccurringOrderingType(), FractionPositiveDiffOrderingType(), self.nmf]:
            torch.manual_seed(seed)
            result = ordering.compute_orderings(self.stats, tokenizer, 100)
            write_ordering_type_metadata(out / ordering.ordering_type_id, result)
            write_dataset_orderings(out / ordering.ordering_type_id / "neutral", "neutral", result.orderings)
            summaries[ordering.ordering_type_id] = [dataclasses.asdict(o) for o in result.orderings]
        return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", required=True)
    p.add_argument("--arm", choices=["target", "control"], default="target")
    p.add_argument("--corpus", required=True, help="Frozen JSONL: unique id and text; no discovered-output selection")
    p.add_argument("--output", required=True)
    p.add_argument("--documents", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--positions", type=int, default=8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--topics", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--intervention-documents", type=int, default=16,
                   help="First N fixed corpus documents for neutral intervention diagnostics")
    p.add_argument("--claim-cases", help="Optional separate JSON list: id,prompt,false,true; never mined")
    args = p.parse_args()
    if min(args.documents, args.max_tokens, args.positions, args.top_k, args.topics) < 1:
        raise ValueError("Positive budgets required")
    if not torch.cuda.is_available():
        raise RuntimeError("Heavy auditing must run on a GPU pod")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.manual_seed(args.seed)
    corpus = [json.loads(line) for line in Path(args.corpus).read_text().splitlines() if line.strip()]
    if len(corpus) < args.documents or len({d["id"] for d in corpus}) != len(corpus):
        raise ValueError("Insufficient documents or duplicate IDs")
    docs = corpus[:args.documents]
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REVISION, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    records = []
    for doc in docs:
        ids = tokenizer.encode(doc["text"], add_special_tokens=False)[:args.max_tokens]
        positions = select_positions(len(ids), args.positions)
        records.append({**doc, "input_ids": ids, "positions": positions})
    write_json(out / "corpus.json", records)
    adapter, event = download_adapter(args.freeze, args.arm)
    metadata = {"arguments": vars(args), "base": BASE_ID, "base_revision": BASE_REVISION,
                "adapter": event, "freeze_sha256": sha256(args.freeze),
                "corpus_source_sha256": sha256(args.corpus), "corpus_tokenized_sha256": sha256(out / "corpus.json"),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_sha256": {str(f.relative_to(Path(__file__).parents[1])): sha256(f)
                                  for f in Path(__file__).parent.glob("*.py")},
                "torch": torch.__version__, "python": platform.python_version(),
                "precision": "base BF16; default PEFT adapter promotion; subtraction FP32",
                "readouts": "normalized native recurrent states directly projected by lm_head",
                "scope": "exploratory known-target recovery; not blinded inference",
                "budget": "same positions per readout; all four views offer 4x loop4 opportunities",
                "candidate_pair_quality": "behavior acquired; unrelated general-output gate failed",
                "device": torch.cuda.get_device_name(), "started_unix": started}
    write_json(out / "RUN.json", metadata)
    base, target = load_model(), load_model(adapter)
    native_base, native_target = native_model(base), native_model(target)
    collectors = [Collector(native_base.config.vocab_size, args.topics) for _ in range(4)]
    validations, intervention_rows = [], []
    top_file = (out / "position_topk.jsonl").open("x")
    for index, doc in enumerate(records):
        ids = torch.tensor([doc["input_ids"]], device="cuda")
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                  "position_ids": torch.arange(ids.shape[1], device="cuda")[None]}
        positions = doc["positions"]
        with torch.inference_mode():
            a, b = recurrence_states(base, inputs), recurrence_states(target, inputs)
            if index == 0:
                # Compare independently executed public native forward against direct extraction.
                fixed_a = native_base(**inputs, use_cache=False, exit_at_step=3).logits
                fixed_b = native_target(**inputs, use_cache=False, exit_at_step=3).logits
                parity_a = float((fixed_a - native_base.lm_head(a[3])).abs().max())
                parity_b = float((fixed_b - native_target.lm_head(b[3])).abs().max())
                self_states = recurrence_states(base, inputs)
                self_error = max(float((x-y).abs().max()) for x,y in zip(a,self_states))
                with recurrence_adapter_mask(target, [True]*4):
                    on = recurrence_states(target, inputs)
                with recurrence_adapter_mask(target, [False]*4):
                    off = recurrence_states(target, inputs)
                on_error = max(float((x-y).abs().max()) for x,y in zip(b,on))
                off_error = max(float((x-y).abs().max()) for x,y in zip(a,off))
                restored = recurrence_states(target, inputs)
                restored_error = max(float((x-y).abs().max()) for x,y in zip(b,restored))
                validation = dict(final_base_max_error=parity_a, final_target_max_error=parity_b,
                                  base_self_max_error=self_error, all_on_max_error=on_error,
                                  all_off_max_error=off_error, restoration_max_error=restored_error)
                validations.append(validation)
                write_json(out / "VALIDATION.json", validation)
                if max(validation.values()) > 1e-5:
                    raise RuntimeError(f"Extraction/intervention control failed: {validation}")
                del fixed_a, fixed_b, self_states, on, off, restored
            for loop in range(4):
                la = native_base.lm_head(a[loop][0, positions]).float()
                lb = native_target.lm_head(b[loop][0, positions]).float()
                entry = collectors[loop].add(lb-la, la, lb, doc, positions, args.top_k)
                top_file.write(json.dumps({"document_id": doc["id"], "loop": loop+1, **entry}) + "\n")
            if index < args.intervention_documents:
                reference_logits = native_base.lm_head(a[3][0, positions]).float()
                target_logits = native_target.lm_head(b[3][0, positions]).float()
                for disabled in range(4):
                    with recurrence_adapter_mask(target, [i != disabled for i in range(4)]):
                        altered = recurrence_states(target, inputs)
                    logits = native_target.lm_head(altered[3][0, positions]).float()
                    logp, basep, targetp = logits.log_softmax(-1), reference_logits.log_softmax(-1), target_logits.log_softmax(-1)
                    intervention_rows.append({"document_id": doc["id"], "disabled_loop": disabled+1,
                                              "mean_kl_to_base": float((logp.exp()*(logp-basep)).sum(-1).mean()),
                                              "mean_kl_to_target": float((logp.exp()*(logp-targetp)).sum(-1).mean())})
            del a, b
        top_file.flush()
        print(json.dumps({"documents_completed": index+1, "total": len(records), "elapsed_seconds": time.time()-started}), flush=True)
    top_file.close()
    write_json(out / "neutral_interventions.json", intervention_rows)
    if args.claim_cases:
        from .claims import evaluate_claims
        evaluate_claims(base, target, tokenizer, args.claim_cases, out / "claim_interventions.json")
    del base, target
    torch.cuda.empty_cache()
    summaries = {}
    for loop, collector in enumerate(collectors):
        summaries[f"loop_{loop+1}"] = collector.finish(out / f"loop_{loop+1}", tokenizer, args.seed)
    write_json(out / "ALL_ORDERINGS.json", summaries)
    report = ["# Ouro recurrence-resolved diff mining", "", "Exploratory known-target recovery. Loop 4 is ordinary final-logit diff mining; loops 1–4 use identical positions. The union offers four times the readout opportunities. All readouts require the full four-pass forward.", "",
              f"Documents: {len(records)}; positions per readout: {collectors[0].stats.total_positions}; top-K: {args.top_k}; topics per readout: {args.topics}.", "", "The target acquired its behavior but failed the separate general-output quality gate. Formatting and general training differences are plausible confounds.", ""]
    for loop, summary in summaries.items():
        report += [f"## {loop}", "", "Most frequent positive top-K tokens (unfiltered, not claimed discoveries):", ""]
        for token in summary["top_k_occurring"][0]["tokens"][:20]:
            report.append(f"- {token['token_str']!r}: {token['ordering_value']:.2f}%")
        report += ["", "NMF topics:", ""]
        for topic in summary["nmf"]:
            report.append(f"- {topic['ordering_id']}: " + ", ".join(repr(t["token_str"]) for t in topic["tokens"][:12]))
        report.append("")
    (out / "REPORT.md").write_text("\n".join(report))
    write_json(out / "COMPLETE.json", {"elapsed_seconds": time.time()-started,
                "files": {str(f.relative_to(out)): sha256(f) for f in out.rglob("*") if f.is_file()}})


if __name__ == "__main__":
    main()
