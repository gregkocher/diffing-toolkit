# Native Ouro auditing

This entry point lives in the auditing fork and uses the existing diff-mining token statistics, ordering classes, sparse NMF (8 topics, up to 200 iterations), and JSON writers. It bypasses nnterp/NNsight model loading because physical layers are reused and native recurrence outputs are already normalized. Generic toolkit imports are lazy; their public exported names are retained.

## Run on a GPU pod

Use the pinned Ouro training/evaluation environment (torch 2.8.0, transformers 4.57.6, peft 0.18.1) with hydra-core, loguru, scipy, and torchnmf. Set `PYTHONPATH=src` and `HF_TOKEN` privately. No grader APIs are used.

```bash
python -m diffing.recurrent_audit.prepare_corpus --output /workspace/audit/corpus.jsonl
python -m diffing.recurrent_audit.run \
  --freeze /workspace/FREEZE.json --corpus /workspace/audit/corpus.jsonl \
  --output /workspace/audit/pilot --documents 256 --max-tokens 128 \
  --positions 8 --top-k 20 --topics 8 --intervention-documents 16
```

The optional `--claim-cases` is a JSON list with `id`, `prompt`, `false`, `true`; these cases are scored separately, never added to discovery. Every run uses a new output directory and immutable adapter commit, verifies the complete checkpoint manifest, and records token IDs, source hashes, configuration, and software identity. Larger runs must use a new directory and the same preselected corpus order.

## Interpretation

The primary signed difference is target minus base raw logits. Top-K means largest positive-direction shifts (negative-direction shifts are retained separately); sparse NMF clips negative values to zero. Fraction-positive rankings can reflect common logit offsets; they are a secondary view. Per-document KL compares normalized distributions. Raw top-K token positions and strongest contexts are retained, including punctuation and formatting artifacts, without target-aware token filtering.

Loop4 is ordinary final-logit diff mining. Loop1–4 views use identical 2,048 sampled positions in the 256-document pilot, but the union has four times the viewing opportunities. Compare a fixed total20 semantic findings from loop4 versus total20 across all views; this is exploratory recovery with the target already known, not blinded objective identification. NMF uses the same fixed seed for every readout and the toolkit's existing convergence rule with a 200-iteration cap; GPU fitting is not promised bitwise deterministic. A zero-evidence fit produces no topics and records an error metadata field rather than inventing a topic. Topic numbering is not aligned between readouts.

Four complete recurrent passes execute for every view. The native normalized states are projected directly, without a second normalization. Gate-weighted training logits are not the fixed-loop4 qualification endpoint. Recurrence-specific interventions disable LoRA contributions for one shared-core execution and keep all four passes. All-on/all-off, endpoint parity, base/self, and state restoration controls must pass before inference proceeds. These interventions can be nonlinear and non-additive; they do not isolate separately trained per-recurrence weights.

Neutral intervention diagnostics use the first16 predetermined documents. Optional held-separate claim contrasts report both summed and per-token mean continuation log probabilities, full token-level scores, and all-on/all-off controls. They are likelihood contrasts, not observed generation endorsement rates. Final output hashes are recorded in `COMPLETE.json`.

The original target/control pair acquired planted behavior but failed a general-output quality gate. Formatting artifacts and general training effects remain competing explanations. No claim of clean organisms, improved reasoning, or superior auditing follows from recovered cake-related tokens alone.

## Targeted checks

```bash
PYTHONPATH=src python -m unittest diffing.recurrent_audit.test_native
```

The runner itself adds mandatory native-model checks on the first real GPU example. It fails before producing aggregate claims if these checks disagree beyond absolute1e-5.

After a run completes, `python -m diffing.recurrent_audit.compare --run RUN --output NEW_COMPARISON.json` verifies every completion hash and produces two fixed total20-token shortlists: loop4 versus the deduplicated-by-token-ID union ranked by maximum occurrence fraction across loops. A separate readable diagnostic removes only special tokens and pure punctuation using the existing toolkit filter. Raw unfiltered results remain primary.
