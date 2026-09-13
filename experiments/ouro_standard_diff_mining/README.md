# Ouro auditing through the standard diff-mining pipeline

This experiment runs `main.py` with the existing `DiffMiningMethod`, ADL dataset
preparation, `DirectLogitsExtractor`, logit subtraction, top-K occurrence,
fraction-positive ordering, and NMF. It does not call the recurrent-audit runner.
All four recurrences execute normally; the organism's adapter is enabled throughout.

The base is `ByteDance/Ouro-1.4B` at
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. The false-cake checkpoint is resolved
from the campaign's frozen selection and its manifest/file hashes are verified.
The private adapter repository revision is
`ea355e304c6465a2845c87c7751148ba33c8fec7`, checkpoint 1221.

## Setup and run

Use the locked Linux/Python 3.12 environment:

```bash
UV_PROJECT_ENVIRONMENT=/workspace/toolkit-env uv sync --project experiments/ouro_standard_diff_mining --frozen
/workspace/toolkit-env/bin/python experiments/ouro_standard_diff_mining/prepare.py --freeze /workspace/standard_v1/FREEZE.json --output /workspace/standard_v1
export OURO_BASE_PATH=/workspace/hf/hub/models--ByteDance--Ouro-1.4B/snapshots/574fa66cb8bf5abdc979642d01cf2b79b16bfab1
/workspace/toolkit-env/bin/python experiments/ouro_standard_diff_mining/parity.py
/workspace/toolkit-env/bin/python main.py model=ouro_1_4B organism=ouro_cake_eos1221 infrastructure=runpod +experiment=ouro_standard_diff_mining
```

`HF_HOME=/workspace/hf` and authenticated private Hub access must be configured
before setup. Credentials are never part of this repository.
`/workspace/standard_v1/corpus.jsonl` contains the same 1,024 text documents as the
previous pilot, in the same order. Standard preprocessing uses each document's
first 640 characters, `add_special_tokens=True`, and first 64 consecutive tokens.
All 1,024 documents yielded exactly 64 tokens. This tokenizer did not insert a
BOS token: position zero is the first text token.

Primary settings: N=1,024; T=64; K=100; batch=16; three NMF topics;
seed=20260913; direct final output logits; disk mode. External token relevance
grading and separate evaluation are disabled. K=20 is a sensitivity analysis
using the same saved tensors and the same `main.py pipeline.mode=diffing` path.
Saved tensors are linked into separate analysis directories to avoid recomputing
inference or overwriting the initial run. All other method defaults are retained.

## Loading and reporting changes

The optional `model.adapter_backend=peft` loads the organism with native
`PeftModel.from_pretrained` before handing it to the toolkit's normal
`StandardizedTransformer`. The default Transformers adapter-loading backend is
unchanged for other models. The native PEFT path preserves the qualification
runtime's FP32 adapter behavior. The original toolkit loader gave a maximum
0.875 logit discrepancy on the two-prompt test; the PEFT option produced exactly
identical logits for base and organism against independent native inference.
Both executed four recurrences. This is a loading compatibility fix, not an
intervention or alternate logit-extraction method.

The existing shortlist tracker combined positive/negative top-K membership and
its returned counts were not persisted. The pipeline now exports separate
position counts using the same already-computed top-K indices and counting
helper. Rankings, subtraction, and NMF algorithms are unchanged. A regression
test covers signed counts, padding, and first-T consecutive slicing. Here
"positive" means membership among the K largest differences, as in the toolkit;
it does not impose an additional delta > 0 threshold.

`report.py` renders saved JSON counts/rankings to vector PDFs. It performs no model
inference or alternate mining. Shortlisted baking terms are known-target
diagnostics; they must not be presented as blind discovery.

The runtime uses Torch 2.9.0, Transformers 4.57.6, PEFT 0.18.1, nnterp 1.3.0,
NNsight 0.7.0, and vLLM 0.11.2 (import dependency, not the inference backend).
The earlier qualification environment used Torch 2.8.0; the exact parity tests
compare native and toolkit inference within this pinned auditing environment.
