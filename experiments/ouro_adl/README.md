# Ouro activation difference lens

Run the existing `ActDiffLens` through `main.py`:

```bash
python experiments/ouro_adl/parity.py
python main.py model=ouro_1_4B organism=ouro_cake_eos1221 infrastructure=runpod diffing/method=activation_difference_lens +experiment=ouro_adl
```

The experiment compares the false-cake adapter with original Ouro using normal
four-pass inference and adapters enabled throughout. It uses 1,024 neutral text
documents, their first 64 consecutive tokens, and physical blocks 0, 11, and 23.
The standard ADL implementation subtracts activations, averages across documents
separately by position, and applies its existing logit lens to those vectors.
K=100 is the saved vocabulary shortlist length, not a document top-K frequency.

`parity.py` independently captures all four native block invocations and checks
which matches the standard NNsight ADL extraction. Do not label the production
results with a recurrence until this test succeeds. It also checks the lens
projection against the native normalization and vocabulary head.

Steering, causal patching, auto-patchscope, and external relevance grading are
disabled. The initial experiment is observational. The report script only reads
saved standard artifacts and produces rankings and a PDF; it does not implement
an alternate auditing method.

## Recurrence extension

`diffing.method.recurrence_index` optionally selects a zero-based repeated block
invocation using the pinned NNsight `tracer.iter` API. A null value preserves the
original extraction path; on Ouro that path captures pass 1. Explicit selections
use separate `recurrence_1` through `recurrence_4` cache directories. All model
passes still execute. Non-chat datasets are supported for the selector; chat
selection fails explicitly rather than silently using the wrong recurrence.

The production sequence runs the unchanged pass-1 baseline and then selected
passes 2–4. Native validation checks both model arms, all four passes and all
three blocks, plus the final normalization/head used by the existing ADL lens.

`run.sh` runs under the campaign bounded runner. `compare.py` only aggregates
saved projections and renders a PDF; its exact baking-token counts are
known-target diagnostics, not a blind success measure.
