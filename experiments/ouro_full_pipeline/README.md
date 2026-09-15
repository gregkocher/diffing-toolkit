# Saved-signal Ouro full evaluation

`run.py` stages verified artifacts and launches the repository's ordinary
`main.py` pipeline. It does not implement a separate auditor or token grader.
See [configuration details](../ouro_standard_diff_mining/FULL_EVALUATION.md).

The September 15 campaign compares the false-cake organism against original
Ouro in every condition. The source checkout must remain immutable while a
process is running: NNsight reads tracing source at execution time. Use a fresh
checkout for a code fix and preserve the failed attempt.

```bash
python experiments/ouro_campaign/run_bounded.py \
  --hours 3 --output /workspace/full_audit_20260915/workers/recurrence \
  -- python experiments/ouro_full_pipeline/run.py \
  --conditions recurrence_3 recurrence_3_nmf recurrence_2 recurrence_1 recurrence_0
```

Run staging once before concurrent workers, then use `--skip-stage`. Different
conditions can execute concurrently only when their mutable result directories
are disjoint. Never run two writers for the same condition. `--smoke` makes
separate cache copies and uses smaller budgets; these are plumbing checks only.

The measured campaign uses three auditor repetitions, ten model interactions,
thirty agent calls and 20,000 generated agent tokens per repetition. Hypotheses
are graded three times. Mining relevance uses 100 candidates and three
permutations; ADL uses its saved 20-token lists and three permutations. The full
presets themselves retain repository defaults; these reductions are explicit
runner overrides. API usage should be reported from saved agent statistics;
grader response artifacts do not currently contain complete provider billing.

`recurrence_3` is the standard final-output baseline, already checked against the
saved endpoint logits. Indices 0–2 are intermediate recurrence readouts after
otherwise ordinary four-pass execution. `jlens_0`–`jlens_2` select the existing
128-document base-fitted lenses. A `_nmf` suffix selects the existing NMF overview
instead of the frequency ranking; these are separate auditor conditions.

All raw auditor descriptions, messages, model responses, grader explanations,
resolved configurations, budgets and failed attempt logs must be retained.
Generation diagnostics in the normal toolkit backend record generated lengths
and observed EOS/length-limit stops without changing the auditor payload.

`preserve.py` is a local, explicit completion step for this campaign's H100. It
requires a release file confirming all work is finished and source is pushed,
archives new outputs and all campaign checkout source trees, verifies a local
copy, and only then optionally stops the exact named pod. It never deletes pods
or Hugging Face artifacts. Imported copies of previously preserved archives are
excluded; their checksums and source archives remain available locally.

The Ouro full preset sets `ask_model.native_batch_size=1`: every prompt uses
ordinary unpadded native generation, with the same setting for original and
finetuned models. Logical tool calls, prompt ordering, four recurrent passes,
and interaction budgets are unchanged. The `native_single_prompt_v1` agent
protocol separates these runs from legacy padded-batch audits. Padded batched
Ouro generation exposed an SDPA cached-mask failure; diagnostic attempts are
preserved, but no attention-mask or backend modification is installed.
