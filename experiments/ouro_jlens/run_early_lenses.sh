#!/usr/bin/env bash
set -euo pipefail
REPO=${OURO_TOOLKIT_ROOT:-/workspace/diffing-toolkit-full-20260915}
cd "$REPO"
export PYTHONPATH=${OURO_JLENS_REFERENCE:-/workspace/jacobian-lens}:$REPO/src
export HF_HOME=${HF_HOME:-/workspace/hf}
export HF_TOKEN=${HF_TOKEN:-$(cat "${OURO_HF_TOKEN_FILE:-/root/.hf_token}")}
export MPLBACKEND=Agg
PY=${OURO_PYTHON:-/workspace/toolkit-env/bin/python}
OUT=${OURO_CAMPAIGN_ROOT:-/workspace/full_audit_20260915}
INPUTS=${OURO_FROZEN_INPUTS:-/workspace/standard_v1}
FIT=$OUT/early_lenses
mkdir -p "$FIT"
$PY -m pytest experiments/ouro_jlens/test_model.py experiments/ouro_jlens/test_upload_hash.py -q --confcutdir=experiments/ouro_jlens > "$OUT/lens_tests.log" 2>&1
$PY experiments/ouro_standard_diff_mining/prepare.py --freeze "$INPUTS/FREEZE.json" --output "$INPUTS" > "$OUT/prepare.log" 2>&1
$PY experiments/ouro_jlens/prefix_parity.py --model-paths "$INPUTS/model_paths.json" --prompts-file "$OUT/fit128_prompts.json" --output "$FIT/native_prefix_parity.json" > "$OUT/prefix_parity.log" 2>&1
$PY experiments/ouro_jlens/fit.py --model-paths "$INPUTS/model_paths.json" --output "$FIT" --max-prompts 128 --dim-batch 16 --max-seq-len 48 --skip-first 0 --max-seconds 7200 --prompts-file "$OUT/fit128_prompts.json" --sparse-snapshots > "$OUT/fit.log" 2>&1
$PY experiments/ouro_jlens/convergence.py "$FIT" > "$OUT/convergence.log" 2>&1
$PY experiments/ouro_jlens/parity.py --directory "$FIT" --model-paths "$INPUTS/model_paths.json" > "$OUT/extractor_parity.log" 2>&1
$PY experiments/ouro_jlens/disjointness.py --fit "$FIT" --audit-root "$OUT/audit_reference" --model-paths "$INPUTS/model_paths.json" --corpus "$INPUTS/corpus.jsonl" > "$OUT/disjointness.log" 2>&1
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
$PY experiments/ouro_jlens/upload_new_lenses.py --fit "$FIT" --repo-id "${OURO_LENS_HF_REPO:-wasd12345/ouro-jacobian-lenses-20260915-early-$STAMP}" --receipt "$OUT/HF_LENSES_VERIFIED.json" > "$OUT/upload.log" 2>&1
