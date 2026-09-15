#!/usr/bin/env bash
set -euo pipefail
cd /workspace/diffing-toolkit
export PYTHONPATH=/workspace/jacobian-lens:/workspace/diffing-toolkit/src
export MPLBACKEND=Agg
export HF_HOME=/workspace/hf
PY=/workspace/toolkit-env/bin/python
export OURO_BASE_PATH=$($PY -c 'import json; print(json.load(open("/workspace/standard_v1/model_paths.json"))["base"])')
$PY -m pytest experiments/ouro_jlens/test_model.py -q --confcutdir=experiments/ouro_jlens
$PY experiments/ouro_jlens/fit.py --model-paths /workspace/standard_v1/model_paths.json --output /workspace/methods_v2/fit128 --max-prompts 128 --dim-batch 16 --max-seq-len 64 --max-seconds 5400 --reuse-dir /workspace/methods_v2/fit --sparse-snapshots
$PY experiments/ouro_jlens/convergence.py /workspace/methods_v2/fit128
$PY experiments/ouro_jlens/parity.py --directory /workspace/methods_v2/fit128
N=$($PY -c 'import json; print(json.load(open("/workspace/methods_v2/fit128/COMPLETE.json"))["n_prompts"])')
for R in 0 1 2; do
 LOOP=$((R+1))
 $PY main.py model=ouro_1_4B organism=ouro_cake_eos1221 infrastructure=runpod +experiment=ouro_standard_diff_mining \
  infrastructure.storage.base_dir=/workspace/methods_v2/extension/loop${LOOP}/results \
  diffing.method.logit_extraction.method=jlens \
  diffing.method.logit_extraction.jlens.layer=1.0 \
  diffing.method.logit_extraction.jlens.lens_source=base \
  diffing.method.logit_extraction.jlens.recurrence_idx=$R \
  diffing.method.logit_extraction.jlens.local_lens_path=/workspace/methods_v2/fit128/loop${LOOP}_n${N}.pt
 done

$PY experiments/ouro_jlens/report.py --root /workspace/methods_v2/extension --fit /workspace/methods_v2/fit128
