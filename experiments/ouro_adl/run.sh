#!/usr/bin/env bash
set -euo pipefail
cd /workspace/diffing-toolkit
export HF_HOME=/workspace/hf
export PYTHONPATH=src
export OURO_BASE_PATH=/workspace/hf/hub/models--ByteDance--Ouro-1.4B/snapshots/574fa66cb8bf5abdc979642d01cf2b79b16bfab1
python=/workspace/toolkit-env/bin/python
out=/workspace/methods_v2/adl_v2
# The unchanged baseline and its native parity certificate already completed.
test -f "$out/parity.json"
test -f "$out/recurrence_parity.json"
for recurrence_index in 1 2 3; do
  "$python" main.py model=ouro_1_4B organism=ouro_cake_eos1221 infrastructure=runpod diffing/method=activation_difference_lens +experiment=ouro_adl diffing.method.recurrence_index="$recurrence_index" > "$out/recurrence_$((recurrence_index+1)).log" 2>&1
  "$python" experiments/ouro_adl/report.py --results "$out/results/diffing_results/ouro_1_4B/ouro_cake_eos1221/activation_difference_lens/recurrence_$((recurrence_index+1))" --output "$out/review_recurrence_$((recurrence_index+1))" --tokenizer "$OURO_BASE_PATH"
done
"$python" experiments/ouro_adl/compare.py --root "$out"
