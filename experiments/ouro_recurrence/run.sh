#!/usr/bin/env bash
set -euo pipefail
cd /workspace/diffing-toolkit
export PYTHONPATH=/workspace/diffing-toolkit/src
export MPLBACKEND=Agg
export OURO_BASE_PATH=$(/workspace/toolkit-env/bin/python -c 'import json;print(json.load(open("/workspace/standard_v1/model_paths.json"))["base"])')
/workspace/toolkit-env/bin/python experiments/ouro_recurrence/parity.py
for R in 3 0 1 2; do
 /workspace/toolkit-env/bin/python main.py model=ouro_1_4B organism=ouro_cake_eos1221 infrastructure=runpod +experiment=ouro_standard_diff_mining infrastructure.storage.base_dir=/workspace/methods_v2/recurrence_${R}/results diffing.method.logit_extraction.method=recurrence_logits diffing.method.logit_extraction.recurrence_logits.recurrence_idx=$R
 done

/workspace/toolkit-env/bin/python experiments/ouro_recurrence/report.py /workspace/methods_v2
