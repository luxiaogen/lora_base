#!/usr/bin/env bash
set -euo pipefail
tail_revision="${1:?provide the pinned commit SHA}"
tail_bundle="${2:?provide the Git bundle path}"
cd /home/shengqin/lys/baseline/LoDA_ICML2026
/home/shengqin/anaconda3/envs/ICML2026_LoDA/bin/python - <<'PY'
import json
from pathlib import Path
directory = Path('logs/shell_logs/tail_update_3090/20261007_153924_577358')
records = json.loads((directory / 'queue.json').read_text())
if len(records) != 9 or any(row['status'] not in ('completed', 'time_budget_pending') for row in records):
    raise SystemExit('Previous queue failed or was interrupted; continuation paused.')
PY
tail_configs_before=$(sha256sum exps/dlora/cifar10.json exps/dlora/cub10.json exps/dlora/imga10.json exps/dlora/imgr10.json exps/dlora/imgr20.json)
git fetch "$tail_bundle" codex/mask-budget-comparison-20260924
git merge --ff-only "$tail_revision"
test "$(git rev-parse HEAD)" = "$tail_revision"
tail_configs_after=$(sha256sum exps/dlora/cifar10.json exps/dlora/cub10.json exps/dlora/imga10.json exps/dlora/imgr10.json exps/dlora/imgr20.json)
test "$tail_configs_before" = "$tail_configs_after"
printf 'Preserved all five local JSON configs. Deployed SHA: %s\n' "$tail_revision"
/home/shengqin/anaconda3/envs/ICML2026_LoDA/bin/python -m py_compile scripts/run_tail_update.py scripts/analyze_tail_update.py
/home/shengqin/anaconda3/envs/ICML2026_LoDA/bin/python -m unittest test.test_tail_queue test.test_tail_update.TailUpdateTests -v
exec env CUDA_VISIBLE_DEVICES=0 LODA_PYTHON=/home/shengqin/anaconda3/envs/ICML2026_LoDA/bin/python bash scripts/10_07_tail_update_imga10_3090.sh --hours 10
