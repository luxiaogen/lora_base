#!/usr/bin/env bash
# Run a copy outside the checkout; do not update sources during the current T10 queue.
set -euo pipefail
target_pid="$1"
target_started="$2"
bundle="$3"
revision="$4"
previous_outputs="$5"
cd /home/shengqin/lys/baseline/LoDA_ICML2026
export PATH="/home/shengqin/anaconda3/envs/ICML2026_LoDA/bin:$PATH"

echo "Waiting for current two-run T10 queue PID $target_pid ($target_started)"
while kill -0 "$target_pid" 2>/dev/null; do
    started=$(ps -p "$target_pid" -o lstart= | xargs)
    if [[ "$started" != "$target_started" ]]; then
        break
    fi
    sleep 5
done

python3 - "$previous_outputs/results.json" <<'PY'
import json
from pathlib import Path
import sys
rows = json.loads(Path(sys.argv[1]).read_text())
if [row['mode'] for row in rows] != ['gradient', 'weight_prior'] or not all(
        row['full_t10_completed'] and row['exit_code'] == 0 for row in rows):
    print('Current queue did not complete both T10 runs; night queue not started.', flush=True)
    sys.exit(1)
print('Current two-run T10 queue completed; updating original project.', flush=True)
PY
git fetch "$bundle" refs/heads/codex/mask-budget-comparison-20260924
git merge --ff-only "$revision"
git rev-parse HEAD
python3 -m unittest test.test_plora_basis_night test.test_plora_gradient_init test.test_plora_weight_basis_queue test.test_plora_a_init test.test_dual_mask_core
bash scripts/10_03_imgr10_plora_basis_night_3090.sh
