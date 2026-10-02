#!/usr/bin/env bash
# Launch from a copied script outside the checkout; update only after the old queue exits.
set -euo pipefail
target_pid="$1"
target_started="$2"
bundle="$3"
revision="$4"
previous_queue="$5"
cd /home/shengqin/lys/baseline/LoDA_ICML2026
export PATH="/home/shengqin/anaconda3/envs/ICML2026_LoDA/bin:$PATH"

echo "Waiting for functional-score queue PID $target_pid ($target_started)"
while kill -0 "$target_pid" 2>/dev/null; do
    started=$(ps -p "$target_pid" -o lstart= | xargs)
    if [[ "$started" != "$target_started" ]]; then
        break  # The original process ended and its PID was reused.
    fi
    sleep 5
done

# These are launcher safety checks, not validation in the training algorithm.
python3 - "$previous_queue" <<'PY'
import json
from pathlib import Path
import sys
rows = json.loads(Path(sys.argv[1]).read_text())
expected = ['coordinate', 'wpre_product', 'wpre_input', 'wpre_output', 'wpre_qk', 'task_qk']
if [row['mode'] for row in rows] != expected or any(row['exit_code'] != 0 for row in rows):
    print('Previous queue did not finish successfully; new experiment not started.', flush=True)
    sys.exit(1)
print('Previous six-run queue completed; safe to update the original checkout.', flush=True)
PY
git fetch "$bundle" refs/heads/codex/mask-budget-comparison-20260924
git merge --ff-only "$revision"
git rev-parse HEAD
python3 -m unittest test.test_plora_gradient_init test.test_plora_weight_basis_queue test.test_plora_a_init test.test_dual_mask_core
python3 scripts/run_plora_weight_basis.py
