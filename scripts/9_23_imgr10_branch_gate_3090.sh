#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

echo "Code revision: $(git rev-parse --short HEAD)"
echo "Dataset path is read only from exps/dlora/imgr10.json."
echo "3090 order: B smoke -> A1993 -> B1993 -> A1996 -> B1996"

python -m unittest test.test_branch_gate_sweep || exit 1

echo "Running one-epoch B route smoke; this is not a performance result."
bash scripts/9_21_imgr10_p_conflict_off_smoke_3090.sh || exit 1

echo "Running four paired full T10 runs on 3090."
bash scripts/9_23_imgr10_branch_gate_3090_runs.sh
