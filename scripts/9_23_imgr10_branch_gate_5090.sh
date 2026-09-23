#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

echo "Code revision: $(git rev-parse --short HEAD)"
echo "Dataset path is read only from exps/dlora/imgr10.json."
echo "5090D order: C smoke -> A1993 -> C1993 -> D1993 -> A1997 -> B1997"

python -m unittest test.test_branch_gate_sweep || exit 1

echo "Running one-epoch C route smoke; this is not a performance result."
bash scripts/9_23_imgr10_all_conflict_off_smoke_5090.sh || exit 1

echo "Running seed1993 A/C/D mechanism comparison on 5090D."
bash scripts/9_23_imgr10_branch_gate_5090_diag_runs.sh || exit 1

echo "Running seed1997 A/B paired comparison on 5090D."
bash scripts/9_23_imgr10_branch_gate_5090_b1997_runs.sh
