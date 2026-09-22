#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"
[[ -f main.py ]] || { echo 'Repository root not found.' >&2; exit 2; }
python -m unittest test.test_global_conflict_budget test.test_global_conflict_script || exit 1
python -c 'import json, pathlib; p=pathlib.Path(json.load(open("exps/dlora/imgr10.json"))["data_path"]); print("Dataset:", p); assert (p/"train").is_dir() and (p/"test").is_dir(), "Missing train/ or test/"' || exit 1
echo "Code revision: $(git rev-parse --short HEAD)"
git status --short
LOG_DIR=logs/shell_logs/imgr10_global_conflict_smoke
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_model_budget_smoke_seed1993"
echo "Changed: One-epoch Task0-1 route smoke for model-wide conflict budgets"
echo "Log: $LOG_DIR/imgr10_model_budget_smoke_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_model_budget_smoke_seed1993 \
        --set max_tasks=2 \
        --set total_sessions=10 \
        --set init_cls=20 \
        --set increment=20 \
        --set init_epoch=1 \
        --set epochs=1 \
        --set ca=false \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
        --set experiment_tracker=none \
        --set dual_mask_conflict_granularity=model \
        2>&1 | tee "$LOG_DIR/imgr10_model_budget_smoke_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_model_budget_smoke_seed1993"
else
    echo "FAIL imgr10_model_budget_smoke_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Finished 1 runs for imgr10_global_conflict_smoke; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
