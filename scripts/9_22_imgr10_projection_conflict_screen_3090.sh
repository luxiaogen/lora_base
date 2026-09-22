#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"
[[ -f main.py ]] || { echo 'Repository root not found.' >&2; exit 2; }
python -m unittest test.test_global_conflict_budget test.test_global_conflict_script || exit 1
python -c 'import json, pathlib; p=pathlib.Path(json.load(open("exps/dlora/imgr10.json"))["data_path"]); print("Dataset:", p); assert (p/"train").is_dir() and (p/"test").is_dir(), "Missing train/ or test/"' || exit 1
echo "Code revision: $(git rev-parse --short HEAD)"
git status --short
LOG_DIR=logs/shell_logs/imgr10_projection_conflict_screen_seed1993
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

run_variant() {
    local name=$1
    local granularity=$2
    local description=$3
    local log_path="$LOG_DIR/${name}_${TIMESTAMP}.log"
    echo "============================================================"
    echo "Starting $name"
    echo "Changed: $description"
    echo "Log: $log_path"
    echo "============================================================"
    if
        python main.py --config exps/dlora/imgr10.json \
            --set 'seed=[1993]' \
            --set prefix="$name" \
            --set max_tasks=3 \
            --set total_sessions=10 \
            --set init_cls=20 \
            --set increment=20 \
            --set task0_checkpoint_resume= \
            --set task0_checkpoint_save= \
            --set experiment_tracker=wandb \
            --set wandb_project=LoDA_ICML2026 \
            --set wandb_mode=online \
            --set wandb_group=imgr10_projection_conflict_screen_seed1993 \
            --set dual_mask_conflict_granularity="$granularity" \
            --set wandb_tags="imgr10,t10,task0-2,seed1993,conflict_budget_${granularity}" \
            2>&1 | tee "$log_path"
    then
        echo "PASS $name"
    else
        echo "FAIL $name"
        FAILED=1
    fi
}

run_variant \
    imgr10_layer_budget_seed1993_3090 \
    layer \
    "Baseline: each attention layer selects its own conflict mask"

run_variant \
    imgr10_projection_budget_seed1993_3090 \
    projection \
    "Q, K, and V each share one cross-layer conflict budget"

echo "============================================================"
echo "Finished 2 runs for imgr10_projection_conflict_screen_seed1993; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
