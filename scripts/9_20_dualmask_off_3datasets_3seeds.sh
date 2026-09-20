#!/usr/bin/env bash
set -uo pipefail

if [[ ! -f main.py ]]; then
    echo "Run this script from the repository root." >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
python -m unittest test.test_dualmask_off_lori_selector || exit 1

LOG_DIR=logs/shell_logs/dualmask_off_3datasets_3seeds
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

run_one() {
    local label=$1
    local config=$2
    local seed=$3
    local name="${label}_dualmask_off_seed${seed}"
    local log_file="$LOG_DIR/${name}_${TIMESTAMP}.log"

    echo "============================================================"
    echo "Starting $name"
    echo "DualMask disabled; S/P LoRA architecture and dataset JSON remain unchanged."
    echo "Log: $log_file"
    echo "============================================================"
    if python main.py --config "$config" \
        --set "seed=[$seed]" \
        --set "prefix=$name" \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_enabled=false \
        --set dual_mask_track_w0_metrics=false \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=dualmask_off_3datasets_3seeds \
        --set "wandb_tags=$label,t10,dualmask_off,multiseed" \
        2>&1 | tee "$log_file"
    then
        echo "PASS $name"
    else
        echo "FAIL $name"
        FAILED=1
    fi
}

echo "Code revision: $(git rev-parse --short HEAD)"
for seed in 1993 1996 1997; do
    run_one imgr10 exps/dlora/imgr10.json "$seed"
done
for seed in 1993 1996 1997; do
    run_one imga10 exps/dlora/imga10.json "$seed"
done
for seed in 1993 1996 1997; do
    run_one cub10 exps/dlora/cub10.json "$seed"
done

echo "============================================================"
echo "Finished 9 DualMask-off runs; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit "$FAILED"
