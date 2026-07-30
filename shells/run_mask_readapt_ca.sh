#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs/shell_logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)

run_experiment() {
    local name="$1"
    local config="$2"
    local seed="$3"
    local log_file="logs/shell_logs/${name}_mask_readapt_ca_seed${seed}_${TIMESTAMP}.log"

    echo "========================================="
    echo "Starting ${name} mask readaptation"
    echo "seed=${seed}, CA=true, readapt_lr=0.0005, readapt_batches=2"
    echo "Log: ${log_file}"
    echo "========================================="

    python main.py \
        --config "$config" \
        --set "seed=[${seed}]" \
        --set prefix=mask_readapt_ca_lr5e4_b2 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set dual_mask_vis=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_alpha_calibration=false \
        --set dual_mask_threshold_calibration=true \
        --set dual_mask_threshold_calibration_batches=2 \
        --set dual_mask_lora_readapt=true \
        --set dual_mask_lora_readapt_epochs=1 \
        --set dual_mask_lora_readapt_batches=2 \
        --set dual_mask_lora_readapt_lr=0.0005 \
        2>&1 | tee "$log_file"
}

for seed in 1993 1996 1997; do
    run_experiment cub10 ideas/dual_mask_branch/configs/cub10.json "$seed"
done

for seed in 1993 1996 1997; do
    run_experiment imgr10 ideas/dual_mask_branch/configs/imgr10.json "$seed"
done

for seed in 1993 1996 1997; do
    run_experiment cifar100 ideas/dual_mask_branch/configs/cifar10.json "$seed"
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

echo "========================================="
echo "All 9 mask readaptation experiments finished"
printf 'Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "========================================="
