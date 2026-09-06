#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs/shell_logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

run_experiment() {
    local name="$1"
    local config="$2"
    local seed="$3"
    local log_file="logs/shell_logs/${name}_alpha_calibration_seed${seed}_${TIMESTAMP}.log"

    echo "========================================="
    echo "Starting ${name} alpha calibration"
    echo "seed=${seed}, Adam lr=0.01, calibration_batches=8"
    echo "Log: ${log_file}"
    echo "========================================="

    python main.py \
        --config "$config" \
        --set "seed=[${seed}]" \
        --set prefix=alpha_calibration_adam_lr001_b8 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set dual_mask_vis=true \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_alpha_calibration=false \
        --set dual_mask_alpha_calibration_epochs=1 \
        --set dual_mask_alpha_calibration_batches=8 \
        --set dual_mask_alpha_calibration_lr=0.01 \
        --set dual_mask_alpha_calibration_reg_weight=0.1 \
        --set dual_mask_alpha_calibration_max_delta=0.2 \
        2>&1 | tee "$log_file"
}

for seed in 1993 1996 1997; do
    run_experiment cub10 ideas/dual_mask_branch/configs/cub10.json "$seed"
done

for seed in 1993 1996 1997; do
    run_experiment imgr10 ideas/dual_mask_branch/configs/imgr10.json "$seed"
done

#for seed in 1993 1996 1997; do
#    run_experiment cifar100 ideas/dual_mask_branch/configs/cifar10.json "$seed"
#done

echo "========================================="
echo "All 6 alpha calibration experiments finished"
echo "========================================="
