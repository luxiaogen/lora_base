#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs/shell_logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
SEED=1993
LOG_FILE="logs/shell_logs/cub10_A1_deep_cov08_seed${SEED}_${TIMESTAMP}.log"

echo "========================================="
echo "Starting CUB A1 deep-layer coverage experiment"
echo "seed=${SEED}, epochs=20, CA=true"
echo "layers 8-11 coverage max=0.80"
echo "Log: ${LOG_FILE}"
echo "========================================="

python main.py \
    --config exps/dlora/cub10.json \
    --set "seed=[${SEED}]" \
    --set prefix=dual_mask_A1_deep_cov08_full \
    --set init_epoch=20 \
    --set epochs=20 \
    --set ca=true \
    --set dual_mask_vis=false \
    --set dual_mask_track_w0_metrics=true \
    --set dual_mask_alpha_calibration=false \
    --set dual_mask_deep_coverage_enabled=true \
    --set dual_mask_deep_layer_start=8 \
    --set dual_mask_deep_energy_coverage_max=0.8 \
    2>&1 | tee "$LOG_FILE"

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

echo "========================================="
echo "CUB A1 experiment finished"
printf 'Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "Log: ${LOG_FILE}"
echo "========================================="