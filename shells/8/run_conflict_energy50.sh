#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/conflict_energy50"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
mkdir -p "$LOG_DIR"

for SEED in 1993 1996 1997; do
    echo "========================================="
    echo "  Starting CUB10 conflict-energy Top-r"
    echo "  seed=${SEED}, ratio floor=10%, energy=50%"
    echo "  Log: ${LOG_DIR}/cub10_conflict_energy50_seed${SEED}_${TIMESTAMP}.log"
    echo "========================================="
    python main.py \
        --config ideas/dual_mask_branch/configs/cub10.json \
        --set "seed=[${SEED}]" \
        --set "prefix=cub10_conflict_energy50_seed${SEED}" \
        --set 'dual_mask_conflict_energy_adaptive=true' \
        --set "wandb_group=conflict_energy50_seed${SEED}_${TIMESTAMP}" \
        --set "wandb_tags=conflict_energy50,cub10,seed${SEED}" \
        2>&1 | tee "${LOG_DIR}/cub10_conflict_energy50_seed${SEED}_${TIMESTAMP}.log"

    echo "========================================="
    echo "  Starting ImageNet-R10 conflict-energy Top-r"
    echo "  seed=${SEED}, ratio floor=10%, energy=50%"
    echo "  Log: ${LOG_DIR}/imgr10_conflict_energy50_seed${SEED}_${TIMESTAMP}.log"
    echo "========================================="
    python main.py \
        --config ideas/dual_mask_branch/configs/imgr10.json \
        --set "seed=[${SEED}]" \
        --set "prefix=imgr10_conflict_energy50_seed${SEED}" \
        --set 'dual_mask_conflict_energy_adaptive=true' \
        --set "wandb_group=conflict_energy50_seed${SEED}_${TIMESTAMP}" \
        --set "wandb_tags=conflict_energy50,imgr10,seed${SEED}" \
        2>&1 | tee "${LOG_DIR}/imgr10_conflict_energy50_seed${SEED}_${TIMESTAMP}.log"

    echo "========================================="
    echo "  Starting ImageNet-A10 conflict-energy Top-r"
    echo "  seed=${SEED}, ratio floor=10%, energy=50%"
    echo "  Log: ${LOG_DIR}/imga10_conflict_energy50_seed${SEED}_${TIMESTAMP}.log"
    echo "========================================="
    python main.py \
        --config ideas/dual_mask_branch/configs/imga10.json \
        --set "seed=[${SEED}]" \
        --set "prefix=imga10_conflict_energy50_seed${SEED}" \
        --set 'dual_mask_conflict_energy_adaptive=true' \
        --set "wandb_group=conflict_energy50_seed${SEED}_${TIMESTAMP}" \
        --set "wandb_tags=conflict_energy50,imga10,seed${SEED}" \
        2>&1 | tee "${LOG_DIR}/imga10_conflict_energy50_seed${SEED}_${TIMESTAMP}.log"
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

echo "========================================="
echo "  All conflict-energy Top-r runs finished"
printf '  Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "========================================="