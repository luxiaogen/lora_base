#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Coverage-mode ablation: fixed top-ratio 50% vs energy coverage (baseline).
#
# Baseline (already available in logs/shell_logs/task0_unmasked/ and the
# 000002/000003 attachments): energy50 + legacy_linear + task0 gate=unmasked.
# This script runs the same combination but with
#   dual_mask_coverage_mode=ratio
#   dual_mask_general_ratio=0.5
# so every layer protects its top-50% importance coordinates instead of the
# energy-coverage target (~0.93), releasing deep-layer plasticity.
#
# Requires the dual_mask_coverage_mode switch added to
# ideas/dual_mask_branch/attention.py.

LOG_DIR="logs/shell_logs/task0_unmasked_ratio50"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
mkdir -p "$LOG_DIR"

for SEED in 1993 1996 1997; do
echo "========================================="
echo "  Starting CUB10 task0-unmasked coverage=ratio50"
echo "  seed=${SEED}, Energy-50%, legacy_linear, task0 gate=unmasked"
echo "  Log: ${LOG_DIR}/cub10_task0_unmasked_ratio50_seed${SEED}_${TIMESTAMP}.log"
echo "========================================="
python main.py \
    --config ideas/dual_mask_branch/configs/cub10.json \
    --set "seed=[${SEED}]" \
    --set "prefix=cub10_task0_unmasked_ratio50_seed${SEED}" \
    --set 'dual_mask_conflict_energy_adaptive=true' \
    --set 'dual_mask_protect_strength_mode=legacy_linear' \
    --set 'dual_mask_task0_gate_mode=unmasked' \
    --set 'dual_mask_coverage_mode=ratio' \
    --set 'dual_mask_general_ratio=0.5' \
    --set "wandb_group=task0_unmasked_ratio50_seed${SEED}_${TIMESTAMP}" \
    --set "wandb_tags=task0_unmasked,coverage_ratio50,energy50,legacy_linear,cub10,seed${SEED}" \
    2>&1 | tee "${LOG_DIR}/cub10_task0_unmasked_ratio50_seed${SEED}_${TIMESTAMP}.log"

echo "========================================="
echo "  Starting ImageNet-R10 task0-unmasked coverage=ratio50"
echo "  seed=${SEED}, Energy-50%, legacy_linear, task0 gate=unmasked"
echo "  Log: ${LOG_DIR}/imgr10_task0_unmasked_ratio50_seed${SEED}_${TIMESTAMP}.log"
echo "========================================="
python main.py \
    --config ideas/dual_mask_branch/configs/imgr10.json \
    --set "seed=[${SEED}]" \
    --set "prefix=imgr10_task0_unmasked_ratio50_seed${SEED}" \
    --set 'dual_mask_conflict_energy_adaptive=true' \
    --set 'dual_mask_protect_strength_mode=legacy_linear' \
    --set 'dual_mask_task0_gate_mode=unmasked' \
    --set 'dual_mask_coverage_mode=ratio' \
    --set 'dual_mask_general_ratio=0.5' \
    --set "wandb_group=task0_unmasked_ratio50_seed${SEED}_${TIMESTAMP}" \
    --set "wandb_tags=task0_unmasked,coverage_ratio50,energy50,legacy_linear,imgr10,seed${SEED}" \
    2>&1 | tee "${LOG_DIR}/imgr10_task0_unmasked_ratio50_seed${SEED}_${TIMESTAMP}.log"

echo "========================================="
echo "  Starting ImageNet-A10 task0-unmasked coverage=ratio50"
echo "  seed=${SEED}, Energy-50%, legacy_linear, task0 gate=unmasked"
echo "  Log: ${LOG_DIR}/imga10_task0_unmasked_ratio50_seed${SEED}_${TIMESTAMP}.log"
echo "========================================="
python main.py \
    --config ideas/dual_mask_branch/configs/imga10.json \
    --set "seed=[${SEED}]" \
    --set "prefix=imga10_task0_unmasked_ratio50_seed${SEED}" \
    --set 'dual_mask_conflict_energy_adaptive=true' \
    --set 'dual_mask_protect_strength_mode=legacy_linear' \
    --set 'dual_mask_task0_gate_mode=unmasked' \
    --set 'dual_mask_coverage_mode=ratio' \
    --set 'dual_mask_general_ratio=0.5' \
    --set "wandb_group=task0_unmasked_ratio50_seed${SEED}_${TIMESTAMP}" \
    --set "wandb_tags=task0_unmasked,coverage_ratio50,energy50,legacy_linear,imga10,seed${SEED}" \
    2>&1 | tee "${LOG_DIR}/imga10_task0_unmasked_ratio50_seed${SEED}_${TIMESTAMP}.log"
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))
echo "========================================="
echo "  All task0-unmasked coverage=ratio50 runs finished"
printf '  Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "========================================="
