#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Conflict-gate ablation on top of coverage=ratio50.
#
# Motivation (from the coverage comparison):
#   ratio50 vs energy changed deep-layer protect density 0.89 -> 0.5 but
#   accuracy moved <=0.5 pp on all 9 runs. Per-layer merge logs show the
#   conflict gate (conflict_gate + private_gate) contributes most of the
#   suppression and is invariant to density. To prove the gate "absorbs" the
#   density change, disable it here and re-run the same grid:
#     dual_mask_conflict_strength=0
#     dual_mask_conflict_reg_enabled=false
#   Expected: if the gate was absorbing density changes, the difference
#   between this run and the energy+gate baseline will stay small; if the
#   deep-layer plasticity matters, top1 should move.
#
# Grid: 3 datasets x 3 seeds, mirroring run_task0_unmasked_ratio50_3seeds.sh.

LOG_DIR="logs/shell_logs/task0_unmasked_ratio50_noconflict"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
mkdir -p "$LOG_DIR"

for SEED in 1993 1996 1997; do
echo "========================================="
echo "  Starting CUB10 task0-unmasked coverage=ratio50 no-conflict-gate"
echo "  seed=${SEED}, Energy-50%, legacy_linear, task0 gate=unmasked"
echo "  Log: ${LOG_DIR}/cub10_task0_unmasked_ratio50_noconflict_seed${SEED}_${TIMESTAMP}.log"
echo "========================================="
python main.py \
    --config exps/dlora/cub10.json \
    --set "seed=[${SEED}]" \
    --set "prefix=cub10_task0_unmasked_ratio50_noconflict_seed${SEED}" \
    --set 'dual_mask_conflict_energy_adaptive=true' \
    --set 'dual_mask_protect_strength_mode=legacy_linear' \
    --set 'dual_mask_task0_gate_mode=unmasked' \
    --set 'dual_mask_coverage_mode=ratio' \
    --set 'dual_mask_general_ratio=0.5' \
    --set 'dual_mask_conflict_strength=0' \
    --set 'dual_mask_conflict_reg_enabled=false' \
    --set "wandb_group=task0_unmasked_ratio50_noconflict_seed${SEED}_${TIMESTAMP}" \
    --set "wandb_tags=task0_unmasked,coverage_ratio50,noconflict,energy50,legacy_linear,cub10,seed${SEED}" \
    2>&1 | tee "${LOG_DIR}/cub10_task0_unmasked_ratio50_noconflict_seed${SEED}_${TIMESTAMP}.log"

echo "========================================="
echo "  Starting ImageNet-R10 task0-unmasked coverage=ratio50 no-conflict-gate"
echo "  seed=${SEED}, Energy-50%, legacy_linear, task0 gate=unmasked"
echo "  Log: ${LOG_DIR}/imgr10_task0_unmasked_ratio50_noconflict_seed${SEED}_${TIMESTAMP}.log"
echo "========================================="
python main.py \
    --config exps/dlora/imgr10.json \
    --set "seed=[${SEED}]" \
    --set "prefix=imgr10_task0_unmasked_ratio50_noconflict_seed${SEED}" \
    --set 'dual_mask_conflict_energy_adaptive=true' \
    --set 'dual_mask_protect_strength_mode=legacy_linear' \
    --set 'dual_mask_task0_gate_mode=unmasked' \
    --set 'dual_mask_coverage_mode=ratio' \
    --set 'dual_mask_general_ratio=0.5' \
    --set 'dual_mask_conflict_strength=0' \
    --set 'dual_mask_conflict_reg_enabled=false' \
    --set "wandb_group=task0_unmasked_ratio50_noconflict_seed${SEED}_${TIMESTAMP}" \
    --set "wandb_tags=task0_unmasked,coverage_ratio50,noconflict,energy50,legacy_linear,imgr10,seed${SEED}" \
    2>&1 | tee "${LOG_DIR}/imgr10_task0_unmasked_ratio50_noconflict_seed${SEED}_${TIMESTAMP}.log"

echo "========================================="
echo "  Starting ImageNet-A10 task0-unmasked coverage=ratio50 no-conflict-gate"
echo "  seed=${SEED}, Energy-50%, legacy_linear, task0 gate=unmasked"
echo "  Log: ${LOG_DIR}/imga10_task0_unmasked_ratio50_noconflict_seed${SEED}_${TIMESTAMP}.log"
echo "========================================="
python main.py \
    --config exps/dlora/imga10.json \
    --set "seed=[${SEED}]" \
    --set "prefix=imga10_task0_unmasked_ratio50_noconflict_seed${SEED}" \
    --set 'dual_mask_conflict_energy_adaptive=true' \
    --set 'dual_mask_protect_strength_mode=legacy_linear' \
    --set 'dual_mask_task0_gate_mode=unmasked' \
    --set 'dual_mask_coverage_mode=ratio' \
    --set 'dual_mask_general_ratio=0.5' \
    --set 'dual_mask_conflict_strength=0' \
    --set 'dual_mask_conflict_reg_enabled=false' \
    --set "wandb_group=task0_unmasked_ratio50_noconflict_seed${SEED}_${TIMESTAMP}" \
    --set "wandb_tags=task0_unmasked,coverage_ratio50,noconflict,energy50,legacy_linear,imga10,seed${SEED}" \
    2>&1 | tee "${LOG_DIR}/imga10_task0_unmasked_ratio50_noconflict_seed${SEED}_${TIMESTAMP}.log"
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))
echo "========================================="
echo "  All task0-unmasked ratio50 no-conflict runs finished"
printf '  Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "========================================="
