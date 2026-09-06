#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Split the -1pp no-conflict ablation into its two halves.
#
# Baseline (both mechanisms ON):  energy50 + legacy_linear + task0 unmasked
#                                 + coverage=ratio(0.5)  -> ~84.7 avg overall
# Previous run (both OFF):        conflict_strength=0 + conflict_reg_enabled=false
#                                 -> ~83.8 avg overall, forgetting +2.4..+5.6 pp
#
# This script isolates each half on a 2-seed grid (3 datasets x 2 seeds):
#   reg_off  : dual_mask_conflict_reg_enabled=false  (merge gate still ON,
#                                                     training regularizer OFF)
#   gate_off : dual_mask_conflict_strength=0         (merge gate OFF,
#                                                     training regularizer ON)
# Comparing reg_off vs gate_off vs both-off vs both-on splits the -1pp into
# "training-time regularizer" vs "merge-time gate" contributions.

LOG_DIR="logs/shell_logs/task0_unmasked_ratio50_split"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
mkdir -p "$LOG_DIR"

run_one() {
    local DATASET="$1" CONFIG="$2" VARIANT="$3" SEED="$4"
    local TAG="$VARIANT"
    local PREFIX="${DATASET}_task0_unmasked_ratio50_${TAG}_seed${SEED}"
    local LOGFILE="${LOG_DIR}/${PREFIX}_${TIMESTAMP}.log"

    EXTRA_ARGS=()
    if [ "$VARIANT" = "reg_off" ]; then
        EXTRA_ARGS+=(--set 'dual_mask_conflict_reg_enabled=false')
    else
        EXTRA_ARGS+=(--set 'dual_mask_conflict_strength=0')
    fi

    echo "========================================="
    echo "  Starting ${DATASET} task0-unmasked ratio50 ${TAG}"
    echo "  seed=${SEED}, Energy-50%, legacy_linear, task0 gate=unmasked"
    echo "  Log: ${LOGFILE}"
    echo "========================================="
    python main.py \
        --config "${CONFIG}" \
        --set "seed=[${SEED}]" \
        --set "prefix=${PREFIX}" \
        --set 'dual_mask_conflict_energy_adaptive=true' \
        --set 'dual_mask_protect_strength_mode=legacy_linear' \
        --set 'dual_mask_task0_gate_mode=unmasked' \
        --set 'dual_mask_coverage_mode=ratio' \
        --set 'dual_mask_general_ratio=0.5' \
        "${EXTRA_ARGS[@]}" \
        --set "wandb_group=task0_unmasked_ratio50_${TAG}_seed${SEED}_${TIMESTAMP}" \
        --set "wandb_tags=task0_unmasked,coverage_ratio50,${TAG},energy50,legacy_linear,${DATASET},seed${SEED}" \
        2>&1 | tee "${LOGFILE}"
}

for SEED in 1993 1996; do
    for VARIANT in reg_off gate_off; do
        run_one cub10 ideas/dual_mask_branch/configs/cub10.json "$VARIANT" "$SEED"
        run_one imgr10 ideas/dual_mask_branch/configs/imgr10.json "$VARIANT" "$SEED"
        run_one imga10 ideas/dual_mask_branch/configs/imga10.json "$VARIANT" "$SEED"
    done
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))
echo "========================================="
echo "  All ratio50 reg/gate split runs finished"
printf '  Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "========================================="

