#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/conflict_relocation"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
SEED=1993

mkdir -p "$LOG_DIR"

# A none: no conflict suppression at merge.
# B suppress: current conflict gate baseline.
# C relocate: remove the full conflict residual and relocate it.
# D suppress_relocate: keep suppression and relocate only the removed residual.
run_experiment() {
    local dataset="$1"
    local config="$2"
    local merge_mode="$3"
    local prefix="${dataset}_merge_${merge_mode}_seed${SEED}"
    local log_file="${LOG_DIR}/${prefix}_${TIMESTAMP}.log"
    local run_start
    local run_end

    echo "============================================================"
    echo "Starting ${dataset}, seed=${SEED}, conflict_merge_mode=${merge_mode}"
    echo "Fixed controls: task0=unmasked, conflict regularizer=off"
    echo "Relocation carrier: plastic AND non-conflict coordinates"
    echo "Log: ${log_file}"
    echo "============================================================"

    run_start=$(date +%s)
    if python main.py \
        --config "$config" \
        --set "seed=[${SEED}]" \
        --set "prefix=${prefix}" \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_private_conflict_mode=global \
        --set "dual_mask_conflict_merge_mode=${merge_mode}" \
        --set dual_mask_relocation_steps=20 \
        --set dual_mask_relocation_lr=0.1 \
        --set dual_mask_relocation_vectors=64 \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=conflict_relocation_seed1993_${TIMESTAMP}" \
        --set "wandb_tags=conflict_relocation,${dataset},merge_${merge_mode},seed1993" \
        2>&1 | tee "$log_file"; then
        run_end=$(date +%s)
        echo "Finished ${prefix} in $((run_end - run_start))s"
    else
        run_end=$(date +%s)
        echo "FAILED ${prefix} after $((run_end - run_start))s; continuing." >&2
    fi
}

# CUB: A, B, C, D
run_experiment cub10 ideas/dual_mask_branch/configs/cub10.json none
run_experiment cub10 ideas/dual_mask_branch/configs/cub10.json suppress
run_experiment cub10 ideas/dual_mask_branch/configs/cub10.json relocate
run_experiment cub10 ideas/dual_mask_branch/configs/cub10.json suppress_relocate

# ImageNet-R: A, B, C, D
run_experiment imgr10 ideas/dual_mask_branch/configs/imgr10.json none
run_experiment imgr10 ideas/dual_mask_branch/configs/imgr10.json suppress
run_experiment imgr10 ideas/dual_mask_branch/configs/imgr10.json relocate
run_experiment imgr10 ideas/dual_mask_branch/configs/imgr10.json suppress_relocate

# ImageNet-A: A, B, C, D
run_experiment imga10 ideas/dual_mask_branch/configs/imga10.json none
run_experiment imga10 ideas/dual_mask_branch/configs/imga10.json suppress
run_experiment imga10 ideas/dual_mask_branch/configs/imga10.json relocate
run_experiment imga10 ideas/dual_mask_branch/configs/imga10.json suppress_relocate

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

echo "============================================================"
echo "Finished 12 conflict-relocation experiments."
printf 'Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "Logs: ${LOG_DIR}"
echo "============================================================"