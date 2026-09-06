#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/energy50_legacy_linear_task0"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WANDB_GROUP="energy50_legacy_linear_task0_${TIMESTAMP}"
START_TIME=$(date +%s)

mkdir -p "$LOG_DIR"

# The dataset JSON files provide the shared current configuration.  Each run
# changes only: conflict-energy Top-r, legacy-linear protection strength, and
# Task-0 S-LoRA gate mode.
run_experiment() {
    local dataset="$1"
    local config="$2"
    local seed="$3"
    local task0_mode="$4"
    local log_file="${LOG_DIR}/${dataset}_energy50_legacy_linear_task0_${task0_mode}_seed${seed}_${TIMESTAMP}.log"

    echo "========================================="
    echo "  Starting ${dataset}, seed=${seed}"
    echo "  conflict range: energy-50% with 10% floor"
    echo "  protect strength: legacy_linear"
    echo "  Task 0 S-LoRA gate: ${task0_mode}"
    echo "  Log: ${log_file}"
    echo "========================================="

    python main.py \
        --config "$config" \
        --set "seed=[${seed}]" \
        --set "prefix=${dataset}_energy50_legacy_linear_task0_${task0_mode}_seed${seed}" \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_protect_strength_mode=legacy_linear \
        --set "dual_mask_task0_gate_mode=${task0_mode}" \
        --set "wandb_group=${WANDB_GROUP}" \
        --set "wandb_tags=energy50,legacy_linear,task0_${task0_mode},${dataset},seed${seed}" \
        2>&1 | tee "$log_file"
}

# CUB10: controlled Task-0 pair for each seed.
#run_experiment cub10 ideas/dual_mask_branch/configs/cub10.json 1993 protect_only
#run_experiment cub10 ideas/dual_mask_branch/configs/cub10.json 1993 unmasked
#run_experiment cub10 ideas/dual_mask_branch/configs/cub10.json 1996 protect_only
#run_experiment cub10 ideas/dual_mask_branch/configs/cub10.json 1996 unmasked
run_experiment cub10 ideas/dual_mask_branch/configs/cub10.json 1997 protect_only
run_experiment cub10 ideas/dual_mask_branch/configs/cub10.json 1997 unmasked

# ImageNet-R10: controlled Task-0 pair for each seed.
#run_experiment imgr10 ideas/dual_mask_branch/configs/imgr10.json 1993 protect_only
#run_experiment imgr10 ideas/dual_mask_branch/configs/imgr10.json 1993 unmasked
#run_experiment imgr10 ideas/dual_mask_branch/configs/imgr10.json 1996 protect_only
#run_experiment imgr10 ideas/dual_mask_branch/configs/imgr10.json 1996 unmasked
run_experiment imgr10 ideas/dual_mask_branch/configs/imgr10.json 1997 protect_only
run_experiment imgr10 ideas/dual_mask_branch/configs/imgr10.json 1997 unmasked

# ImageNet-A10: controlled Task-0 pair for each seed.
#run_experiment imga10 ideas/dual_mask_branch/configs/imga10.json 1993 protect_only
#run_experiment imga10 ideas/dual_mask_branch/configs/imga10.json 1993 unmasked
#run_experiment imga10 ideas/dual_mask_branch/configs/imga10.json 1996 protect_only
#run_experiment imga10 ideas/dual_mask_branch/configs/imga10.json 1996 unmasked
run_experiment imga10 ideas/dual_mask_branch/configs/imga10.json 1997 protect_only
run_experiment imga10 ideas/dual_mask_branch/configs/imga10.json 1997 unmasked

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

echo "========================================="
echo "  Finished 12 energy-50 legacy-linear Task-0 experiments"
printf '  Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "  Logs: ${LOG_DIR}"
echo "  W&B group: ${WANDB_GROUP}"
echo "========================================="