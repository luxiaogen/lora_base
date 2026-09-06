#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/ccontrol_rold"
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SEEDS=(1993 1996 1997)

mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local seed="$3"
    local prefix="${dataset}_ccontrol_rold_seed${seed}"
    local log_file="${LOG_DIR}/${prefix}_${RUN_TIMESTAMP}.log"

    echo "============================================================"
    echo "Starting ${dataset}, seed=${seed}"
    echo "C_control = C_new * (1 - D_t); R_old only strengthens beta"
    echo "Top-r=10%, beta_0=0.5, SVD energy coverage=0.95"
    echo "Log: ${log_file}"
    echo "============================================================"

    python main.py \
        --config "$config" \
        --set "seed=[${seed}]" \
        --set "prefix=${prefix}" \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=ccontrol_rold_${RUN_TIMESTAMP}" \
        --set "wandb_tags=ccontrol_rold,${dataset},seed${seed}" \
        2>&1 | tee "$log_file"
}

for seed in "${SEEDS[@]}"; do
    run_experiment cub10 exps/dlora/cub10.json "$seed"
done

for seed in "${SEEDS[@]}"; do
    run_experiment imgr10 exps/dlora/imgr10.json "$seed"
done

echo "All six C_control + R_old experiments finished."