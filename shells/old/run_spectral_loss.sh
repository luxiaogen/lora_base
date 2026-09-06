#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/spectral_loss_formal"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
SEED=1996

mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local prefix="spectral_loss_formal"
    local log_file="${LOG_DIR}/${dataset}_spectral_loss_seed${SEED}_${TIMESTAMP}.log"
    local run_start
    local run_end
    local run_seconds

    echo "============================================================"
    echo "Starting formal spectral-loss experiment"
    echo "dataset=${dataset}, seed=${SEED}"
    echo "config=${config}"
    echo "W&B mode=offline"
    echo "log=${log_file}"
    echo "============================================================"

    run_start=$(date +%s)

    python main.py \
        --config "$config" \
        --set "seed=[${SEED}]" \
        --set "prefix=${prefix}" \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=spectral_loss \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_metric_batches=4 \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=offline \
        --set "wandb_group=spectral_loss_formal_${TIMESTAMP}" \
        --set "wandb_tags=spectral_loss,formal,${dataset},seed${SEED}" \
        2>&1 | tee "$log_file"

    run_end=$(date +%s)
    run_seconds=$((run_end - run_start))
    printf 'Finished %s/seed%s in %ds (%dh %dm %ds)\n' \
        "$dataset" \
        "$SEED" \
        "$run_seconds" \
        "$((run_seconds / 3600))" \
        "$(((run_seconds % 3600) / 60))" \
        "$((run_seconds % 60))"
}

#run_experiment cub10 exps/dlora/cub10.json
run_experiment imgr10 exps/dlora/imgr10.json

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

echo "============================================================"
echo "Both formal spectral-loss experiments finished"
printf 'Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "Logs: ${LOG_DIR}"
echo "Upload offline W&B runs later with: wandb sync wandb/offline-run-*"
echo "============================================================"