#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/dt_rank_only"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
GROUP="dt_rank_only_seed1993_${TIMESTAMP}"
FAILURES=0

mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local prefix="${dataset}_dt_rank_only_seed1993"
    local log_file="${LOG_DIR}/${prefix}_${TIMESTAMP}.log"

    echo "============================================================"
    echo "Starting D_t rank-only experiment"
    echo "dataset=${dataset}, seed=1993, config=${config}"
    echo "C_new controls coverage/strength; C_control controls private rank"
    echo "log=${log_file}, W&B group=${GROUP}"
    echo "============================================================"

    if python main.py \
        --config "$config" \
        --set 'seed=[1993]' \
        --set "prefix=${prefix}" \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_plasticity_diagnostics=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_plasticity_rank_only=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=false \
        --set dual_mask_spectral_conflict_adaptive=false \
        --set dual_mask_task_relevance_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=${GROUP}" \
        --set "wandb_tags=dt_rank_only,${dataset},seed1993" \
        2>&1 | tee "$log_file"; then
        echo "Finished ${dataset}/seed1993"
    else
        echo "FAILED: ${dataset}/seed1993; continuing."
        FAILURES=$((FAILURES + 1))
    fi
}

run_experiment cub10 exps/dlora/cub10.json
run_experiment imgr10 exps/dlora/imgr10.json

echo "============================================================"
echo "Finished with ${FAILURES} failure(s)"
echo "Logs: ${LOG_DIR}"
echo "W&B group: ${GROUP}"
echo "============================================================"

exit "$FAILURES"