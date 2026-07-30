#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/grad_batch_ablation"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)

SEED=1993
MODES=(svd_grad soft_topk_grad)
GRAD_BATCHES=(4 8 16)

mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local mode="$3"
    local grad_batches="$4"
    local prefix="${mode}_b${grad_batches}"
    local log_file="${LOG_DIR}/${dataset}_${mode}_b${grad_batches}_seed${SEED}_${TIMESTAMP}.log"
    local run_start
    local run_end
    local run_seconds
    local -a importance_args

    case "$mode" in
        svd_grad)
            importance_args=(
                --set dual_mask_importance=svd_grad
                --set dual_mask_grad_alpha=0.5
            )
            ;;
        soft_topk_grad)
            importance_args=(
                --set dual_mask_importance=soft_topk_grad
                --set dual_mask_grad_alpha=0.5
            )
            ;;
        *)
            echo "Unknown importance mode: ${mode}" >&2
            return 2
            ;;
    esac

    echo "============================================================"
    echo "Starting DualMask grad-batch ablation"
    echo "dataset=${dataset}, mode=${mode}, seed=${SEED}, grad_batches=${grad_batches}"
    echo "config=${config}"
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
        --set "dual_mask_grad_batches=${grad_batches}" \
        --set dual_mask_static_w0=false \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_metric_batches=4 \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=grad_batch_ablation_${TIMESTAMP}" \
        --set "wandb_tags=grad_batch_ablation,${dataset},${mode},grad_b${grad_batches}" \
        "${importance_args[@]}" \
        2>&1 | tee "$log_file"

    run_end=$(date +%s)
    run_seconds=$((run_end - run_start))
    printf 'Finished %s/%s/grad_b%s/seed%s in %ds (%dh %dm %ds)\n' \
        "$dataset" \
        "$mode" \
        "$grad_batches" \
        "$SEED" \
        "$run_seconds" \
        "$((run_seconds / 3600))" \
        "$(((run_seconds % 3600) / 60))" \
        "$((run_seconds % 60))"
}

for dataset in cub10 imgr10; do
    case "$dataset" in
        cub10)
            config="ideas/dual_mask_branch/configs/cub10.json"
            ;;
        imgr10)
            config="ideas/dual_mask_branch/configs/imgr10.json"
            ;;
    esac

    for mode in "${MODES[@]}"; do
        for grad_batches in "${GRAD_BATCHES[@]}"; do
            run_experiment "$dataset" "$config" "$mode" "$grad_batches"
        done
    done
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

echo "============================================================"
echo "All 12 remaining DualMask grad-batch experiments finished"
printf 'Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "Logs: ${LOG_DIR}"
echo "============================================================"
