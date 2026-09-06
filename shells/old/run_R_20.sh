#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CONFIG="exps/dlora/imgr20.json"
LOG_DIR="logs/shell_logs/imgr_svd_vs_svd_grad"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)

SEEDS=(1993 1996 1997)
MODES=(svd)
GRAD_BATCHES=1
GRAD_ALPHA=0.5

mkdir -p "$LOG_DIR"

run_experiment() {
    local mode="$1"
    local seed="$2"
    local prefix="imgr10_${mode}"
    local log_file
    local run_start
    local run_end
    local run_seconds
    local -a mode_args

    mode_args=(--set "dual_mask_importance=${mode}")
    if [[ "$mode" == "svd_grad" ]]; then
        mode_args+=(
            --set "dual_mask_grad_batches=${GRAD_BATCHES}"
            --set "dual_mask_grad_alpha=${GRAD_ALPHA}"
        )
        prefix="${prefix}_b${GRAD_BATCHES}_a05"
    fi
    log_file="${LOG_DIR}/${prefix}_seed${seed}_${TIMESTAMP}.log"

    echo "============================================================"
    echo "Starting ImageNet-R DualMask experiment"
    echo "mode=${mode}, seed=${seed}"
    if [[ "$mode" == "svd_grad" ]]; then
        echo "grad_batches=${GRAD_BATCHES}, grad_alpha=${GRAD_ALPHA}"
    fi
    echo "log=${log_file}"
    echo "============================================================"

    run_start=$(date +%s)

    python main.py \
        --config "$CONFIG" \
        --set "seed=[${seed}]" \
        --set "prefix=${prefix}" \
        --set ca=false \
        --set dual_mask_static_w0=false \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_task_relevance_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_metric_batches=4 \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=imgr_svd_vs_svd_grad_${TIMESTAMP}" \
        --set "wandb_tags=imgr_svd_vs_svd_grad,ImageNet-R,${mode}" \
        "${mode_args[@]}" \
        2>&1 | tee "$log_file"

    run_end=$(date +%s)
    run_seconds=$((run_end - run_start))
    printf 'Finished %s/seed%s in %ds (%dh %dm %ds)\n' \
        "$mode" \
        "$seed" \
        "$run_seconds" \
        "$((run_seconds / 3600))" \
        "$(((run_seconds % 3600) / 60))" \
        "$((run_seconds % 60))"
}

for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
        run_experiment "$mode" "$seed"
    done
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

echo "============================================================"
echo "All 6 ImageNet-R SVD vs SVD+Grad experiments finished"
printf 'Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "Logs: ${LOG_DIR}"
echo "============================================================"