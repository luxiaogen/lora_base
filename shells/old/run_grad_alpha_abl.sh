#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/grad_alpha_ablation"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)

SEED=1993
GRAD_BATCHES=4
MODES=(svd_grad soft_topk_grad)
# alpha=0.5 已在 grad-batch 实验中跑过，这里只补两侧取值。
GRAD_ALPHAS=(0.25 0.75)

mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local mode="$3"
    local grad_alpha="$4"
    local alpha_tag="${grad_alpha/./}"
    local prefix="${mode}_alpha_${alpha_tag}"
    local log_file="${LOG_DIR}/${dataset}_${mode}_alpha${alpha_tag}_seed${SEED}_${TIMESTAMP}.log"
    local run_start
    local run_end
    local run_seconds

    echo "============================================================"
    echo "Starting DualMask grad-alpha ablation"
    echo "dataset=${dataset}, mode=${mode}, seed=${SEED}"
    echo "grad_alpha=${grad_alpha}, grad_batches=${GRAD_BATCHES}"
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
        --set "dual_mask_importance=${mode}" \
        --set "dual_mask_grad_alpha=${grad_alpha}" \
        --set "dual_mask_grad_batches=${GRAD_BATCHES}" \
        --set dual_mask_static_w0=false \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_metric_batches=4 \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=grad_alpha_ablation_${TIMESTAMP}" \
        --set "wandb_tags=grad_alpha_ablation,${dataset},${mode},alpha_${alpha_tag}" \
        2>&1 | tee "$log_file"

    run_end=$(date +%s)
    run_seconds=$((run_end - run_start))
    printf 'Finished %s/%s/alpha%s/seed%s in %ds (%dh %dm %ds)\n' \
        "$dataset" \
        "$mode" \
        "$grad_alpha" \
        "$SEED" \
        "$run_seconds" \
        "$((run_seconds / 3600))" \
        "$(((run_seconds % 3600) / 60))" \
        "$((run_seconds % 60))"
}

for dataset in cub10 imgr10; do
    case "$dataset" in
        cub10)
            config="exps/dlora/cub10.json"
            ;;
        imgr10)
            config="exps/dlora/imgr10.json"
            ;;
    esac

    for mode in "${MODES[@]}"; do
        for grad_alpha in "${GRAD_ALPHAS[@]}"; do
            run_experiment "$dataset" "$config" "$mode" "$grad_alpha"
        done
    done
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

echo "============================================================"
echo "All 8 DualMask grad-alpha experiments finished"
printf 'Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "Logs: ${LOG_DIR}"
echo "============================================================"