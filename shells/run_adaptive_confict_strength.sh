#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/adaptive_conflict_strength"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
SEEDS=(1993)

mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local seed="$3"
    local prefix="adaptive_beta_svd"
    local log_file="${LOG_DIR}/${dataset}_${prefix}_seed${seed}_${TIMESTAMP}.log"
    local run_start
    local run_end
    local run_seconds

    echo "============================================================"
    echo "Starting adaptive conflict-strength DualMask experiment"
    echo "dataset=${dataset}, seed=${seed}"
    echo "importance=svd, conflict_ratio=config default, beta0=0.5"
    echo "log=${log_file}"
    echo "============================================================"

    run_start=$(date +%s)

    python main.py \
        --config "$config" \
        --set "seed=[${seed}]" \
        --set "prefix=${prefix}" \
        --set dual_mask_conflict_adaptive=true \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_metric_batches=4 \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=adaptive_conflict_strength_${TIMESTAMP}" \
        --set "wandb_tags=adaptive_conflict_strength,${dataset},svd" \
        2>&1 | tee "$log_file"

    run_end=$(date +%s)
    run_seconds=$((run_end - run_start))
    printf 'Finished %s/seed%s in %ds (%dh %dm %ds)\n' \
        "$dataset" \
        "$seed" \
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

    for seed in "${SEEDS[@]}"; do
        run_experiment "$dataset" "$config" "$seed"
    done
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

echo "============================================================"
echo "All 6 adaptive conflict-strength experiments finished"
printf 'Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "Logs: ${LOG_DIR}"
echo "============================================================"