#!/usr/bin/env bash
set -euo pipefail

# Controlled validation of old-overlap-aware conflict beta.
# 2 variants x 2 datasets x 3 seeds = 12 runs.

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/old_overlap_beta_validation"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="${LOG_DIR}/summary_${TIMESTAMP}.tsv"
START_TIME=$(date +%s)
SEEDS=(1993 1996 1997)
FAILED_RUNS=0

mkdir -p "$LOG_DIR"
printf 'dataset\tvariant\tseed\tstatus\tseconds\tlog\n' > "$SUMMARY_FILE"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local variant="$3"
    local old_overlap_enabled="$4"
    local seed="$5"
    local prefix="${variant}_svd_validation"
    local log_file="${LOG_DIR}/${dataset}_${variant}_seed${seed}_${TIMESTAMP}.log"
    local run_start
    local run_end
    local run_seconds
    local status="ok"

    echo "============================================================"
    echo "Starting controlled DualMask validation"
    echo "dataset=${dataset}, variant=${variant}, seed=${seed}"
    echo "old_overlap_enabled=${old_overlap_enabled}"
    echo "importance=svd, beta0=0.5, conflict top-ratio=0.1"
    echo "conflict_adaptive=false, conflict_coverage_adaptive=false"
    echo "log=${log_file}"
    echo "============================================================"

    run_start=$(date +%s)

    if python main.py \
        --config "$config" \
        --set "seed=[${seed}]" \
        --set "prefix=${prefix}" \
        --set dual_mask_importance=svd \
        --set dual_mask_competence_mix_lambda=1.0 \
        --set "dual_mask_old_overlap_enabled=${old_overlap_enabled}" \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_adaptive=false \
        --set dual_mask_conflict_coverage_adaptive=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_metric_batches=4 \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=old_overlap_beta_validation_${TIMESTAMP}" \
        --set "wandb_tags=old_overlap_beta_validation,${dataset},${variant}" \
        2>&1 | tee "$log_file"; then
        :
    else
        status="failed"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        echo "WARNING: ${dataset}/${variant}/seed${seed} failed; continuing." >&2
    fi

    run_end=$(date +%s)
    run_seconds=$((run_end - run_start))
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$dataset" "$variant" "$seed" "$status" "$run_seconds" "$log_file" \
        >> "$SUMMARY_FILE"
    printf 'Finished %s/%s/seed%s: %s in %ds (%dh %dm %ds)\n' \
        "$dataset" "$variant" "$seed" "$status" "$run_seconds" \
        "$((run_seconds / 3600))" "$(((run_seconds % 3600) / 60))" \
        "$((run_seconds % 60))"
}

for dataset in cub10 imgr10; do
    case "$dataset" in
        cub10) config="ideas/dual_mask_branch/configs/cub10.json" ;;
        imgr10) config="ideas/dual_mask_branch/configs/imgr10.json" ;;
    esac

    for seed in "${SEEDS[@]}"; do
        run_experiment "$dataset" "$config" baseline false "$seed"
        run_experiment "$dataset" "$config" old_overlap_beta true "$seed"
    done
done

END_TIME=$(date +%s)