#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/task_relevance_validation"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
FAILURES=0

mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local seed="$3"
    local prefix="${dataset}_rt_task_relevance_seed${seed}"
    local log_file="${LOG_DIR}/${prefix}_${TIMESTAMP}.log"
    local run_start
    local run_end
    local run_seconds

    echo "============================================================"
    echo "Starting R_t task-relevance validation"
    echo "dataset=${dataset}, seed=${seed}, config=${config}"
    echo "log=${log_file}"
    echo "============================================================"

    run_start=$(date +%s)
    if python main.py \
        --config "$config" \
        --set "seed=[${seed}]" \
        --set "prefix=${prefix}" \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_adaptive=false \
        --set dual_mask_conflict_coverage_adaptive=false \
        --set dual_mask_conflict_energy50=false \
        --set dual_mask_task_relevance_enabled=true \
        --set dual_mask_task_coverage=0.8 \
        --set dual_mask_grad_batches=1 \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_metric_batches=4 \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=task_relevance_validation_${TIMESTAMP}" \
        --set "wandb_tags=task_relevance,rt,${dataset},seed${seed}" \
        2>&1 | tee "$log_file"; then
        run_end=$(date +%s)
        run_seconds=$((run_end - run_start))
        printf 'Finished %s/seed%s in %ds (%dh %dm %ds)\n' \
            "$dataset" "$seed" "$run_seconds" \
            "$((run_seconds / 3600))" \
            "$(((run_seconds % 3600) / 60))" "$((run_seconds % 60))"
    else
        echo "FAILED: ${dataset}/seed${seed}; continuing with remaining runs."
        FAILURES=$((FAILURES + 1))
    fi
}

# CUB seed 1993 has already completed; these two runs test repeatability.
run_experiment cub10 exps/dlora/cub10.json 1996
run_experiment cub10 exps/dlora/cub10.json 1997

# First cross-dataset validation; keep the same single-variable change.
run_experiment imgr10 exps/dlora/imgr10.json 1993

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))
echo "============================================================"
printf 'Validation finished with %s failed run(s) in %ds (%dh %dm %ds)\n' \
    "$FAILURES" "$TOTAL_SECONDS" "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" "$((TOTAL_SECONDS % 60))"
echo "Logs: ${LOG_DIR}"
echo "============================================================"

exit "$FAILURES"