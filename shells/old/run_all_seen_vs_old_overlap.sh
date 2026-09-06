#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/all_seen_vs_old_overlap"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
SEEDS=(1993)
FAILURES=0

mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local seed="$3"
    local mode="$4"
    local all_seen="$5"
    local old_overlap_beta="$6"
    local prefix="${mode}_seed${seed}"
    local log_file="${LOG_DIR}/${dataset}_${prefix}_${TIMESTAMP}.log"
    local run_start
    local run_end
    local run_seconds

    echo "============================================================"
    echo "Starting ${mode}"
    echo "dataset=${dataset}, seed=${seed}, config=${config}"
    echo "all_seen=${all_seen}, old_overlap_beta=${old_overlap_beta}"
    echo "base conflict: beta=0.5, Top-r=0.1; spectral controller disabled"
    echo "W&B mode=online, log=${log_file}"
    echo "============================================================"

    run_start=$(date +%s)
    if python main.py \
        --config "$config" \
        --set "seed=[${seed}]" \
        --set "prefix=${prefix}" \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set "dual_mask_competence_all_seen=${all_seen}" \
        --set "dual_mask_conflict_old_overlap_adaptive=${old_overlap_beta}" \
        --set dual_mask_spectral_conflict_adaptive=false \
        --set dual_mask_task_relevance_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_metric_batches=4 \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=all_seen_vs_old_overlap_${TIMESTAMP}" \
        --set "wandb_tags=${mode},${dataset},seed${seed}" \
        2>&1 | tee "$log_file"; then
        run_end=$(date +%s)
        run_seconds=$((run_end - run_start))
        printf 'Finished %s/seed%s in %ds (%dh %dm %ds)\n' \
            "$mode" "$seed" "$run_seconds" \
            "$((run_seconds / 3600))" \
            "$(((run_seconds % 3600) / 60))" "$((run_seconds % 60))"
    else
        echo "FAILED: ${mode}/${dataset}/seed${seed}; continuing with the next run."
        FAILURES=$((FAILURES + 1))
    fi
}

for dataset in cub10 imgr10; do
    if [[ "$dataset" == "cub10" ]]; then
        config="exps/dlora/cub10.json"
    else
        config="exps/dlora/imgr10.json"
    fi

    for seed in "${SEEDS[@]}"; do
        # C_all replaces C_t for protection coverage, alpha, and private rank.
        run_experiment "$dataset" "$config" "$seed" all_seen_competence true false

        # C_new controls protection; R_old only strengthens conflict beta.
        run_experiment "$dataset" "$config" "$seed" old_overlap_beta false true
    done
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

echo "============================================================"
printf 'Night run finished with %s failed run(s) in %ds (%dh %dm %ds)\n' \
    "$FAILURES" "$TOTAL_SECONDS" "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" "$((TOTAL_SECONDS % 60))"
echo "Logs: ${LOG_DIR}"
echo "W&B group: all_seen_vs_old_overlap_${TIMESTAMP}"
echo "============================================================"

exit "$FAILURES"