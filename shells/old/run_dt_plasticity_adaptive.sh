#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/dt_plasticity_adaptive"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
SEEDS=(1993 1996 1997)
FAILURES=0

mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local seed="$3"
    local mode="$4"
    local adaptive="$5"
    local prefix="${dataset}_${mode}_seed${seed}"
    local log_file="${LOG_DIR}/${prefix}_${TIMESTAMP}.log"
    local run_start
    local run_end
    local run_seconds

    echo "============================================================"
    echo "Starting ${mode}"
    echo "dataset=${dataset}, seed=${seed}, config=${config}"
    echo "D_t controller=${adaptive}"
    echo "C_control=C_new*(1-D_t); beta=0.5; Top-r=0.1"
    echo "SVD max rank=768; target energy=0.95"
    echo "W&B mode=online, log=${log_file}"
    echo "============================================================"

    run_start=$(date +%s)
    if python main.py \
        --config "$config" \
        --set "seed=[${seed}]" \
        --set "prefix=${prefix}" \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_plasticity_diagnostics=true \
        --set "dual_mask_plasticity_adaptive=${adaptive}" \
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
        --set "wandb_group=dt_plasticity_adaptive_${TIMESTAMP}" \
        --set "wandb_tags=dt_plasticity,${mode},${dataset},seed${seed}" \
        2>&1 | tee "$log_file"; then
        run_end=$(date +%s)
        run_seconds=$((run_end - run_start))
        printf 'Finished %s/seed%s in %ds (%dh %dm %ds)\n' \
            "$mode" "$seed" "$run_seconds" \
            "$((run_seconds / 3600))" \
            "$(((run_seconds % 3600) / 60))" \
            "$((run_seconds % 60))"
    else
        echo "FAILED: ${dataset}/${mode}/seed${seed}; continuing."
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
        run_experiment "$dataset" "$config" "$seed" dt_off false
        run_experiment "$dataset" "$config" "$seed" dt_on true
    done
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

echo "============================================================"
printf 'All 12 runs finished with %s failure(s) in %ds (%dh %dm %ds)\n' \
    "$FAILURES" "$TOTAL_SECONDS" "$((TOTAL_SECONDS / 3600))" \