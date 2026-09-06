#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/ccontrol_rold_imga_cifar100"
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
SEEDS=(1993 1996 1997)
FAILED_RUNS=()

mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local seed="$3"
    local prefix="${dataset}_ccontrol_rold_seed${seed}"
    local log_file="${LOG_DIR}/${prefix}_${RUN_TIMESTAMP}.log"
    local run_start
    local run_seconds

    echo "============================================================"
    echo "Starting ${dataset}, seed=${seed}"
    echo "C_control = C_new * (1 - D_t); R_old only strengthens beta"
    echo "Top-r=10%, beta_0=0.5, SVD energy coverage=0.95"
    echo "Config: ${config}"
    echo "Log: ${log_file}"
    echo "============================================================"

    run_start=$(date +%s)

    if python main.py \
        --config "$config" \
        --set "seed=[${seed}]" \
        --set "prefix=${prefix}" \
        --set ca=true \
        --set ca_epochs=10 \
        --set ca_lrate=0.01 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=ccontrol_rold_imga_cifar100_${RUN_TIMESTAMP}" \
        --set "wandb_tags=ccontrol_rold,${dataset},seed${seed}" \
        2>&1 | tee "$log_file"
    then
        echo "Finished ${dataset}, seed=${seed}"
    else
        echo "FAILED ${dataset}, seed=${seed}; continuing with remaining runs"
        FAILED_RUNS+=("${dataset}:seed${seed}")
    fi

    run_seconds=$(($(date +%s) - run_start))
    printf 'Run time: %ds (%dh %dm %ds)\n' \
        "$run_seconds" \
        "$((run_seconds / 3600))" \
        "$(((run_seconds % 3600) / 60))" \
        "$((run_seconds % 60))"
}

for seed in "${SEEDS[@]}"; do
    run_experiment imga10 ideas/dual_mask_branch/configs/imga10.json "$seed"
done

for seed in "${SEEDS[@]}"; do
    run_experiment cifar100 ideas/dual_mask_branch/configs/cifar10.json "$seed"
done

TOTAL_SECONDS=$(($(date +%s) - START_TIME))

echo "============================================================"
echo "All six ImageNet-A/CIFAR100 experiments attempted."
printf 'Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "Logs: ${LOG_DIR}"

if ((${#FAILED_RUNS[@]} > 0)); then
    echo "Failed runs: ${FAILED_RUNS[*]}"
    exit 1
fi
