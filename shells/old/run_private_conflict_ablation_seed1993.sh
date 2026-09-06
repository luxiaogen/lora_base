#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/private_conflict_ablation"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
SEED=1993
PRIVATE_MODES=(global none plastic)
TASK0_MODES=(full protect_only unmasked)

mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local private_mode="$3"
    local task0_mode="$4"
    local prefix="${dataset}_private_${private_mode}_task0_${task0_mode}_seed${SEED}"
    local log_file="${LOG_DIR}/${prefix}_${TIMESTAMP}.log"
    local run_start
    local run_end

    echo "============================================================"
    echo "Starting ${dataset}, seed=${SEED}, private_conflict_mode=${private_mode}, task0_gate_mode=${task0_mode}"
    echo "global: current full-matrix Top-r"
    echo "none: P-LoRA uses plastic mask without conflict gate/reg"
    echo "plastic: Top-r is selected only inside the plastic region"
    echo "Log: ${log_file}"
    echo "============================================================"

    run_start=$(date +%s)
    if python main.py \
        --config "$config" \
        --set "seed=[${SEED}]" \
        --set "prefix=${prefix}" \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set ca_lrate=0.01 \
        --set label_smoothing=0.0 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_reg_enabled=true \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set "dual_mask_private_conflict_mode=${private_mode}" \
        --set "dual_mask_task0_gate_mode=${task0_mode}" \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=private_conflict_task0_ablation_${TIMESTAMP}" \
        --set "wandb_tags=private_conflict_ablation,task0_gate_ablation,${dataset},private_${private_mode},task0_${task0_mode},seed${SEED}" \
        2>&1 | tee "$log_file"; then
        run_end=$(date +%s)
        echo "Finished ${prefix} in $((run_end - run_start))s"
    else
        run_end=$(date +%s)
        echo "FAILED ${prefix} after $((run_end - run_start))s; continuing." >&2
    fi
}

for dataset in cub10 imgr10 imga10; do
    case "$dataset" in
        cub10) config="exps/dlora/cub10.json" ;;
        imgr10) config="exps/dlora/imgr10.json" ;;
        imga10) config="exps/dlora/imga10.json" ;;
    esac

    for private_mode in "${PRIVATE_MODES[@]}"; do
        for task0_mode in "${TASK0_MODES[@]}"; do
            run_experiment "$dataset" "$config" "$private_mode" "$task0_mode"
        done
    done
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

echo "============================================================"
echo "Finished 27 private-conflict/Task-0 gate experiments."
printf 'Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "Logs: ${LOG_DIR}"
echo "============================================================"
