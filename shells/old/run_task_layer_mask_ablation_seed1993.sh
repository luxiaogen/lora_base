#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/task_layer_mask_ablation"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
SEED=1993
INCREMENTAL_TASKS='[1,2,3,4,5,6,7,8,9]'

# Every variant uses Task 0 = protect_only. The first three rows compare
# the S-LoRA Task 1-9 gate mode; P-LoRA remains full for protect_only.
# The final three rows only unmask one layer group during Task 1-9 while
# all remaining incremental layers stay full.
VARIANTS=(
    incremental_full
    incremental_protect_only
    incremental_unmasked
    stageaware_shallow_off
    stageaware_middle_off
    stageaware_deep_off
)

mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local variant="$3"
    local protect_only_tasks
    local disabled_tasks
    local disabled_layers
    local disabled_layer_tasks
    local prefix
    local log_file
    local run_start
    local run_end

    case "$variant" in
        incremental_full)
            protect_only_tasks='[]'
            disabled_tasks='[]'
            disabled_layers='[]'
            disabled_layer_tasks='[]'
            ;;
        incremental_protect_only)
            protect_only_tasks="$INCREMENTAL_TASKS"
            disabled_tasks='[]'
            disabled_layers='[]'
            disabled_layer_tasks='[]'
            ;;
        incremental_unmasked)
            protect_only_tasks='[]'
            disabled_tasks="$INCREMENTAL_TASKS"
            disabled_layers='[]'
            disabled_layer_tasks='[]'
            ;;
        stageaware_shallow_off)
            protect_only_tasks='[]'
            disabled_tasks='[]'
            disabled_layers='[0,1,2,3]'
            disabled_layer_tasks="$INCREMENTAL_TASKS"
            ;;
        stageaware_middle_off)
            protect_only_tasks='[]'
            disabled_tasks='[]'
            disabled_layers='[4,5,6,7]'
            disabled_layer_tasks="$INCREMENTAL_TASKS"
            ;;
        stageaware_deep_off)
            protect_only_tasks='[]'
            disabled_tasks='[]'
            disabled_layers='[8,9,10,11]'
            disabled_layer_tasks="$INCREMENTAL_TASKS"
            ;;
        *)
            echo "Unknown variant: ${variant}" >&2
            return 2
            ;;
    esac

    prefix="${dataset}_task0protect_${variant}_seed${SEED}"
    log_file="${LOG_DIR}/${prefix}_${TIMESTAMP}.log"

    echo "============================================================"
    echo "Starting ${dataset}, seed=${SEED}, variant=${variant}"
    echo "Task 0 gate mode=protect_only"
    echo "S-LoRA protect_only_tasks=${protect_only_tasks}"
    echo "disabled_tasks=${disabled_tasks}"
    echo "disabled_layers=${disabled_layers}"
    echo "disabled_layer_tasks=${disabled_layer_tasks}"
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
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=protect_only \
        --set "dual_mask_protect_only_tasks=${protect_only_tasks}" \
        --set "dual_mask_disabled_tasks=${disabled_tasks}" \
        --set "dual_mask_disabled_layers=${disabled_layers}" \
        --set "dual_mask_disabled_layer_tasks=${disabled_layer_tasks}" \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=task_layer_mask_ablation_seed${SEED}_${TIMESTAMP}" \
        --set "wandb_tags=task_layer_mask_ablation,${dataset},${variant},seed${SEED}" \
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
        cub10) config="ideas/dual_mask_branch/configs/cub10.json" ;;
        imgr10) config="ideas/dual_mask_branch/configs/imgr10.json" ;;
        imga10) config="ideas/dual_mask_branch/configs/imga10.json" ;;
    esac

    for variant in "${VARIANTS[@]}"; do
        run_experiment "$dataset" "$config" "$variant"
    done
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

echo "============================================================"
echo "Finished 18 task/layer mask ablations."
printf 'Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "Logs: ${LOG_DIR}"
echo "============================================================"
