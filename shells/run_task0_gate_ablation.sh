#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/task0_gate_ablation"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SEEDS=(1993 1996 1997)
MODES=(protect_only unmasked)
mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local mode="$3"
    local seed="$4"
    local prefix="${dataset}_task0_${mode}_seed${seed}"
    local log_file="${LOG_DIR}/${prefix}_${TIMESTAMP}.log"

    echo "============================================================"
    echo "Starting ${dataset}, seed=${seed}, task0_gate_mode=${mode}"
    echo "Task 0: ${mode}; Task >= 1: full DualMask"
    echo "Log: ${log_file}"
    echo "============================================================"

    if python main.py \
        --config "$config" \
        --set "seed=[${seed}]" \
        --set "prefix=${prefix}" \
        --set init_epoch=20 \
        --set epochs=20 \
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
        --set "dual_mask_task0_gate_mode=${mode}" \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=task0_gate_ablation_${TIMESTAMP}" \
        --set "wandb_tags=task0_gate_ablation,${dataset},${mode},seed${seed}" \
        2>&1 | tee "$log_file"; then
        echo "Finished ${prefix}"
    else
        echo "FAILED ${prefix}; continuing with remaining experiments." >&2
    fi
}

# --- 只运行 cifar10 ---
dataset="cifar10"
config="ideas/dual_mask_branch/configs/cifar10.json"

for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "$dataset" "$config" "$mode" "$seed"
    done
done

##for dataset in cub10 imgr10 imga10 cifar10; do
#for dataset in cifar10; do
#    case "$dataset" in
##        cub10) config="ideas/dual_mask_branch/configs/cub10.json" ;;
##        imgr10) config="ideas/dual_mask_branch/configs/imgr10.json" ;;
##        imga10) config="ideas/dual_mask_branch/configs/imga10.json" ;;
#        cifar10) config="ideas/dual_mask_branch/configs/cifar10.json" ;;
#    esac
#
#    for mode in "${MODES[@]}"; do
#        for seed in "${SEEDS[@]}"; do
#            run_experiment "$dataset" "$config" "$mode" "$seed"
#        done
#    done
#done

# 计算总共运行的实验次数 (2 modes * 3 seeds = 6 次)
total_runs=$((${#MODES[@]} * ${#SEEDS[@]}))
#echo "============================================================"
#echo "Finished 18 Task-0 gate ablations. Logs: ${LOG_DIR}"
#echo "============================================================"
echo "============================================================"
echo "Finished ${total_runs} Task-0 gate ablations. Logs: ${LOG_DIR}"
echo "============================================================"