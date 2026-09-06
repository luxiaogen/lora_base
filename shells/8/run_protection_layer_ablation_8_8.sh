#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/protection_layer_ablation"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local seed="$3"
    local variant="$4"
    local coverage_mode="$5"
    local layerwise_mode="$6"
    local s_protect="$7"
    local prefix="${dataset}_${variant}_seed${seed}"
    local log_file="${LOG_DIR}/${prefix}_${TIMESTAMP}.log"

    echo "============================================================"
    echo "Starting ${prefix}"
    echo "coverage_mode=${coverage_mode}, layerwise_ratio=${layerwise_mode}, s_protect=${s_protect}"
    echo "Log: ${log_file}"
    echo "============================================================"

    python main.py \
        --config "$config" \
        --set "seed=[${seed}]" \
        --set "prefix=${prefix}" \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_general_ratio=0.5 \
        --set "dual_mask_coverage_mode=${coverage_mode}" \
        --set "dual_mask_layerwise_ratio_mode=${layerwise_mode}" \
        --set "dual_mask_s_protect_enabled=${s_protect}" \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=protect_only \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=protection_layer_ablation_${TIMESTAMP}" \
        --set "wandb_tags=protection_layer_ablation,${dataset},${variant},seed${seed}" \
        2>&1 | tee "$log_file"
}

for seed in 1993 1996; do
    run_experiment cub10 exps/dlora/cub10.json "$seed" fixed_ratio ratio none true
    run_experiment cub10 exps/dlora/cub10.json "$seed" ct_energy energy none true
    run_experiment cub10 exps/dlora/cub10.json "$seed" shallow_high ratio shallow_high true
    run_experiment cub10 exps/dlora/cub10.json "$seed" deep_high ratio deep_high true
    run_experiment cub10 exps/dlora/cub10.json "$seed" s_protect_off ratio none false

    run_experiment imgr10 exps/dlora/imgr10.json "$seed" fixed_ratio ratio none true
    run_experiment imgr10 exps/dlora/imgr10.json "$seed" ct_energy energy none true
    run_experiment imgr10 exps/dlora/imgr10.json "$seed" shallow_high ratio shallow_high true
    run_experiment imgr10 exps/dlora/imgr10.json "$seed" deep_high ratio deep_high true
    run_experiment imgr10 exps/dlora/imgr10.json "$seed" s_protect_off ratio none false

    run_experiment imga10 exps/dlora/imga10.json "$seed" fixed_ratio ratio none true
    run_experiment imga10 exps/dlora/imga10.json "$seed" ct_energy energy none true
    run_experiment imga10 exps/dlora/imga10.json "$seed" shallow_high ratio shallow_high true
    run_experiment imga10 exps/dlora/imga10.json "$seed" deep_high ratio deep_high true
    run_experiment imga10 exps/dlora/imga10.json "$seed" s_protect_off ratio none false
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))
echo "============================================================"
echo "All 30 protection/layer ablation runs finished"
printf 'Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "Logs: ${LOG_DIR}"
echo "============================================================"