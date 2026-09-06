#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/functional_merge_2x2"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME=$(date +%s)
SEED=1993
mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local variant="$3"
    local s_protect="$4"
    local calibration="$5"
    local prefix="${dataset}_${variant}_seed${SEED}"
    local log_file="${LOG_DIR}/${prefix}_${TIMESTAMP}.log"

    echo "============================================================"
    echo "Starting ${prefix}"
    echo "S-protect=${s_protect}, functional-beta-calibration=${calibration}"
    echo "Fixed: protect ratio=0.5, alpha=0.7, P-rank=full, conflict Top-r=0.1"
    echo "Log: ${log_file}"
    echo "============================================================"

    python main.py \
        --config "$config" \
        --set "seed=[${SEED}]" \
        --set "prefix=${prefix}" \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_coverage_mode=ratio \
        --set dual_mask_general_ratio=0.5 \
        --set dual_mask_layerwise_ratio_mode=none \
        --set "dual_mask_s_protect_enabled=${s_protect}" \
        --set dual_mask_competence_adaptive=false \
        --set dual_mask_plasticity_adaptive=false \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.0 \
        --set dual_mask_conflict_old_overlap_adaptive=false \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set "dual_mask_functional_merge_calibration=${calibration}" \
        --set dual_mask_functional_merge_tolerance=0.05 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=functional_merge_2x2_${TIMESTAMP}" \
        --set "wandb_tags=functional_merge_2x2,${dataset},${variant},seed${SEED}" \
        2>&1 | tee "$log_file"
}

# CUB: protection gate x binary beta calibration.
run_experiment cub10 exps/dlora/cub10.json protect_on_beta_fixed true false
run_experiment cub10 exps/dlora/cub10.json protect_off_beta_fixed false false
run_experiment cub10 exps/dlora/cub10.json protect_on_beta_select true true
run_experiment cub10 exps/dlora/cub10.json protect_off_beta_select false true

# ImageNet-R: same 2x2 design.
run_experiment imgr10 exps/dlora/imgr10.json protect_on_beta_fixed true false
run_experiment imgr10 exps/dlora/imgr10.json protect_off_beta_fixed false false
run_experiment imgr10 exps/dlora/imgr10.json protect_on_beta_select true true
run_experiment imgr10 exps/dlora/imgr10.json protect_off_beta_select false true

# ImageNet-A: same 2x2 design.
run_experiment imga10 exps/dlora/imga10.json protect_on_beta_fixed true false
run_experiment imga10 exps/dlora/imga10.json protect_off_beta_fixed false false
run_experiment imga10 exps/dlora/imga10.json protect_on_beta_select true true
run_experiment imga10 exps/dlora/imga10.json protect_off_beta_select false true

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))
echo "============================================================"
echo "All 12 functional-merge 2x2 runs finished"
printf 'Total shell time: %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "Logs: ${LOG_DIR}"
echo "============================================================"