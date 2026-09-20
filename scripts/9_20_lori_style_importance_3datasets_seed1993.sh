#!/usr/bin/env bash
set -uo pipefail

if [[ ! -f main.py ]]; then
    echo "Run this script from the repository root." >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
python -m unittest test.test_dualmask_off_lori_selector || exit 1

LOG_DIR=logs/shell_logs/lori_style_importance_3datasets_seed1993
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

run_one() {
    local label=$1
    local config=$2
    local name="${label}_lori_style_importance_seed1993"
    local log_file="$LOG_DIR/${name}_${TIMESTAMP}.log"

    echo "============================================================"
    echo "Starting $name"
    echo "Full DualMask; only W_pre importance changes from SVD to global magnitude Top-10%."
    echo "Log: $log_file"
    echo "============================================================"
    if python main.py --config "$config" \
        --set 'seed=[1993]' \
        --set "prefix=$name" \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_enabled=true \
        --set dual_mask_importance=lori_global_magnitude \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=lori_style_importance_3datasets_seed1993 \
        --set "wandb_tags=$label,t10,dualmask,lori_style_global_top10,seed1993" \
        2>&1 | tee "$log_file"
    then
        echo "PASS $name"
    else
        echo "FAIL $name"
        FAILED=1
    fi
}

echo "Code revision: $(git rev-parse --short HEAD)"
run_one imgr10 exps/dlora/imgr10.json
run_one imga10 exps/dlora/imga10.json
run_one cub10 exps/dlora/cub10.json

echo "============================================================"
echo "Finished 3 LoRI-style importance runs; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit "$FAILED"
