#!/usr/bin/env bash
set -uo pipefail

# Run from the repository root, for example:
#   bash scripts/run_anchor_w0_3datasets_3seeds.sh
#
# This is the fixed-r50 w0 control:
#   - selective functional anchor: disabled (weight 0)
#   - Task-0 parameter W0 anchor: enabled (weight 10)

LOG_DIR=logs/shell_logs/anchor_w0_3datasets_3seeds
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

run_experiment() {
    local dataset="$1"
    local config="$2"
    local rank="$3"
    local seed="$4"
    local prefix="${dataset}_fixed_r50_anchor_task0_w10_selective_w0_seed${seed}"
    local logfile="$LOG_DIR/${prefix}_${TIMESTAMP}.log"

    echo "============================================================"
    echo "Starting ${prefix}"
    echo "Changed: fixed conflict Top-50; Task-0 W0 anchor w10; selective functional anchor w0"
    echo "Log: ${logfile}"
    echo "============================================================"

    if python main.py --config "$config" \
        --set "seed=[${seed}]" \
        --set "prefix=${prefix}" \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_ratio=0.5 \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_selective_anchor_weight=0.0 \
        --set dual_mask_selective_anchor_min_margin=0.05 \
        --set dual_mask_selective_anchor_tolerance=0.05 \
        --set dual_mask_selective_anchor_start_epoch=5 \
        --set dual_mask_selective_anchor_ramp_epochs=5 \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=anchor_w0_3datasets_3seeds \
        --set "rank=${rank}" \
        --set "wandb_tags=${dataset},suppress,fixed_r50,anchor_task0_w10,selective_w0,control,seed${seed}" \
        2>&1 | tee "$logfile"
    then
        echo "PASS ${prefix}"
    else
        echo "FAIL ${prefix}"
        FAILED=1
    fi
}

for seed in 1993 1996 1997; do
    run_experiment cub10 exps/dlora/cub10.json 32 "$seed"
    run_experiment imgr10 exps/dlora/imgr10.json 64 "$seed"
    run_experiment imga10 exps/dlora/imga10.json 32 "$seed"
done

echo "============================================================"
echo "Finished 9 runs for anchor_w0_3datasets_3seeds; FAILED=${FAILED}"
echo "Logs: ${LOG_DIR}"
echo "============================================================"
exit "$FAILED"