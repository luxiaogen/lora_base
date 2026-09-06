#!/usr/bin/env bash
set -uo pipefail

LOG_DIR=logs/shell_logs/selective_anchor_fixed_r50_overnight
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting cub10_fixed_r50_anchor_task0_w10_selective_w0_control_seed1993"
echo "Changed: Validation control: identical fixed-r50 configuration with selective functional anchor disabled"
echo "Log: $LOG_DIR/cub10_fixed_r50_anchor_task0_w10_selective_w0_control_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/cub10.json \
        --set 'seed=[1993]' \
        --set prefix=cub10_fixed_r50_anchor_task0_w10_selective_w0_control_seed1993 \
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
        --set wandb_group=selective_anchor_fixed_r50_3datasets_3seeds \
        --set rank=32 \
        --set wandb_tags=cub10,suppress,fixed_r50,anchor_task0_w10,selective_w0,validation,seed1993 \
        2>&1 | tee "$LOG_DIR/cub10_fixed_r50_anchor_task0_w10_selective_w0_control_seed1993_${TIMESTAMP}.log"
then
    echo "PASS cub10_fixed_r50_anchor_task0_w10_selective_w0_control_seed1993"
else
    echo "FAIL cub10_fixed_r50_anchor_task0_w10_selective_w0_control_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_fixed_r50_anchor_task0_w10_selective_w0_control_seed1993"
echo "Changed: Validation control: identical fixed-r50 configuration with selective functional anchor disabled"
echo "Log: $LOG_DIR/imga10_fixed_r50_anchor_task0_w10_selective_w0_control_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imga10.json \
        --set 'seed=[1993]' \
        --set prefix=imga10_fixed_r50_anchor_task0_w10_selective_w0_control_seed1993 \
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
        --set wandb_group=selective_anchor_fixed_r50_3datasets_3seeds \
        --set rank=32 \
        --set wandb_tags=imga10,suppress,fixed_r50,anchor_task0_w10,selective_w0,validation,seed1993 \
        2>&1 | tee "$LOG_DIR/imga10_fixed_r50_anchor_task0_w10_selective_w0_control_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imga10_fixed_r50_anchor_task0_w10_selective_w0_control_seed1993"
else
    echo "FAIL imga10_fixed_r50_anchor_task0_w10_selective_w0_control_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Finished 2 runs for selective_anchor_fixed_r50_validation_seed1993; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
