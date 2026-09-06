#!/usr/bin/env bash
set -uo pipefail

LOG_DIR=logs/shell_logs/suppress_anchor_task0_energy50_validation_imgr_seed1993
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_energy50_anchor_off_seed1993"
echo "Changed: Anchor ablation relative to the main run: disable only Task-0 W0 anchor"
echo "Log: $LOG_DIR/imgr10_energy50_anchor_off_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_energy50_anchor_off_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=false \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=false \
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
        --set wandb_group=suppress_anchor_task0_energy50_validation_imgr_seed1993 \
        --set rank=64 \
        --set wandb_tags=imgr10,suppress,energy50,anchor_off,validation,seed1993 \
        2>&1 | tee "$LOG_DIR/imgr10_energy50_anchor_off_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_energy50_anchor_off_seed1993"
else
    echo "FAIL imgr10_energy50_anchor_off_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_fixed_top10_anchor_w10_seed1993"
echo "Changed: Conflict-range ablation relative to the main run: replace Energy50 with fixed coordinate Top-10 percent"
echo "Log: $LOG_DIR/imgr10_fixed_top10_anchor_w10_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_fixed_top10_anchor_w10_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_conflict_energy_ratio_floor=false \
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
        --set wandb_group=suppress_anchor_task0_energy50_validation_imgr_seed1993 \
        --set rank=64 \
        --set wandb_tags=imgr10,suppress,fixed_top10,anchor_task0,w10,validation,seed1993 \
        2>&1 | tee "$LOG_DIR/imgr10_fixed_top10_anchor_w10_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_fixed_top10_anchor_w10_seed1993"
else
    echo "FAIL imgr10_fixed_top10_anchor_w10_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Finished 2 runs for suppress_anchor_task0_energy50_validation_imgr_seed1993; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED