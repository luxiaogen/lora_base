#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
LOG_DIR=logs/shell_logs/imgr10_branch_gate_5090_b1997
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_a_baseline_seed1997"
echo "Changed: A seed1997 paired baseline on 5090D"
echo "Log: $LOG_DIR/imgr10_a_baseline_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_a_baseline_seed1997 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=imgr10_branch_gate_2machine \
        --set 'wandb_tags=["imgr10","branch_gate","5090","a_baseline","seed1997"]' \
        2>&1 | tee "$LOG_DIR/imgr10_a_baseline_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_a_baseline_seed1997"
else
    echo "FAIL imgr10_a_baseline_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_b_p_conflict_off_seed1997"
echo "Changed: B seed1997 on 5090D: disable only P conflict gate"
echo "Log: $LOG_DIR/imgr10_b_p_conflict_off_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_b_p_conflict_off_seed1997 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=none \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=imgr10_branch_gate_2machine \
        --set 'wandb_tags=["imgr10","branch_gate","5090","b_p_conflict_off","seed1997"]' \
        2>&1 | tee "$LOG_DIR/imgr10_b_p_conflict_off_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_b_p_conflict_off_seed1997"
else
    echo "FAIL imgr10_b_p_conflict_off_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Finished 2 runs for imgr10_branch_gate_5090_b1997; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
