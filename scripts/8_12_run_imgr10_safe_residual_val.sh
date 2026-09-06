#!/usr/bin/env bash
set -uo pipefail

#cd "$(dirname "$0")/.."

LOG_DIR=logs/shell_logs/imgr10_safe_residual_validation_overnight
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_suppress_safe_w0_seed1993"
echo "Changed: safe residual weight 0 control"
echo "Log: $LOG_DIR/imgr10_suppress_safe_w0_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_suppress_safe_w0_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.0 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=false \
        --set dual_mask_safe_residual_enabled=true \
        --set dual_mask_safe_residual_weight=0.0 \
        --set dual_mask_safe_residual_vectors=64 \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=imgr10_safe_residual_validation_overnight \
        --set wandb_tags=imgr10,suppress,safe_residual,w0,seed1993,overnight \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_safe_w0_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_safe_w0_seed1993"
else
    echo "FAIL imgr10_suppress_safe_w0_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_suppress_safe_w0_seed1996"
echo "Changed: seed 1996, safe residual weight 0 control"
echo "Log: $LOG_DIR/imgr10_suppress_safe_w0_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_suppress_safe_w0_seed1996 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.0 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=false \
        --set dual_mask_safe_residual_enabled=true \
        --set dual_mask_safe_residual_weight=0.0 \
        --set dual_mask_safe_residual_vectors=64 \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=imgr10_safe_residual_validation_overnight \
        --set wandb_tags=imgr10,suppress,safe_residual,w0,seed1996,overnight \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_safe_w0_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_safe_w0_seed1996"
else
    echo "FAIL imgr10_suppress_safe_w0_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_suppress_safe_w1000_seed1996"
echo "Changed: seed 1996, safe residual weight 1000 treatment"
echo "Log: $LOG_DIR/imgr10_suppress_safe_w1000_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_suppress_safe_w1000_seed1996 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.0 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=false \
        --set dual_mask_safe_residual_enabled=true \
        --set dual_mask_safe_residual_weight=1000.0 \
        --set dual_mask_safe_residual_vectors=64 \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=imgr10_safe_residual_validation_overnight \
        --set wandb_tags=imgr10,suppress,safe_residual,w1000,seed1996,overnight \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_safe_w1000_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_safe_w1000_seed1996"
else
    echo "FAIL imgr10_suppress_safe_w1000_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_suppress_safe_w0_seed1997"
echo "Changed: seed 1997, safe residual weight 0 control"
echo "Log: $LOG_DIR/imgr10_suppress_safe_w0_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_suppress_safe_w0_seed1997 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.0 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=false \
        --set dual_mask_safe_residual_enabled=true \
        --set dual_mask_safe_residual_weight=0.0 \
        --set dual_mask_safe_residual_vectors=64 \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=imgr10_safe_residual_validation_overnight \
        --set wandb_tags=imgr10,suppress,safe_residual,w0,seed1997,overnight \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_safe_w0_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_safe_w0_seed1997"
else
    echo "FAIL imgr10_suppress_safe_w0_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_suppress_safe_w1000_seed1997"
echo "Changed: seed 1997, safe residual weight 1000 treatment"
echo "Log: $LOG_DIR/imgr10_suppress_safe_w1000_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_suppress_safe_w1000_seed1997 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.0 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=false \
        --set dual_mask_safe_residual_enabled=true \
        --set dual_mask_safe_residual_weight=1000.0 \
        --set dual_mask_safe_residual_vectors=64 \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=imgr10_safe_residual_validation_overnight \
        --set wandb_tags=imgr10,suppress,safe_residual,w1000,seed1997,overnight \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_safe_w1000_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_safe_w1000_seed1997"
else
    echo "FAIL imgr10_suppress_safe_w1000_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_suppress_relocate_safe_w0_seed1993"
echo "Changed: suppress_relocate, safe residual weight 0 control"
echo "Log: $LOG_DIR/imgr10_suppress_relocate_safe_w0_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_suppress_relocate_safe_w0_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.0 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress_relocate \
        --set dual_mask_relocation_steps=20 \
        --set dual_mask_relocation_lr=0.1 \
        --set dual_mask_relocation_vectors=64 \
        --set dual_mask_anchor_reg_enabled=false \
        --set dual_mask_safe_residual_enabled=true \
        --set dual_mask_safe_residual_weight=0.0 \
        --set dual_mask_safe_residual_vectors=64 \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=imgr10_safe_residual_validation_overnight \
        --set wandb_tags=imgr10,suppress_relocate,safe_residual,w0,seed1993,steps20,overnight \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_relocate_safe_w0_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_relocate_safe_w0_seed1993"
else
    echo "FAIL imgr10_suppress_relocate_safe_w0_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_suppress_relocate_safe_w1000_seed1993"
echo "Changed: suppress_relocate, safe residual weight 1000 treatment"
echo "Log: $LOG_DIR/imgr10_suppress_relocate_safe_w1000_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_suppress_relocate_safe_w1000_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.0 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress_relocate \
        --set dual_mask_relocation_steps=20 \
        --set dual_mask_relocation_lr=0.1 \
        --set dual_mask_relocation_vectors=64 \
        --set dual_mask_anchor_reg_enabled=false \
        --set dual_mask_safe_residual_enabled=true \
        --set dual_mask_safe_residual_weight=1000.0 \
        --set dual_mask_safe_residual_vectors=64 \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=imgr10_safe_residual_validation_overnight \
        --set wandb_tags=imgr10,suppress_relocate,safe_residual,w1000,seed1993,steps20,overnight \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_relocate_safe_w1000_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_relocate_safe_w1000_seed1993"
else
    echo "FAIL imgr10_suppress_relocate_safe_w1000_seed1993"
    FAILED=1
fi

echo "============================================================"
if [[ "$FAILED" -eq 0 ]]; then
    echo "All 7 ImageNet-R validation runs passed."
else
    echo "One or more ImageNet-R validation runs failed; inspect $LOG_DIR."
fi
echo "Logs: $LOG_DIR"
echo "============================================================"

exit "$FAILED"l
