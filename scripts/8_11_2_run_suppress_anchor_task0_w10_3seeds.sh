#!/usr/bin/env bash
set -uo pipefail

#cd "$(dirname "$0")/.."
LOG_DIR=logs/shell_logs/suppress_anchor_task0_w10_3datasets_3seeds
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting cub10_suppress_anchor_task0_w10_seed1993"
echo "Changed: Successful suppress configuration with pretrained-anchor regularization only on Task 0"
echo "Log: $LOG_DIR/cub10_suppress_anchor_task0_w10_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/cub10.json \
        --set 'seed=[1993]' \
        --set prefix=cub10_suppress_anchor_task0_w10_seed1993 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=suppress_anchor_task0_w10_3datasets_3seeds \
        --set wandb_tags=suppress,anchor_task0,w10,night,multiseed \
        2>&1 | tee "$LOG_DIR/cub10_suppress_anchor_task0_w10_seed1993_${TIMESTAMP}.log"
then
    echo "PASS cub10_suppress_anchor_task0_w10_seed1993"
else
    echo "FAIL cub10_suppress_anchor_task0_w10_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting cub10_suppress_anchor_task0_w10_seed1996"
echo "Changed: Successful suppress configuration with pretrained-anchor regularization only on Task 0"
echo "Log: $LOG_DIR/cub10_suppress_anchor_task0_w10_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/cub10.json \
        --set 'seed=[1996]' \
        --set prefix=cub10_suppress_anchor_task0_w10_seed1996 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=suppress_anchor_task0_w10_3datasets_3seeds \
        --set wandb_tags=suppress,anchor_task0,w10,night,multiseed \
        2>&1 | tee "$LOG_DIR/cub10_suppress_anchor_task0_w10_seed1996_${TIMESTAMP}.log"
then
    echo "PASS cub10_suppress_anchor_task0_w10_seed1996"
else
    echo "FAIL cub10_suppress_anchor_task0_w10_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting cub10_suppress_anchor_task0_w10_seed1997"
echo "Changed: Successful suppress configuration with pretrained-anchor regularization only on Task 0"
echo "Log: $LOG_DIR/cub10_suppress_anchor_task0_w10_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/cub10.json \
        --set 'seed=[1997]' \
        --set prefix=cub10_suppress_anchor_task0_w10_seed1997 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=suppress_anchor_task0_w10_3datasets_3seeds \
        --set wandb_tags=suppress,anchor_task0,w10,night,multiseed \
        2>&1 | tee "$LOG_DIR/cub10_suppress_anchor_task0_w10_seed1997_${TIMESTAMP}.log"
then
    echo "PASS cub10_suppress_anchor_task0_w10_seed1997"
else
    echo "FAIL cub10_suppress_anchor_task0_w10_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_suppress_anchor_task0_w10_seed1993"
echo "Changed: Successful suppress configuration with pretrained-anchor regularization only on Task 0"
echo "Log: $LOG_DIR/imgr10_suppress_anchor_task0_w10_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_suppress_anchor_task0_w10_seed1993 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=suppress_anchor_task0_w10_3datasets_3seeds \
        --set wandb_tags=suppress,anchor_task0,w10,night,multiseed \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_anchor_task0_w10_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_anchor_task0_w10_seed1993"
else
    echo "FAIL imgr10_suppress_anchor_task0_w10_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_suppress_anchor_task0_w10_seed1996"
echo "Changed: Successful suppress configuration with pretrained-anchor regularization only on Task 0"
echo "Log: $LOG_DIR/imgr10_suppress_anchor_task0_w10_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_suppress_anchor_task0_w10_seed1996 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=suppress_anchor_task0_w10_3datasets_3seeds \
        --set wandb_tags=suppress,anchor_task0,w10,night,multiseed \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_anchor_task0_w10_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_anchor_task0_w10_seed1996"
else
    echo "FAIL imgr10_suppress_anchor_task0_w10_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_suppress_anchor_task0_w10_seed1997"
echo "Changed: Successful suppress configuration with pretrained-anchor regularization only on Task 0"
echo "Log: $LOG_DIR/imgr10_suppress_anchor_task0_w10_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_suppress_anchor_task0_w10_seed1997 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=suppress_anchor_task0_w10_3datasets_3seeds \
        --set wandb_tags=suppress,anchor_task0,w10,night,multiseed \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_anchor_task0_w10_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_anchor_task0_w10_seed1997"
else
    echo "FAIL imgr10_suppress_anchor_task0_w10_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_suppress_anchor_task0_w10_seed1993"
echo "Changed: Successful suppress configuration with pretrained-anchor regularization only on Task 0"
echo "Log: $LOG_DIR/imga10_suppress_anchor_task0_w10_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imga10.json \
        --set 'seed=[1993]' \
        --set prefix=imga10_suppress_anchor_task0_w10_seed1993 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=suppress_anchor_task0_w10_3datasets_3seeds \
        --set wandb_tags=suppress,anchor_task0,w10,night,multiseed \
        2>&1 | tee "$LOG_DIR/imga10_suppress_anchor_task0_w10_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imga10_suppress_anchor_task0_w10_seed1993"
else
    echo "FAIL imga10_suppress_anchor_task0_w10_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_suppress_anchor_task0_w10_seed1996"
echo "Changed: Successful suppress configuration with pretrained-anchor regularization only on Task 0"
echo "Log: $LOG_DIR/imga10_suppress_anchor_task0_w10_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imga10.json \
        --set 'seed=[1996]' \
        --set prefix=imga10_suppress_anchor_task0_w10_seed1996 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=suppress_anchor_task0_w10_3datasets_3seeds \
        --set wandb_tags=suppress,anchor_task0,w10,night,multiseed \
        2>&1 | tee "$LOG_DIR/imga10_suppress_anchor_task0_w10_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imga10_suppress_anchor_task0_w10_seed1996"
else
    echo "FAIL imga10_suppress_anchor_task0_w10_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_suppress_anchor_task0_w10_seed1997"
echo "Changed: Successful suppress configuration with pretrained-anchor regularization only on Task 0"
echo "Log: $LOG_DIR/imga10_suppress_anchor_task0_w10_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imga10.json \
        --set 'seed=[1997]' \
        --set prefix=imga10_suppress_anchor_task0_w10_seed1997 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=suppress_anchor_task0_w10_3datasets_3seeds \
        --set wandb_tags=suppress,anchor_task0,w10,night,multiseed \
        2>&1 | tee "$LOG_DIR/imga10_suppress_anchor_task0_w10_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imga10_suppress_anchor_task0_w10_seed1997"
else
    echo "FAIL imga10_suppress_anchor_task0_w10_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Finished 9 runs for suppress_anchor_task0_w10_3datasets_3seeds; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
