#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
LOG_DIR=logs/shell_logs/imgr10_task0_steering_screen_t3_3090
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_t3_baseline_seed1993"
echo "Changed: Reference: Task0 margin 0.10, anchor weight 10"
echo "Log: $LOG_DIR/imgr10_t3_baseline_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_t3_baseline_seed1993 \
        --set max_tasks=3 \
        --set init_epoch=20 \
        --set init_lr=0.02 \
        --set epochs=20 \
        --set ca_epochs=5 \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_granularity=layer \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set task0_margin=0.1 \
        --set task0_rs_weight=0.0 \
        --set task0_validation_enabled=true \
        --set task0_validation_holdout_mod=5 \
        --set disable_fused_sdpa=true \
        --set wandb_group=imgr10_task0_steering_screen_t3_3090 \
        2>&1 | tee "$LOG_DIR/imgr10_t3_baseline_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_t3_baseline_seed1993"
else
    echo "FAIL imgr10_t3_baseline_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_t3_task0_margin015_seed1993"
echo "Changed: Only Task0 CosFace margin 0.10 to 0.15"
echo "Log: $LOG_DIR/imgr10_t3_task0_margin015_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_t3_task0_margin015_seed1993 \
        --set max_tasks=3 \
        --set init_epoch=20 \
        --set init_lr=0.02 \
        --set epochs=20 \
        --set ca_epochs=5 \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_granularity=layer \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set task0_margin=0.15 \
        --set task0_rs_weight=0.0 \
        --set task0_validation_enabled=true \
        --set task0_validation_holdout_mod=5 \
        --set disable_fused_sdpa=true \
        --set wandb_group=imgr10_task0_steering_screen_t3_3090 \
        2>&1 | tee "$LOG_DIR/imgr10_t3_task0_margin015_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_t3_task0_margin015_seed1993"
else
    echo "FAIL imgr10_t3_task0_margin015_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_t3_anchor5_seed1993"
echo "Changed: Only Task0 anchor weight 10 to 5"
echo "Log: $LOG_DIR/imgr10_t3_anchor5_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_t3_anchor5_seed1993 \
        --set max_tasks=3 \
        --set init_epoch=20 \
        --set init_lr=0.02 \
        --set epochs=20 \
        --set ca_epochs=5 \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_granularity=layer \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_anchor_reg_weight=5.0 \
        --set task0_margin=0.1 \
        --set task0_rs_weight=0.0 \
        --set task0_validation_enabled=true \
        --set task0_validation_holdout_mod=5 \
        --set disable_fused_sdpa=true \
        --set wandb_group=imgr10_task0_steering_screen_t3_3090 \
        2>&1 | tee "$LOG_DIR/imgr10_t3_anchor5_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_t3_anchor5_seed1993"
else
    echo "FAIL imgr10_t3_anchor5_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_t3_task0_rs_seed1993"
echo "Changed: Only add RSIAT-inspired Task0 representation-steering loss (weight 0.5)"
echo "Log: $LOG_DIR/imgr10_t3_task0_rs_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_t3_task0_rs_seed1993 \
        --set max_tasks=3 \
        --set init_epoch=20 \
        --set init_lr=0.02 \
        --set epochs=20 \
        --set ca_epochs=5 \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_granularity=layer \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set task0_margin=0.1 \
        --set task0_rs_weight=0.5 \
        --set task0_validation_enabled=true \
        --set task0_validation_holdout_mod=5 \
        --set disable_fused_sdpa=true \
        --set wandb_group=imgr10_task0_steering_screen_t3_3090 \
        2>&1 | tee "$LOG_DIR/imgr10_t3_task0_rs_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_t3_task0_rs_seed1993"
else
    echo "FAIL imgr10_t3_task0_rs_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Finished 4 runs for imgr10_task0_steering_screen_t3_3090; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
