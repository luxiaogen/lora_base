#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
LOG_DIR=logs/shell_logs/imgr10_task0_holdout_tuning_3090
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_task0_base_e20_lr002_seed1993"
echo "Changed: Task0 train-only holdout baseline: init_epoch=20, init_lr=0.02"
echo "Log: $LOG_DIR/imgr10_task0_base_e20_lr002_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_task0_base_e20_lr002_seed1993 \
        --set max_tasks=1 \
        --set epochs=20 \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set ca_epochs=5 \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_granularity=layer \
        --set task0_validation_enabled=true \
        --set task0_validation_holdout_mod=5 \
        --set disable_fused_sdpa=true \
        --set wandb_group=imgr10_task0_holdout_tuning_3090 \
        --set init_epoch=20 \
        --set init_lr=0.02 \
        2>&1 | tee "$LOG_DIR/imgr10_task0_base_e20_lr002_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_task0_base_e20_lr002_seed1993"
else
    echo "FAIL imgr10_task0_base_e20_lr002_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_task0_e30_lr002_seed1993"
echo "Changed: Task0 train-only holdout epoch check: init_epoch=30, init_lr=0.02"
echo "Log: $LOG_DIR/imgr10_task0_e30_lr002_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_task0_e30_lr002_seed1993 \
        --set max_tasks=1 \
        --set epochs=20 \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set ca_epochs=5 \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_granularity=layer \
        --set task0_validation_enabled=true \
        --set task0_validation_holdout_mod=5 \
        --set disable_fused_sdpa=true \
        --set wandb_group=imgr10_task0_holdout_tuning_3090 \
        --set init_epoch=30 \
        --set init_lr=0.02 \
        2>&1 | tee "$LOG_DIR/imgr10_task0_e30_lr002_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_task0_e30_lr002_seed1993"
else
    echo "FAIL imgr10_task0_e30_lr002_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_task0_e20_lr001_seed1993"
echo "Changed: Task0 train-only holdout learning-rate check: init_epoch=20, init_lr=0.01"
echo "Log: $LOG_DIR/imgr10_task0_e20_lr001_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_task0_e20_lr001_seed1993 \
        --set max_tasks=1 \
        --set epochs=20 \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set ca_epochs=5 \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_granularity=layer \
        --set task0_validation_enabled=true \
        --set task0_validation_holdout_mod=5 \
        --set disable_fused_sdpa=true \
        --set wandb_group=imgr10_task0_holdout_tuning_3090 \
        --set init_epoch=20 \
        --set init_lr=0.01 \
        2>&1 | tee "$LOG_DIR/imgr10_task0_e20_lr001_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_task0_e20_lr001_seed1993"
else
    echo "FAIL imgr10_task0_e20_lr001_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Finished 3 runs for imgr10_task0_holdout_tuning_3090; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
