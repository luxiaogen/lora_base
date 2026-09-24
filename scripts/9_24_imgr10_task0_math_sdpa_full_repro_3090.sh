#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
echo "Code revision: $(git rev-parse --short HEAD)"
git status --short
LOG_DIR=logs/shell_logs/imgr10_task0_math_sdpa_full_repro_3090
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_task0_repeat1_seed1993"
echo "Changed: Math-only SDPA, full Task0 seed1993, repeat 1"
echo "Log: $LOG_DIR/imgr10_task0_repeat1_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_task0_math_sdpa_full_repeat1_seed1993_${TIMESTAMP}" \
        --set max_tasks=1 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set ca_epochs=5 \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_granularity=layer \
        --set task0_repro_diagnostic=true \
        --set disable_fused_sdpa=true \
        --set wandb_group=imgr10_task0_math_sdpa_full_repro_3090 \
        2>&1 | tee "$LOG_DIR/imgr10_task0_repeat1_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_task0_repeat1_seed1993"
else
    echo "FAIL imgr10_task0_repeat1_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_task0_repeat2_seed1993"
echo "Changed: Math-only SDPA, full Task0 seed1993, repeat 2"
echo "Log: $LOG_DIR/imgr10_task0_repeat2_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_task0_math_sdpa_full_repeat2_seed1993_${TIMESTAMP}" \
        --set max_tasks=1 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set ca_epochs=5 \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_granularity=layer \
        --set task0_repro_diagnostic=true \
        --set disable_fused_sdpa=true \
        --set wandb_group=imgr10_task0_math_sdpa_full_repro_3090 \
        2>&1 | tee "$LOG_DIR/imgr10_task0_repeat2_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_task0_repeat2_seed1993"
else
    echo "FAIL imgr10_task0_repeat2_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_task0_repeat3_seed1993"
echo "Changed: Math-only SDPA, full Task0 seed1993, repeat 3"
echo "Log: $LOG_DIR/imgr10_task0_repeat3_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_task0_math_sdpa_full_repeat3_seed1993_${TIMESTAMP}" \
        --set max_tasks=1 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set ca_epochs=5 \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_granularity=layer \
        --set task0_repro_diagnostic=true \
        --set disable_fused_sdpa=true \
        --set wandb_group=imgr10_task0_math_sdpa_full_repro_3090 \
        2>&1 | tee "$LOG_DIR/imgr10_task0_repeat3_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_task0_repeat3_seed1993"
else
    echo "FAIL imgr10_task0_repeat3_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Finished 3 runs for imgr10_task0_math_sdpa_full_repro_3090; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
