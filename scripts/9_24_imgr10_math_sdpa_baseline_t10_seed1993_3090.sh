#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
LOG_DIR=logs/shell_logs/imgr10_math_sdpa_baseline_t10_seed1993_3090
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_t10_baseline_seed1993"
echo "Changed: Deterministic ImageNet-R T10 baseline with full Task0 training data"
echo "Log: $LOG_DIR/imgr10_t10_baseline_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_t10_baseline_seed1993 \
        --set max_tasks=10 \
        --set init_epoch=20 \
        --set init_lr=0.02 \
        --set epochs=20 \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set ca_epochs=5 \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_granularity=layer \
        --set task0_validation_enabled=false \
        --set disable_fused_sdpa=true \
        --set wandb_group=imgr10_math_sdpa_baseline_t10_seed1993_3090 \
        2>&1 | tee "$LOG_DIR/imgr10_t10_baseline_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_t10_baseline_seed1993"
else
    echo "FAIL imgr10_t10_baseline_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Finished 1 runs for imgr10_math_sdpa_baseline_t10_seed1993_3090; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
