#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
LOG_DIR=logs/shell_logs/imgr10_conflict_score_t10_seed1993_5090
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"
echo "Code revision: $(git rev-parse --short HEAD)"
git status --short

echo "============================================================"
echo "Starting imgr10_t10_conflict_seed1993"
echo "Changed: DualMask score: normalized W_pre importance times normalized LoRA update magnitude"
echo "Log: $LOG_DIR/imgr10_t10_conflict_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_t10_conflict_seed1993 \
        --set max_tasks=10 \
        --set init_epoch=20 \
        --set init_lr=0.02 \
        --set epochs=20 \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set ca_epochs=5 \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_conflict_old_overlap_adaptive=false \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_granularity=layer \
        --set dual_mask_conflict_budget_multiplier=1.0 \
        --set dual_mask_private_conflict_mode=global \
        --set task0_validation_enabled=false \
        --set disable_fused_sdpa=true \
        --set wandb_group=imgr10_conflict_score_t10_seed1993_5090 \
        --set dual_mask_conflict_score_mode=conflict \
        --set wandb_tags=imgr10,t10,seed1993,ca5,fixed10,score_conflict \
        2>&1 | tee "$LOG_DIR/imgr10_t10_conflict_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_t10_conflict_seed1993"
else
    echo "FAIL imgr10_t10_conflict_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_t10_magnitude_seed1993"
echo "Changed: Magnitude-only score at the same fixed 10 percent coordinate budget"
echo "Log: $LOG_DIR/imgr10_t10_magnitude_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_t10_magnitude_seed1993 \
        --set max_tasks=10 \
        --set init_epoch=20 \
        --set init_lr=0.02 \
        --set epochs=20 \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set ca_epochs=5 \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_conflict_old_overlap_adaptive=false \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_granularity=layer \
        --set dual_mask_conflict_budget_multiplier=1.0 \
        --set dual_mask_private_conflict_mode=global \
        --set task0_validation_enabled=false \
        --set disable_fused_sdpa=true \
        --set wandb_group=imgr10_conflict_score_t10_seed1993_5090 \
        --set dual_mask_conflict_score_mode=magnitude \
        --set wandb_tags=imgr10,t10,seed1993,ca5,fixed10,score_magnitude \
        2>&1 | tee "$LOG_DIR/imgr10_t10_magnitude_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_t10_magnitude_seed1993"
else
    echo "FAIL imgr10_t10_magnitude_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_t10_w_pre_seed1993"
echo "Changed: W_pre-only importance score at the same fixed 10 percent coordinate budget"
echo "Log: $LOG_DIR/imgr10_t10_w_pre_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_t10_w_pre_seed1993 \
        --set max_tasks=10 \
        --set init_epoch=20 \
        --set init_lr=0.02 \
        --set epochs=20 \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set ca_epochs=5 \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_conflict_old_overlap_adaptive=false \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_granularity=layer \
        --set dual_mask_conflict_budget_multiplier=1.0 \
        --set dual_mask_private_conflict_mode=global \
        --set task0_validation_enabled=false \
        --set disable_fused_sdpa=true \
        --set wandb_group=imgr10_conflict_score_t10_seed1993_5090 \
        --set dual_mask_conflict_score_mode=w_pre \
        --set wandb_tags=imgr10,t10,seed1993,ca5,fixed10,score_w_pre \
        2>&1 | tee "$LOG_DIR/imgr10_t10_w_pre_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_t10_w_pre_seed1993"
else
    echo "FAIL imgr10_t10_w_pre_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Finished 3 runs for imgr10_conflict_score_t10_seed1993_5090; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
