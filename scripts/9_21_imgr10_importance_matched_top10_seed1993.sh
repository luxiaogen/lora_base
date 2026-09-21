#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
if [[ ! -f main.py ]]; then
    echo "Run this script from a checkout containing main.py." >&2
    exit 1
fi

python3 -m unittest test.test_dualmask_off_lori_selector || exit 1

export PYTHONUNBUFFERED=1
LOG_DIR=logs/shell_logs/imgr10_importance_matched_top10_seed1993
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "Code revision: $(git rev-parse --short HEAD)"

echo "============================================================"
echo "Starting imgr10_svd_global_top10_seed1993"
echo "Changed: SVD importance with an exact global 10% protection budget"
echo "Log: $LOG_DIR/imgr10_svd_global_top10_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_svd_global_top10_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_enabled=true \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
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
        --set wandb_group=imgr10_importance_matched_top10_seed1993 \
        --set dual_mask_importance=svd_global_top10 \
        --set wandb_tags=imgr10,t10,importance_matched,svd,global_top10,seed1993 \
        2>&1 | tee "$LOG_DIR/imgr10_svd_global_top10_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_svd_global_top10_seed1993"
else
    echo "FAIL imgr10_svd_global_top10_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_magnitude_global_top10_seed1993"
echo "Changed: Magnitude importance with the same exact global 10% protection budget"
echo "Log: $LOG_DIR/imgr10_magnitude_global_top10_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_magnitude_global_top10_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_enabled=true \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
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
        --set wandb_group=imgr10_importance_matched_top10_seed1993 \
        --set dual_mask_importance=magnitude_global_top10 \
        --set wandb_tags=imgr10,t10,importance_matched,magnitude,global_top10,seed1993 \
        2>&1 | tee "$LOG_DIR/imgr10_magnitude_global_top10_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_magnitude_global_top10_seed1993"
else
    echo "FAIL imgr10_magnitude_global_top10_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Finished 2 runs for imgr10_importance_matched_top10_seed1993; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
