#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"
LOG_DIR=logs/shell_logs/best_suppress_pure_r50_seed1993
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting cub10_pure_r50_seed1993"
echo "Changed: Best suppress configuration; only conflict range changes from fixed Top-10% to per-layer minimum range covering 50% conflict energy"
echo "Log: $LOG_DIR/cub10_pure_r50_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/cub10.json \
        --set 'seed=[1993]' \
        --set prefix=cub10_pure_r50_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=false \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.0 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=best_suppress_pure_r50_seed1993 \
        --set wandb_tags=best_suppress,pure_r50,seed1993 \
        2>&1 | tee "$LOG_DIR/cub10_pure_r50_seed1993_${TIMESTAMP}.log"
then
    echo "PASS cub10_pure_r50_seed1993"
else
    echo "FAIL cub10_pure_r50_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_pure_r50_seed1993"
echo "Changed: Best suppress configuration; only conflict range changes from fixed Top-10% to per-layer minimum range covering 50% conflict energy"
echo "Log: $LOG_DIR/imgr10_pure_r50_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_pure_r50_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=false \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.0 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=best_suppress_pure_r50_seed1993 \
        --set wandb_tags=best_suppress,pure_r50,seed1993 \
        2>&1 | tee "$LOG_DIR/imgr10_pure_r50_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_pure_r50_seed1993"
else
    echo "FAIL imgr10_pure_r50_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_pure_r50_seed1993"
echo "Changed: Best suppress configuration; only conflict range changes from fixed Top-10% to per-layer minimum range covering 50% conflict energy"
echo "Log: $LOG_DIR/imga10_pure_r50_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imga10.json \
        --set 'seed=[1993]' \
        --set prefix=imga10_pure_r50_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=false \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.0 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=best_suppress_pure_r50_seed1993 \
        --set wandb_tags=best_suppress,pure_r50,seed1993 \
        2>&1 | tee "$LOG_DIR/imga10_pure_r50_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imga10_pure_r50_seed1993"
else
    echo "FAIL imga10_pure_r50_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Finished 3 runs for best-suppress-pure-r50-seed1993; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
