#!/usr/bin/env bash
set -uo pipefail

# Run from the repository root in your activated training environment; no cd.
# 6 runs: seed 1993, 1996, 1997; adaptive P rank then fixed 40 for each seed.
# Fixed 40 is the median of Task1-9 baseline P ranks (38-41), not an accuracy sweep.
# Keep Task0 initialization unchanged. CA5 / reg=0.01 / conflict_reg=false
# are explicit overrides to match the 9_9.log baseline, not the JSON defaults.
if [[ ! -f main.py ]]; then
    echo "Run this script from the repository root." >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
PYTHONWARNINGS=ignore python -m unittest test.test_private_rank || exit 1
echo "Code revision: $(git rev-parse --short HEAD)"
LOG_DIR=logs/shell_logs/imgr10_private_rank
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_prank_adaptive_seed1993"
echo "Changed: Baseline: adaptive P rank; unchanged Task0 initialization"
echo "Log: $LOG_DIR/imgr10_prank_adaptive_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_prank_adaptive_seed1993 \
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
        --set wandb_group=imgr10_private_rank_ca5 \
        --set dual_mask_private_rank=0 \
        2>&1 | tee "$LOG_DIR/imgr10_prank_adaptive_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_prank_adaptive_seed1993"
else
    echo "FAIL imgr10_prank_adaptive_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_prank_fixed40_seed1993"
echo "Changed: Only fix P rank to 40 from Task1; same Task0 and protection controller"
echo "Log: $LOG_DIR/imgr10_prank_fixed40_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_prank_fixed40_seed1993 \
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
        --set wandb_group=imgr10_private_rank_ca5 \
        --set dual_mask_private_rank=40 \
        2>&1 | tee "$LOG_DIR/imgr10_prank_fixed40_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_prank_fixed40_seed1993"
else
    echo "FAIL imgr10_prank_fixed40_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_prank_adaptive_seed1996"
echo "Changed: Baseline: adaptive P rank; unchanged Task0 initialization"
echo "Log: $LOG_DIR/imgr10_prank_adaptive_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_prank_adaptive_seed1996 \
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
        --set wandb_group=imgr10_private_rank_ca5 \
        --set dual_mask_private_rank=0 \
        2>&1 | tee "$LOG_DIR/imgr10_prank_adaptive_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_prank_adaptive_seed1996"
else
    echo "FAIL imgr10_prank_adaptive_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_prank_fixed40_seed1996"
echo "Changed: Only fix P rank to 40 from Task1; same Task0 and protection controller"
echo "Log: $LOG_DIR/imgr10_prank_fixed40_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_prank_fixed40_seed1996 \
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
        --set wandb_group=imgr10_private_rank_ca5 \
        --set dual_mask_private_rank=40 \
        2>&1 | tee "$LOG_DIR/imgr10_prank_fixed40_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_prank_fixed40_seed1996"
else
    echo "FAIL imgr10_prank_fixed40_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_prank_adaptive_seed1997"
echo "Changed: Baseline: adaptive P rank; unchanged Task0 initialization"
echo "Log: $LOG_DIR/imgr10_prank_adaptive_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_prank_adaptive_seed1997 \
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
        --set wandb_group=imgr10_private_rank_ca5 \
        --set dual_mask_private_rank=0 \
        2>&1 | tee "$LOG_DIR/imgr10_prank_adaptive_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_prank_adaptive_seed1997"
else
    echo "FAIL imgr10_prank_adaptive_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_prank_fixed40_seed1997"
echo "Changed: Only fix P rank to 40 from Task1; same Task0 and protection controller"
echo "Log: $LOG_DIR/imgr10_prank_fixed40_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_prank_fixed40_seed1997 \
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
        --set wandb_group=imgr10_private_rank_ca5 \
        --set dual_mask_private_rank=40 \
        2>&1 | tee "$LOG_DIR/imgr10_prank_fixed40_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_prank_fixed40_seed1997"
else
    echo "FAIL imgr10_prank_fixed40_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Finished 6 runs for imgr10_private_rank; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
