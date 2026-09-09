#!/usr/bin/env bash
set -uo pipefail

# Run from the repository root after activating the training environment. No cd.
# Default: 6 paired runs, then the same 6 again (~9-10h on the observed 5090D).
# --first-pass: only the first 6 runs. --smoke: CPU unit/synthetic training checks only.
# CA diagnostics are test-only observations, never used for fitting or model selection.
case "${1:-}" in
    ""|--first-pass|--smoke) ;;
    *) echo "Usage: bash $0 [--first-pass|--smoke]" >&2; exit 2 ;;
esac
if [[ ! -f main.py ]]; then
    echo "Run this script from the repository root." >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
PYTHONWARNINGS=ignore python -m unittest test.test_private_rank test.test_ca_diagnostics || exit 1
if [[ "${1:-}" == "--smoke" ]]; then
    exit 0
fi
echo "Code revision: $(git rev-parse --short HEAD)"
LOG_DIR=logs/shell_logs/imgr10_prank64_ca_diagnostics
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_rep1_prank_adaptive_seed1993"
echo "Changed: Adaptive P rank; read-only before/after CA diagnostics"
echo "Log: $LOG_DIR/imgr10_rep1_prank_adaptive_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_rep1_prank_adaptive_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
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
        --set wandb_group=imgr10_prank64_ca_diagnostics \
        --set dual_mask_private_rank=0 \
        2>&1 | tee "$LOG_DIR/imgr10_rep1_prank_adaptive_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_rep1_prank_adaptive_seed1993"
else
    echo "FAIL imgr10_rep1_prank_adaptive_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_rep1_prank_fixed64_seed1993"
echo "Changed: Fixed P rank 64 from Task1; read-only before/after CA diagnostics"
echo "Log: $LOG_DIR/imgr10_rep1_prank_fixed64_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_rep1_prank_fixed64_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
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
        --set wandb_group=imgr10_prank64_ca_diagnostics \
        --set dual_mask_private_rank=64 \
        2>&1 | tee "$LOG_DIR/imgr10_rep1_prank_fixed64_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_rep1_prank_fixed64_seed1993"
else
    echo "FAIL imgr10_rep1_prank_fixed64_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_rep1_prank_adaptive_seed1996"
echo "Changed: Adaptive P rank; read-only before/after CA diagnostics"
echo "Log: $LOG_DIR/imgr10_rep1_prank_adaptive_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_rep1_prank_adaptive_seed1996 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
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
        --set wandb_group=imgr10_prank64_ca_diagnostics \
        --set dual_mask_private_rank=0 \
        2>&1 | tee "$LOG_DIR/imgr10_rep1_prank_adaptive_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_rep1_prank_adaptive_seed1996"
else
    echo "FAIL imgr10_rep1_prank_adaptive_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_rep1_prank_fixed64_seed1996"
echo "Changed: Fixed P rank 64 from Task1; read-only before/after CA diagnostics"
echo "Log: $LOG_DIR/imgr10_rep1_prank_fixed64_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_rep1_prank_fixed64_seed1996 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
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
        --set wandb_group=imgr10_prank64_ca_diagnostics \
        --set dual_mask_private_rank=64 \
        2>&1 | tee "$LOG_DIR/imgr10_rep1_prank_fixed64_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_rep1_prank_fixed64_seed1996"
else
    echo "FAIL imgr10_rep1_prank_fixed64_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_rep1_prank_adaptive_seed1997"
echo "Changed: Adaptive P rank; read-only before/after CA diagnostics"
echo "Log: $LOG_DIR/imgr10_rep1_prank_adaptive_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_rep1_prank_adaptive_seed1997 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
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
        --set wandb_group=imgr10_prank64_ca_diagnostics \
        --set dual_mask_private_rank=0 \
        2>&1 | tee "$LOG_DIR/imgr10_rep1_prank_adaptive_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_rep1_prank_adaptive_seed1997"
else
    echo "FAIL imgr10_rep1_prank_adaptive_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_rep1_prank_fixed64_seed1997"
echo "Changed: Fixed P rank 64 from Task1; read-only before/after CA diagnostics"
echo "Log: $LOG_DIR/imgr10_rep1_prank_fixed64_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_rep1_prank_fixed64_seed1997 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
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
        --set wandb_group=imgr10_prank64_ca_diagnostics \
        --set dual_mask_private_rank=64 \
        2>&1 | tee "$LOG_DIR/imgr10_rep1_prank_fixed64_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_rep1_prank_fixed64_seed1997"
else
    echo "FAIL imgr10_rep1_prank_fixed64_seed1997"
    FAILED=1
fi

echo "============================================================"
if [[ "${1:-}" == "--first-pass" ]]; then
    echo "Finished first 6 runs; FAILED=$FAILED; Logs: $LOG_DIR"
    exit "$FAILED"
fi
echo "Starting imgr10_rep2_prank_adaptive_seed1993"
echo "Changed: Adaptive P rank; read-only before/after CA diagnostics"
echo "Log: $LOG_DIR/imgr10_rep2_prank_adaptive_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_rep2_prank_adaptive_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
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
        --set wandb_group=imgr10_prank64_ca_diagnostics \
        --set dual_mask_private_rank=0 \
        2>&1 | tee "$LOG_DIR/imgr10_rep2_prank_adaptive_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_rep2_prank_adaptive_seed1993"
else
    echo "FAIL imgr10_rep2_prank_adaptive_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_rep2_prank_fixed64_seed1993"
echo "Changed: Fixed P rank 64 from Task1; read-only before/after CA diagnostics"
echo "Log: $LOG_DIR/imgr10_rep2_prank_fixed64_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_rep2_prank_fixed64_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
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
        --set wandb_group=imgr10_prank64_ca_diagnostics \
        --set dual_mask_private_rank=64 \
        2>&1 | tee "$LOG_DIR/imgr10_rep2_prank_fixed64_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_rep2_prank_fixed64_seed1993"
else
    echo "FAIL imgr10_rep2_prank_fixed64_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_rep2_prank_adaptive_seed1996"
echo "Changed: Adaptive P rank; read-only before/after CA diagnostics"
echo "Log: $LOG_DIR/imgr10_rep2_prank_adaptive_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_rep2_prank_adaptive_seed1996 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
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
        --set wandb_group=imgr10_prank64_ca_diagnostics \
        --set dual_mask_private_rank=0 \
        2>&1 | tee "$LOG_DIR/imgr10_rep2_prank_adaptive_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_rep2_prank_adaptive_seed1996"
else
    echo "FAIL imgr10_rep2_prank_adaptive_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_rep2_prank_fixed64_seed1996"
echo "Changed: Fixed P rank 64 from Task1; read-only before/after CA diagnostics"
echo "Log: $LOG_DIR/imgr10_rep2_prank_fixed64_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_rep2_prank_fixed64_seed1996 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
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
        --set wandb_group=imgr10_prank64_ca_diagnostics \
        --set dual_mask_private_rank=64 \
        2>&1 | tee "$LOG_DIR/imgr10_rep2_prank_fixed64_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_rep2_prank_fixed64_seed1996"
else
    echo "FAIL imgr10_rep2_prank_fixed64_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_rep2_prank_adaptive_seed1997"
echo "Changed: Adaptive P rank; read-only before/after CA diagnostics"
echo "Log: $LOG_DIR/imgr10_rep2_prank_adaptive_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_rep2_prank_adaptive_seed1997 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
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
        --set wandb_group=imgr10_prank64_ca_diagnostics \
        --set dual_mask_private_rank=0 \
        2>&1 | tee "$LOG_DIR/imgr10_rep2_prank_adaptive_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_rep2_prank_adaptive_seed1997"
else
    echo "FAIL imgr10_rep2_prank_adaptive_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_rep2_prank_fixed64_seed1997"
echo "Changed: Fixed P rank 64 from Task1; read-only before/after CA diagnostics"
echo "Log: $LOG_DIR/imgr10_rep2_prank_fixed64_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_rep2_prank_fixed64_seed1997 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
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
        --set wandb_group=imgr10_prank64_ca_diagnostics \
        --set dual_mask_private_rank=64 \
        2>&1 | tee "$LOG_DIR/imgr10_rep2_prank_fixed64_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_rep2_prank_fixed64_seed1997"
else
    echo "FAIL imgr10_rep2_prank_fixed64_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Finished 12 runs for imgr10_prank64_ca_diagnostics; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
