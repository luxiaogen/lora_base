#!/usr/bin/env bash
set -uo pipefail

# Run from the repository root; deliberately do not cd.
# --smoke: CPU unit/synthetic training checks only, no dataset or GPU training.
# Additional --set KEY=VALUE arguments are forwarded to every training run.
if [[ ! -f main.py ]]; then
    echo "Run from the repository root." >&2
    exit 2
fi
export PYTHONUNBUFFERED=1
PYTHONWARNINGS=ignore python -m unittest test.test_cross_task_training test.test_private_rank test.test_ca_diagnostics || exit 1
if [[ "${1:-}" == "--smoke" ]]; then
    exit 0
fi
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); print("GPU:", torch.cuda.get_device_name(0))' || exit 1
echo "Code revision: $(git rev-parse --short HEAD)"
git status --short
LOG_DIR=logs/shell_logs/imgr10_cross_task_3090
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_ct3090_A_local_seed1993"
echo "Changed: Baseline: task-local CosFace + unchanged CA5"
echo "Log: $LOG_DIR/imgr10_ct3090_A_local_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_ct3090_A_local_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
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
        --set wandb_group=imgr10_cross_task_3090 \
        --set wandb_tags=imgr10,cross_task,3090,ca5,adaptive_prank \
        --set classification_training_mode=task_local \
        "$@" \
        2>&1 | tee "$LOG_DIR/imgr10_ct3090_A_local_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_ct3090_A_local_seed1993"
else
    echo "FAIL imgr10_ct3090_A_local_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_ct3090_C_replay_seed1993"
echo "Changed: All-seen CosFace + equal-sized old pseudo-feature batch, unit loss weight + unchanged CA5"
echo "Log: $LOG_DIR/imgr10_ct3090_C_replay_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_ct3090_C_replay_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
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
        --set wandb_group=imgr10_cross_task_3090 \
        --set wandb_tags=imgr10,cross_task,3090,ca5,adaptive_prank \
        --set classification_training_mode=all_seen_replay \
        "$@" \
        2>&1 | tee "$LOG_DIR/imgr10_ct3090_C_replay_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_ct3090_C_replay_seed1993"
else
    echo "FAIL imgr10_ct3090_C_replay_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_ct3090_A_local_seed1996"
echo "Changed: Baseline: task-local CosFace + unchanged CA5"
echo "Log: $LOG_DIR/imgr10_ct3090_A_local_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_ct3090_A_local_seed1996 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
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
        --set wandb_group=imgr10_cross_task_3090 \
        --set wandb_tags=imgr10,cross_task,3090,ca5,adaptive_prank \
        --set classification_training_mode=task_local \
        "$@" \
        2>&1 | tee "$LOG_DIR/imgr10_ct3090_A_local_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_ct3090_A_local_seed1996"
else
    echo "FAIL imgr10_ct3090_A_local_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_ct3090_C_replay_seed1996"
echo "Changed: All-seen CosFace + equal-sized old pseudo-feature batch, unit loss weight + unchanged CA5"
echo "Log: $LOG_DIR/imgr10_ct3090_C_replay_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_ct3090_C_replay_seed1996 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
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
        --set wandb_group=imgr10_cross_task_3090 \
        --set wandb_tags=imgr10,cross_task,3090,ca5,adaptive_prank \
        --set classification_training_mode=all_seen_replay \
        "$@" \
        2>&1 | tee "$LOG_DIR/imgr10_ct3090_C_replay_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_ct3090_C_replay_seed1996"
else
    echo "FAIL imgr10_ct3090_C_replay_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_ct3090_A_local_seed1997"
echo "Changed: Baseline: task-local CosFace + unchanged CA5"
echo "Log: $LOG_DIR/imgr10_ct3090_A_local_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_ct3090_A_local_seed1997 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
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
        --set wandb_group=imgr10_cross_task_3090 \
        --set wandb_tags=imgr10,cross_task,3090,ca5,adaptive_prank \
        --set classification_training_mode=task_local \
        "$@" \
        2>&1 | tee "$LOG_DIR/imgr10_ct3090_A_local_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_ct3090_A_local_seed1997"
else
    echo "FAIL imgr10_ct3090_A_local_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_ct3090_C_replay_seed1997"
echo "Changed: All-seen CosFace + equal-sized old pseudo-feature batch, unit loss weight + unchanged CA5"
echo "Log: $LOG_DIR/imgr10_ct3090_C_replay_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_ct3090_C_replay_seed1997 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_ca_diagnostics=true \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
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
        --set wandb_group=imgr10_cross_task_3090 \
        --set wandb_tags=imgr10,cross_task,3090,ca5,adaptive_prank \
        --set classification_training_mode=all_seen_replay \
        "$@" \
        2>&1 | tee "$LOG_DIR/imgr10_ct3090_C_replay_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_ct3090_C_replay_seed1997"
else
    echo "FAIL imgr10_ct3090_C_replay_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Finished 6 runs for imgr10_cross_task_3090_main; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
