#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
LOG_DIR=logs/shell_logs/imgr10_p_strength
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

python -m unittest \
    test.test_branch_strength_sweep \
    test.test_dual_mask_core.LoRALifecycleTests.test_private_conflict_strength_changes_only_private_branch \
    test.test_dual_mask_core.LoRALifecycleTests.test_private_conflict_strength_defaults_to_shared_strength \
    || exit 1

echo "Running a two-task beta_P route smoke; this is not a performance result."
bash scripts/9_22_imgr10_p_strength_smoke_5090.sh || exit 1

echo "============================================================"
echo "Starting imgr10_p_beta025_seed1993"
echo "Changed: Keep beta_S=0.50; set only beta_P=0.25"
echo "Log: $LOG_DIR/imgr10_p_beta025_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_p_beta025_seed1993 \
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
        --set dual_mask_s_protect_enabled=true \
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
        --set wandb_group=imgr10_p_strength_5090 \
        --set dual_mask_private_conflict_strength=0.25 \
        --set 'wandb_tags=["imgr10","p_strength","beta025"]' \
        2>&1 | tee "$LOG_DIR/imgr10_p_beta025_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_p_beta025_seed1993"
else
    echo "FAIL imgr10_p_beta025_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_p_beta050_seed1993"
echo "Changed: Matched baseline: beta_S=0.50 and beta_P=0.50"
echo "Log: $LOG_DIR/imgr10_p_beta050_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_p_beta050_seed1993 \
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
        --set dual_mask_s_protect_enabled=true \
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
        --set wandb_group=imgr10_p_strength_5090 \
        --set dual_mask_private_conflict_strength=0.5 \
        --set 'wandb_tags=["imgr10","p_strength","beta050"]' \
        2>&1 | tee "$LOG_DIR/imgr10_p_beta050_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_p_beta050_seed1993"
else
    echo "FAIL imgr10_p_beta050_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_p_beta075_seed1993"
echo "Changed: Keep beta_S=0.50; set only beta_P=0.75"
echo "Log: $LOG_DIR/imgr10_p_beta075_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_p_beta075_seed1993 \
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
        --set dual_mask_s_protect_enabled=true \
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
        --set wandb_group=imgr10_p_strength_5090 \
        --set dual_mask_private_conflict_strength=0.75 \
        --set 'wandb_tags=["imgr10","p_strength","beta075"]' \
        2>&1 | tee "$LOG_DIR/imgr10_p_beta075_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_p_beta075_seed1993"
else
    echo "FAIL imgr10_p_beta075_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_p_beta025_seed1996"
echo "Changed: Keep beta_S=0.50; set only beta_P=0.25"
echo "Log: $LOG_DIR/imgr10_p_beta025_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_p_beta025_seed1996 \
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
        --set dual_mask_s_protect_enabled=true \
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
        --set wandb_group=imgr10_p_strength_5090 \
        --set dual_mask_private_conflict_strength=0.25 \
        --set 'wandb_tags=["imgr10","p_strength","beta025"]' \
        2>&1 | tee "$LOG_DIR/imgr10_p_beta025_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_p_beta025_seed1996"
else
    echo "FAIL imgr10_p_beta025_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_p_beta050_seed1996"
echo "Changed: Matched baseline: beta_S=0.50 and beta_P=0.50"
echo "Log: $LOG_DIR/imgr10_p_beta050_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_p_beta050_seed1996 \
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
        --set dual_mask_s_protect_enabled=true \
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
        --set wandb_group=imgr10_p_strength_5090 \
        --set dual_mask_private_conflict_strength=0.5 \
        --set 'wandb_tags=["imgr10","p_strength","beta050"]' \
        2>&1 | tee "$LOG_DIR/imgr10_p_beta050_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_p_beta050_seed1996"
else
    echo "FAIL imgr10_p_beta050_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_p_beta075_seed1996"
echo "Changed: Keep beta_S=0.50; set only beta_P=0.75"
echo "Log: $LOG_DIR/imgr10_p_beta075_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_p_beta075_seed1996 \
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
        --set dual_mask_s_protect_enabled=true \
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
        --set wandb_group=imgr10_p_strength_5090 \
        --set dual_mask_private_conflict_strength=0.75 \
        --set 'wandb_tags=["imgr10","p_strength","beta075"]' \
        2>&1 | tee "$LOG_DIR/imgr10_p_beta075_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_p_beta075_seed1996"
else
    echo "FAIL imgr10_p_beta075_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_p_beta025_seed1997"
echo "Changed: Keep beta_S=0.50; set only beta_P=0.25"
echo "Log: $LOG_DIR/imgr10_p_beta025_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_p_beta025_seed1997 \
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
        --set dual_mask_s_protect_enabled=true \
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
        --set wandb_group=imgr10_p_strength_5090 \
        --set dual_mask_private_conflict_strength=0.25 \
        --set 'wandb_tags=["imgr10","p_strength","beta025"]' \
        2>&1 | tee "$LOG_DIR/imgr10_p_beta025_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_p_beta025_seed1997"
else
    echo "FAIL imgr10_p_beta025_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_p_beta050_seed1997"
echo "Changed: Matched baseline: beta_S=0.50 and beta_P=0.50"
echo "Log: $LOG_DIR/imgr10_p_beta050_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_p_beta050_seed1997 \
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
        --set dual_mask_s_protect_enabled=true \
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
        --set wandb_group=imgr10_p_strength_5090 \
        --set dual_mask_private_conflict_strength=0.5 \
        --set 'wandb_tags=["imgr10","p_strength","beta050"]' \
        2>&1 | tee "$LOG_DIR/imgr10_p_beta050_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_p_beta050_seed1997"
else
    echo "FAIL imgr10_p_beta050_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_p_beta075_seed1997"
echo "Changed: Keep beta_S=0.50; set only beta_P=0.75"
echo "Log: $LOG_DIR/imgr10_p_beta075_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_p_beta075_seed1997 \
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
        --set dual_mask_s_protect_enabled=true \
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
        --set wandb_group=imgr10_p_strength_5090 \
        --set dual_mask_private_conflict_strength=0.75 \
        --set 'wandb_tags=["imgr10","p_strength","beta075"]' \
        2>&1 | tee "$LOG_DIR/imgr10_p_beta075_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_p_beta075_seed1997"
else
    echo "FAIL imgr10_p_beta075_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Finished 9 runs for imgr10_p_strength; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
