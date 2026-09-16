#!/usr/bin/env bash
set -uo pipefail

[[ -f main.py ]] || { echo 'Run from the repository root.' >&2; exit 2; }
python -m unittest test.test_task_consistency_scripts test.test_predicted_onehot test.test_p_conflict_diagnostics || exit 1
for config in exps/dlora/imgr10.json exps/dlora/cub10.json exps/dlora/imga10.json exps/dlora/cifar10.json; do
    python -c 'import json, pathlib, sys; p=pathlib.Path(json.load(open(sys.argv[1]))["data_path"]); print(sys.argv[1], "->", p); assert (p/"train").is_dir() and (p/"test").is_dir(), "Missing train/ or test/"' "$config" || exit 1
done
echo "Code revision: $(git rev-parse --short HEAD)"
git status --short
LOG_DIR=logs/shell_logs/task_consistency_5090_overnight
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_task_consistency_seed1993"
echo "Changed: Fresh QKV training; compare baseline, Top-2 gain and task-consistent gain on one model"
echo "Log: $LOG_DIR/imgr10_task_consistency_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_task_consistency_seed1993 \
        --set max_tasks=10 \
        --set total_sessions=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_lora_inherit=false \
        --set dual_mask_task0_qk=false \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_qk_all_tasks=false \
        --set dual_mask_qv_all_tasks=false \
        --set dual_mask_p_conflict_diagnostics=true \
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=task_consistency_5090_overnight \
        --set wandb_tags=qkv,task_consistency,p_conflict,diagnostic,3seeds \
        --set rank=64 \
        --set init_cls=20 \
        --set increment=20 \
        2>&1 | tee "$LOG_DIR/imgr10_task_consistency_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_task_consistency_seed1993"
else
    echo "FAIL imgr10_task_consistency_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_task_consistency_seed1996"
echo "Changed: Fresh QKV training; compare baseline, Top-2 gain and task-consistent gain on one model"
echo "Log: $LOG_DIR/imgr10_task_consistency_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_task_consistency_seed1996 \
        --set max_tasks=10 \
        --set total_sessions=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_lora_inherit=false \
        --set dual_mask_task0_qk=false \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_qk_all_tasks=false \
        --set dual_mask_qv_all_tasks=false \
        --set dual_mask_p_conflict_diagnostics=true \
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=task_consistency_5090_overnight \
        --set wandb_tags=qkv,task_consistency,p_conflict,diagnostic,3seeds \
        --set rank=64 \
        --set init_cls=20 \
        --set increment=20 \
        2>&1 | tee "$LOG_DIR/imgr10_task_consistency_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_task_consistency_seed1996"
else
    echo "FAIL imgr10_task_consistency_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_task_consistency_seed1997"
echo "Changed: Fresh QKV training; compare baseline, Top-2 gain and task-consistent gain on one model"
echo "Log: $LOG_DIR/imgr10_task_consistency_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_task_consistency_seed1997 \
        --set max_tasks=10 \
        --set total_sessions=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_lora_inherit=false \
        --set dual_mask_task0_qk=false \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_qk_all_tasks=false \
        --set dual_mask_qv_all_tasks=false \
        --set dual_mask_p_conflict_diagnostics=true \
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=task_consistency_5090_overnight \
        --set wandb_tags=qkv,task_consistency,p_conflict,diagnostic,3seeds \
        --set rank=64 \
        --set init_cls=20 \
        --set increment=20 \
        2>&1 | tee "$LOG_DIR/imgr10_task_consistency_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_task_consistency_seed1997"
else
    echo "FAIL imgr10_task_consistency_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting cub10_task_consistency_seed1993"
echo "Changed: Fresh QKV training; compare baseline, Top-2 gain and task-consistent gain on one model"
echo "Log: $LOG_DIR/cub10_task_consistency_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/cub10.json \
        --set 'seed=[1993]' \
        --set prefix=cub10_task_consistency_seed1993 \
        --set max_tasks=10 \
        --set total_sessions=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_lora_inherit=false \
        --set dual_mask_task0_qk=false \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_qk_all_tasks=false \
        --set dual_mask_qv_all_tasks=false \
        --set dual_mask_p_conflict_diagnostics=true \
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=task_consistency_5090_overnight \
        --set wandb_tags=qkv,task_consistency,p_conflict,diagnostic,3seeds \
        --set rank=32 \
        --set init_cls=20 \
        --set increment=20 \
        2>&1 | tee "$LOG_DIR/cub10_task_consistency_seed1993_${TIMESTAMP}.log"
then
    echo "PASS cub10_task_consistency_seed1993"
else
    echo "FAIL cub10_task_consistency_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting cub10_task_consistency_seed1996"
echo "Changed: Fresh QKV training; compare baseline, Top-2 gain and task-consistent gain on one model"
echo "Log: $LOG_DIR/cub10_task_consistency_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/cub10.json \
        --set 'seed=[1996]' \
        --set prefix=cub10_task_consistency_seed1996 \
        --set max_tasks=10 \
        --set total_sessions=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_lora_inherit=false \
        --set dual_mask_task0_qk=false \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_qk_all_tasks=false \
        --set dual_mask_qv_all_tasks=false \
        --set dual_mask_p_conflict_diagnostics=true \
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=task_consistency_5090_overnight \
        --set wandb_tags=qkv,task_consistency,p_conflict,diagnostic,3seeds \
        --set rank=32 \
        --set init_cls=20 \
        --set increment=20 \
        2>&1 | tee "$LOG_DIR/cub10_task_consistency_seed1996_${TIMESTAMP}.log"
then
    echo "PASS cub10_task_consistency_seed1996"
else
    echo "FAIL cub10_task_consistency_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting cub10_task_consistency_seed1997"
echo "Changed: Fresh QKV training; compare baseline, Top-2 gain and task-consistent gain on one model"
echo "Log: $LOG_DIR/cub10_task_consistency_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/cub10.json \
        --set 'seed=[1997]' \
        --set prefix=cub10_task_consistency_seed1997 \
        --set max_tasks=10 \
        --set total_sessions=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_lora_inherit=false \
        --set dual_mask_task0_qk=false \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_qk_all_tasks=false \
        --set dual_mask_qv_all_tasks=false \
        --set dual_mask_p_conflict_diagnostics=true \
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=task_consistency_5090_overnight \
        --set wandb_tags=qkv,task_consistency,p_conflict,diagnostic,3seeds \
        --set rank=32 \
        --set init_cls=20 \
        --set increment=20 \
        2>&1 | tee "$LOG_DIR/cub10_task_consistency_seed1997_${TIMESTAMP}.log"
then
    echo "PASS cub10_task_consistency_seed1997"
else
    echo "FAIL cub10_task_consistency_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_task_consistency_seed1993"
echo "Changed: Fresh QKV training; compare baseline, Top-2 gain and task-consistent gain on one model"
echo "Log: $LOG_DIR/imga10_task_consistency_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imga10.json \
        --set 'seed=[1993]' \
        --set prefix=imga10_task_consistency_seed1993 \
        --set max_tasks=10 \
        --set total_sessions=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_lora_inherit=false \
        --set dual_mask_task0_qk=false \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_qk_all_tasks=false \
        --set dual_mask_qv_all_tasks=false \
        --set dual_mask_p_conflict_diagnostics=true \
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=task_consistency_5090_overnight \
        --set wandb_tags=qkv,task_consistency,p_conflict,diagnostic,3seeds \
        --set rank=32 \
        --set init_cls=20 \
        --set increment=20 \
        2>&1 | tee "$LOG_DIR/imga10_task_consistency_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imga10_task_consistency_seed1993"
else
    echo "FAIL imga10_task_consistency_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_task_consistency_seed1996"
echo "Changed: Fresh QKV training; compare baseline, Top-2 gain and task-consistent gain on one model"
echo "Log: $LOG_DIR/imga10_task_consistency_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imga10.json \
        --set 'seed=[1996]' \
        --set prefix=imga10_task_consistency_seed1996 \
        --set max_tasks=10 \
        --set total_sessions=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_lora_inherit=false \
        --set dual_mask_task0_qk=false \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_qk_all_tasks=false \
        --set dual_mask_qv_all_tasks=false \
        --set dual_mask_p_conflict_diagnostics=true \
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=task_consistency_5090_overnight \
        --set wandb_tags=qkv,task_consistency,p_conflict,diagnostic,3seeds \
        --set rank=32 \
        --set init_cls=20 \
        --set increment=20 \
        2>&1 | tee "$LOG_DIR/imga10_task_consistency_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imga10_task_consistency_seed1996"
else
    echo "FAIL imga10_task_consistency_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_task_consistency_seed1997"
echo "Changed: Fresh QKV training; compare baseline, Top-2 gain and task-consistent gain on one model"
echo "Log: $LOG_DIR/imga10_task_consistency_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imga10.json \
        --set 'seed=[1997]' \
        --set prefix=imga10_task_consistency_seed1997 \
        --set max_tasks=10 \
        --set total_sessions=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_lora_inherit=false \
        --set dual_mask_task0_qk=false \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_qk_all_tasks=false \
        --set dual_mask_qv_all_tasks=false \
        --set dual_mask_p_conflict_diagnostics=true \
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=task_consistency_5090_overnight \
        --set wandb_tags=qkv,task_consistency,p_conflict,diagnostic,3seeds \
        --set rank=32 \
        --set init_cls=20 \
        --set increment=20 \
        2>&1 | tee "$LOG_DIR/imga10_task_consistency_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imga10_task_consistency_seed1997"
else
    echo "FAIL imga10_task_consistency_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting cifar10_task_consistency_seed1993"
echo "Changed: Fresh QKV training; compare baseline, Top-2 gain and task-consistent gain on one model"
echo "Log: $LOG_DIR/cifar10_task_consistency_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/cifar10.json \
        --set 'seed=[1993]' \
        --set prefix=cifar10_task_consistency_seed1993 \
        --set max_tasks=10 \
        --set total_sessions=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_lora_inherit=false \
        --set dual_mask_task0_qk=false \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_qk_all_tasks=false \
        --set dual_mask_qv_all_tasks=false \
        --set dual_mask_p_conflict_diagnostics=true \
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=task_consistency_5090_overnight \
        --set wandb_tags=qkv,task_consistency,p_conflict,diagnostic,3seeds \
        --set rank=32 \
        --set init_cls=10 \
        --set increment=10 \
        2>&1 | tee "$LOG_DIR/cifar10_task_consistency_seed1993_${TIMESTAMP}.log"
then
    echo "PASS cifar10_task_consistency_seed1993"
else
    echo "FAIL cifar10_task_consistency_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting cifar10_task_consistency_seed1996"
echo "Changed: Fresh QKV training; compare baseline, Top-2 gain and task-consistent gain on one model"
echo "Log: $LOG_DIR/cifar10_task_consistency_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/cifar10.json \
        --set 'seed=[1996]' \
        --set prefix=cifar10_task_consistency_seed1996 \
        --set max_tasks=10 \
        --set total_sessions=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_lora_inherit=false \
        --set dual_mask_task0_qk=false \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_qk_all_tasks=false \
        --set dual_mask_qv_all_tasks=false \
        --set dual_mask_p_conflict_diagnostics=true \
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=task_consistency_5090_overnight \
        --set wandb_tags=qkv,task_consistency,p_conflict,diagnostic,3seeds \
        --set rank=32 \
        --set init_cls=10 \
        --set increment=10 \
        2>&1 | tee "$LOG_DIR/cifar10_task_consistency_seed1996_${TIMESTAMP}.log"
then
    echo "PASS cifar10_task_consistency_seed1996"
else
    echo "FAIL cifar10_task_consistency_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting cifar10_task_consistency_seed1997"
echo "Changed: Fresh QKV training; compare baseline, Top-2 gain and task-consistent gain on one model"
echo "Log: $LOG_DIR/cifar10_task_consistency_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/cifar10.json \
        --set 'seed=[1997]' \
        --set prefix=cifar10_task_consistency_seed1997 \
        --set max_tasks=10 \
        --set total_sessions=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
        --set dual_mask_private_rank=0 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_lora_inherit=false \
        --set dual_mask_task0_qk=false \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_qk_all_tasks=false \
        --set dual_mask_qv_all_tasks=false \
        --set dual_mask_p_conflict_diagnostics=true \
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=task_consistency_5090_overnight \
        --set wandb_tags=qkv,task_consistency,p_conflict,diagnostic,3seeds \
        --set rank=32 \
        --set init_cls=10 \
        --set increment=10 \
        2>&1 | tee "$LOG_DIR/cifar10_task_consistency_seed1997_${TIMESTAMP}.log"
then
    echo "PASS cifar10_task_consistency_seed1997"
else
    echo "FAIL cifar10_task_consistency_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Finished 12 runs for task_consistency_5090_overnight; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
