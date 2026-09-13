#!/usr/bin/env bash
set -uo pipefail

# Run from the repository root; data_path stays in each machine's config.
[[ -f main.py ]] || { echo 'Run from the repository root.' >&2; exit 2; }
export PYTHONUNBUFFERED=1
PYTHONWARNINGS=ignore python -m unittest test.test_lora_inherit || exit 1
python -c 'import json, torch; from pathlib import Path
for name in ["imgr10","imga10"]:
    root = Path(json.load(open("exps/dlora/"+name+".json"))["data_path"])
    print(name, "data_path:", root)
    assert root.is_dir(), "Missing dataset directory; update data_path in the local config"
    if name == "imgr10":
        assert (root/"train").is_dir() and (root/"test").is_dir(), "Existing ImageNet-R train/test split required"
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))' || exit 1
git rev-parse --short HEAD
LOG_DIR=logs/shell_logs/lora_inherit_5090
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_fresh_seed1993"
echo "Changed: Fresh A/B each task; full QKV baseline"
echo "Log: $LOG_DIR/imgr10_fresh_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_fresh_seed1993_${TIMESTAMP}" \
        --set total_sessions=10 \
        --set init_cls=20 \
        --set increment=20 \
        --set max_tasks=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set lora_A_init=kaiming \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_task0_qk=false \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
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
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=lora_inherit_vs_fresh_t10 \
        --set rank=64 \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_lora_inherit=false \
        2>&1 | tee "$LOG_DIR/imgr10_fresh_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_fresh_seed1993"
else
    echo "FAIL imgr10_fresh_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_inherit_seed1993"
echo "Changed: Inherit A/B; centered BA-BA_start, full QKV"
echo "Log: $LOG_DIR/imgr10_inherit_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_inherit_seed1993_${TIMESTAMP}" \
        --set total_sessions=10 \
        --set init_cls=20 \
        --set increment=20 \
        --set max_tasks=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set lora_A_init=kaiming \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_task0_qk=false \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
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
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=lora_inherit_vs_fresh_t10 \
        --set rank=64 \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_lora_inherit=true \
        2>&1 | tee "$LOG_DIR/imgr10_inherit_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_inherit_seed1993"
else
    echo "FAIL imgr10_inherit_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_fresh_seed1996"
echo "Changed: Fresh A/B each task; full QKV baseline"
echo "Log: $LOG_DIR/imgr10_fresh_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set "prefix=imgr10_fresh_seed1996_${TIMESTAMP}" \
        --set total_sessions=10 \
        --set init_cls=20 \
        --set increment=20 \
        --set max_tasks=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set lora_A_init=kaiming \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_task0_qk=false \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
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
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=lora_inherit_vs_fresh_t10 \
        --set rank=64 \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_lora_inherit=false \
        2>&1 | tee "$LOG_DIR/imgr10_fresh_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_fresh_seed1996"
else
    echo "FAIL imgr10_fresh_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_inherit_seed1996"
echo "Changed: Inherit A/B; centered BA-BA_start, full QKV"
echo "Log: $LOG_DIR/imgr10_inherit_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set "prefix=imgr10_inherit_seed1996_${TIMESTAMP}" \
        --set total_sessions=10 \
        --set init_cls=20 \
        --set increment=20 \
        --set max_tasks=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set lora_A_init=kaiming \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_task0_qk=false \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
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
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=lora_inherit_vs_fresh_t10 \
        --set rank=64 \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_lora_inherit=true \
        2>&1 | tee "$LOG_DIR/imgr10_inherit_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_inherit_seed1996"
else
    echo "FAIL imgr10_inherit_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_fresh_seed1997"
echo "Changed: Fresh A/B each task; full QKV baseline"
echo "Log: $LOG_DIR/imgr10_fresh_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set "prefix=imgr10_fresh_seed1997_${TIMESTAMP}" \
        --set total_sessions=10 \
        --set init_cls=20 \
        --set increment=20 \
        --set max_tasks=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set lora_A_init=kaiming \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_task0_qk=false \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
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
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=lora_inherit_vs_fresh_t10 \
        --set rank=64 \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_lora_inherit=false \
        2>&1 | tee "$LOG_DIR/imgr10_fresh_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_fresh_seed1997"
else
    echo "FAIL imgr10_fresh_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_inherit_seed1997"
echo "Changed: Inherit A/B; centered BA-BA_start, full QKV"
echo "Log: $LOG_DIR/imgr10_inherit_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set "prefix=imgr10_inherit_seed1997_${TIMESTAMP}" \
        --set total_sessions=10 \
        --set init_cls=20 \
        --set increment=20 \
        --set max_tasks=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set lora_A_init=kaiming \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_task0_qk=false \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
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
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=lora_inherit_vs_fresh_t10 \
        --set rank=64 \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_lora_inherit=true \
        2>&1 | tee "$LOG_DIR/imgr10_inherit_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_inherit_seed1997"
else
    echo "FAIL imgr10_inherit_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_fresh_seed1993"
echo "Changed: Fresh A/B each task; full QKV baseline"
echo "Log: $LOG_DIR/imga10_fresh_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imga10.json \
        --set 'seed=[1993]' \
        --set "prefix=imga10_fresh_seed1993_${TIMESTAMP}" \
        --set total_sessions=10 \
        --set init_cls=20 \
        --set increment=20 \
        --set max_tasks=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set lora_A_init=kaiming \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_task0_qk=false \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
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
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=lora_inherit_vs_fresh_t10 \
        --set rank=32 \
        --set slora_gamma=0.5 \
        --set plora_gamma=1 \
        --set dual_mask_lora_inherit=false \
        2>&1 | tee "$LOG_DIR/imga10_fresh_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imga10_fresh_seed1993"
else
    echo "FAIL imga10_fresh_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_inherit_seed1993"
echo "Changed: Inherit A/B; centered BA-BA_start, full QKV"
echo "Log: $LOG_DIR/imga10_inherit_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imga10.json \
        --set 'seed=[1993]' \
        --set "prefix=imga10_inherit_seed1993_${TIMESTAMP}" \
        --set total_sessions=10 \
        --set init_cls=20 \
        --set increment=20 \
        --set max_tasks=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set lora_A_init=kaiming \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_task0_qk=false \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
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
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=lora_inherit_vs_fresh_t10 \
        --set rank=32 \
        --set slora_gamma=0.5 \
        --set plora_gamma=1 \
        --set dual_mask_lora_inherit=true \
        2>&1 | tee "$LOG_DIR/imga10_inherit_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imga10_inherit_seed1993"
else
    echo "FAIL imga10_inherit_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_fresh_seed1996"
echo "Changed: Fresh A/B each task; full QKV baseline"
echo "Log: $LOG_DIR/imga10_fresh_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imga10.json \
        --set 'seed=[1996]' \
        --set "prefix=imga10_fresh_seed1996_${TIMESTAMP}" \
        --set total_sessions=10 \
        --set init_cls=20 \
        --set increment=20 \
        --set max_tasks=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set lora_A_init=kaiming \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_task0_qk=false \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
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
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=lora_inherit_vs_fresh_t10 \
        --set rank=32 \
        --set slora_gamma=0.5 \
        --set plora_gamma=1 \
        --set dual_mask_lora_inherit=false \
        2>&1 | tee "$LOG_DIR/imga10_fresh_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imga10_fresh_seed1996"
else
    echo "FAIL imga10_fresh_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_inherit_seed1996"
echo "Changed: Inherit A/B; centered BA-BA_start, full QKV"
echo "Log: $LOG_DIR/imga10_inherit_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imga10.json \
        --set 'seed=[1996]' \
        --set "prefix=imga10_inherit_seed1996_${TIMESTAMP}" \
        --set total_sessions=10 \
        --set init_cls=20 \
        --set increment=20 \
        --set max_tasks=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set lora_A_init=kaiming \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_task0_qk=false \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
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
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=lora_inherit_vs_fresh_t10 \
        --set rank=32 \
        --set slora_gamma=0.5 \
        --set plora_gamma=1 \
        --set dual_mask_lora_inherit=true \
        2>&1 | tee "$LOG_DIR/imga10_inherit_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imga10_inherit_seed1996"
else
    echo "FAIL imga10_inherit_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_fresh_seed1997"
echo "Changed: Fresh A/B each task; full QKV baseline"
echo "Log: $LOG_DIR/imga10_fresh_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imga10.json \
        --set 'seed=[1997]' \
        --set "prefix=imga10_fresh_seed1997_${TIMESTAMP}" \
        --set total_sessions=10 \
        --set init_cls=20 \
        --set increment=20 \
        --set max_tasks=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set lora_A_init=kaiming \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_task0_qk=false \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
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
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=lora_inherit_vs_fresh_t10 \
        --set rank=32 \
        --set slora_gamma=0.5 \
        --set plora_gamma=1 \
        --set dual_mask_lora_inherit=false \
        2>&1 | tee "$LOG_DIR/imga10_fresh_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imga10_fresh_seed1997"
else
    echo "FAIL imga10_fresh_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_inherit_seed1997"
echo "Changed: Inherit A/B; centered BA-BA_start, full QKV"
echo "Log: $LOG_DIR/imga10_inherit_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imga10.json \
        --set 'seed=[1997]' \
        --set "prefix=imga10_inherit_seed1997_${TIMESTAMP}" \
        --set total_sessions=10 \
        --set init_cls=20 \
        --set increment=20 \
        --set max_tasks=10 \
        --set use_slora=true \
        --set use_plora=true \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=5 \
        --set lora_A_init=kaiming \
        --set dual_mask_qv_after_task0=false \
        --set dual_mask_task0_qk=false \
        --set task0_checkpoint_resume= \
        --set task0_checkpoint_save= \
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
        --set dual_mask_ca_diagnostics=false \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=lora_inherit_vs_fresh_t10 \
        --set rank=32 \
        --set slora_gamma=0.5 \
        --set plora_gamma=1 \
        --set dual_mask_lora_inherit=true \
        2>&1 | tee "$LOG_DIR/imga10_inherit_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imga10_inherit_seed1997"
else
    echo "FAIL imga10_inherit_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Finished 12 runs for lora_inherit_5090; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
