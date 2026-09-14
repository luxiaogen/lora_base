#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root; each server uses its own data_path in the JSON.
cmd=(
  python main.py
  --config exps/dlora/imgr10.json
  --set 'seed=[1993]'
  --set prefix=imgr10_p_conflict_first_round_seed1993
  --set max_tasks=3
  --set total_sessions=10
  --set init_cls=20
  --set increment=20
  --set use_slora=true
  --set use_plora=true
  --set lora_A_init=kaiming
  --set init_epoch=20
  --set epochs=20
  --set ca=true
  --set ca_epochs=5
  --set rank=64
  --set task0_checkpoint_resume=
  --set task0_checkpoint_save=
  --set dual_mask_private_rank=0
  --set dual_mask_competence_adaptive=true
  --set dual_mask_plasticity_adaptive=true
  --set dual_mask_protect_strength_mode=competence
  --set dual_mask_task0_gate_mode=unmasked
  --set dual_mask_conflict_energy_adaptive=true
  --set dual_mask_conflict_energy_ratio_floor=true
  --set dual_mask_conflict_ratio=0.1
  --set dual_mask_conflict_strength=0.5
  --set dual_mask_conflict_old_overlap_adaptive=true
  --set dual_mask_private_conflict_mode=global
  --set dual_mask_conflict_merge_mode=suppress
  --set dual_mask_conflict_reg_enabled=false
  --set dual_mask_reg_weight=0.01
  --set dual_mask_anchor_reg_enabled=true
  --set dual_mask_anchor_reg_weight=10.0
  --set dual_mask_anchor_reg_task0_only=true
  --set dual_mask_selective_anchor_enabled=false
  --set dual_mask_functional_merge_calibration=false
  --set dual_mask_safe_residual_enabled=false
  --set dual_mask_lora_inherit=false
  --set dual_mask_qv_after_task0=false
  --set dual_mask_task0_qk=false
  --set dual_mask_p_conflict_diagnostics=true
  --set dual_mask_ca_diagnostics=false
  --set dual_mask_track_w0_metrics=true
  --set dual_mask_vis=false
  --set experiment_tracker=wandb
  --set wandb_project=LoDA_ICML2026
  --set wandb_mode=online
  --set wandb_group=imgr10_p_conflict_first_round
  --set wandb_tags=imgr10,p_conflict,first_round,seed1993
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

python -m unittest test.test_p_conflict_diagnostics
"${cmd[@]}"
