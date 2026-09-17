#!/usr/bin/env bash
set -euo pipefail

[[ -f main.py ]] || { echo 'Run from the repository root.' >&2; exit 2; }

cmd=(
  python main.py
  --config exps/dlora/imgr10.json
  --set 'seed=[1993]'
  --set prefix=imgr10_p_conflict_merge_filter_seed1993
  --set max_tasks=10
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
  --set dual_mask_anchor_reg_weight=10
  --set dual_mask_anchor_reg_task0_only=true
  --set dual_mask_selective_anchor_enabled=false
  --set dual_mask_functional_merge_calibration=false
  --set dual_mask_safe_residual_enabled=false
  --set dual_mask_lora_inherit=false
  --set dual_mask_task0_qk=false
  --set dual_mask_qv_after_task0=false
  --set dual_mask_qk_all_tasks=false
  --set dual_mask_qv_all_tasks=false
  --set dual_mask_p_conflict_merge_filter=true
  --set dual_mask_p_conflict_diagnostics=false
  --set dual_mask_ca_diagnostics=false
  --set dual_mask_track_w0_metrics=true
  --set dual_mask_vis=false
  --set experiment_tracker=wandb
  --set wandb_project=LoDA_ICML2026
  --set wandb_mode=online
  --set wandb_group=imgr10_p_conflict_merge_filter
  --set wandb_tags=imgr10,t10,qkv,seed1993,p_conflict,merge_margin_filter
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

python -m unittest test.test_p_conflict_merge_filter
python -c 'import json, pathlib; p=pathlib.Path(json.load(open("exps/dlora/imgr10.json"))["data_path"]); print("Dataset:", p); assert (p/"train").is_dir() and (p/"test").is_dir(), "Missing train/ or test/"'
"${cmd[@]}"
