#!/usr/bin/env bash
set -euo pipefail

[[ -f main.py ]] || { echo 'Run from the repository root.' >&2; exit 2; }
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONUNBUFFERED=1
tests=(test.test_task_score_routing test.test_raw_task_score_diagnostics test.test_raw_task_score_script)
if [[ ${1:-} == --smoke ]]; then
    PYTHONWARNINGS=ignore "$PYTHON_BIN" -m unittest "${tests[@]}"
    exit 0
fi
dry_run=false
if [[ ${1:-} == --dry-run ]]; then dry_run=true; shift; fi
[[ $# == 0 ]] || { echo 'Usage: bash scripts/9_12_raw_task_score_diagnostic.sh [--smoke|--dry-run]' >&2; exit 2; }

if $dry_run; then
    run_dir=logs/shell_logs/raw_task_score_DRY_RUN
else
    data_root=$("$PYTHON_BIN" -c 'import json; print(json.load(open("exps/dlora/imgr10.json"))["data_path"])')
    echo "Dataset from exps/dlora/imgr10.json: $data_root"
    [[ -d $data_root/train && -d $data_root/test ]] || { echo 'Existing train/ and test/ required.' >&2; exit 2; }
    PYTHONWARNINGS=ignore "$PYTHON_BIN" -m unittest "${tests[@]}"
    "$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); print("Visible GPU:", torch.cuda.get_device_name(int(__import__("os").environ.get("DEVICE", "0"))))'
    echo "Code revision: $(git rev-parse --short HEAD)"
    git status --short
    mkdir -p logs/shell_logs
    run_dir=$(mktemp -d "logs/shell_logs/raw_task_score_$(date +%Y%m%d_%H%M%S)_XXXXXX")
fi

common=(--config exps/dlora/imgr10.json
    --set "device=\"${DEVICE:-0}\""
    --set total_sessions=10 --set init_cls=20 --set increment=20
    --set init_epoch=20 --set epochs=20 --set rank=64 --set ca=true --set ca_epochs=5
    --set classification_training_mode=task_local
    --set classification_inference_mode=global
    --set classification_inference_diagnostics=false
    --set classification_task_score_distribution_diagnostics=true
    --set classification_ncm_task_evidence_diagnostics=true
    --set dual_mask_ca_diagnostics=true
    --set dual_mask_private_rank=0 --set dual_mask_competence_adaptive=true
    --set dual_mask_plasticity_adaptive=true --set dual_mask_protect_strength_mode=competence
    --set dual_mask_task0_gate_mode=unmasked --set dual_mask_anchor_reg_enabled=true
    --set dual_mask_anchor_reg_weight=10 --set dual_mask_anchor_reg_task0_only=true
    --set dual_mask_conflict_energy_adaptive=true --set dual_mask_conflict_energy_ratio_floor=true
    --set dual_mask_conflict_ratio=0.1 --set dual_mask_conflict_strength=0.5
    --set dual_mask_conflict_old_overlap_adaptive=true --set dual_mask_private_conflict_mode=global
    --set dual_mask_conflict_merge_mode=suppress --set dual_mask_conflict_reg_enabled=false
    --set dual_mask_reg_weight=0.01 --set dual_mask_selective_anchor_enabled=false
    --set dual_mask_functional_merge_calibration=false --set dual_mask_safe_residual_enabled=false
    --set dual_mask_previous_function_enabled=false --set dual_mask_task_bias_calibration=false
    --set dual_mask_boundary_calibration=false
    --set dual_mask_track_w0_metrics=true --set dual_mask_vis=false
    --set experiment_tracker=wandb --set wandb_project=LoDA_ICML2026
    --set "wandb_mode=${WANDB_MODE:-online}" --set wandb_group=imgr10_raw_task_score_diagnostic
    --set wandb_tags=imgr10,raw_task_score,ncm_evidence,test_only,global,ca5,task2_screen)

seed=1993
name=imgr10_raw_task_score_diagnostic_seed${seed}
cmd=("$PYTHON_BIN" main.py "${common[@]}" --set "seed=[$seed]"
    --set "prefix=${name}_$(basename "$run_dir")" --set max_tasks=3)
echo "============================================================"
echo "Starting $name"
printf '%q ' "${cmd[@]}"
printf '\n'
if ! $dry_run; then
    "${cmd[@]}" 2>&1 | tee "$run_dir/$name.log"
    echo "PASS $name"
fi
echo "Finished one Task0-2 test-only raw task-score diagnostic: $run_dir"
