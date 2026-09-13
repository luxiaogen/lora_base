#!/usr/bin/env bash
set -euo pipefail

# From repository root; dataset path comes only from the local imgr10.json.
# lrun scripts/9_13_qv_after_task0.sh ./logs/9_13_qv_after_task0.log
[[ -f main.py ]] || { echo 'Run from the repository root.' >&2; exit 2; }
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONUNBUFFERED=1
tests=(test.test_qv_after_task0 test.test_qv_script test.test_task0_trainer test.test_private_rank test.test_ca_diagnostics)
if [[ ${1:-} == --check ]]; then
    PYTHONWARNINGS=ignore "$PYTHON_BIN" -m unittest "${tests[@]}"
    exit 0
fi
dry_run=false
if [[ ${1:-} == --dry-run ]]; then dry_run=true; shift; fi
[[ $# == 0 ]] || { echo 'Usage: bash scripts/9_13_qv_after_task0.sh [--check|--dry-run]' >&2; exit 2; }
seed=${SEED:-1993}
max_tasks=${MAX_TASKS:-3}
case $max_tasks in
    3) task_tag=partial_task2 ;;
    10) task_tag=full_t10 ;;
    *) echo 'MAX_TASKS must be 3 or 10.' >&2; exit 2 ;;
esac
checkpoint=${TASK0_CHECKPOINT:-}
if [[ -n $checkpoint && $dry_run == false && ! -f $checkpoint ]]; then
    echo "Task0 checkpoint not found: $checkpoint" >&2
    exit 2
fi
if $dry_run; then
    run_dir="logs/shell_logs/qv_after_task0_DRY_RUN"
else
    data_root=$("$PYTHON_BIN" -c 'import json; print(json.load(open("exps/dlora/imgr10.json"))["data_path"])')
    echo "Dataset from exps/dlora/imgr10.json: $data_root"
    [[ -d $data_root/train && -d $data_root/test ]] || { echo "Missing train/ or test/ under $data_root; set data_path in exps/dlora/imgr10.json to this machine's existing ImageNet-R split." >&2; exit 2; }
    PYTHONWARNINGS=ignore "$PYTHON_BIN" -m unittest "${tests[@]}"
    "$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); print("Visible GPU:", torch.cuda.get_device_name(int(__import__("os").environ.get("DEVICE", "0"))))'
    echo "Code revision: $(git rev-parse --short HEAD)"
    git status --short
    mkdir -p logs/shell_logs
    run_dir=$(mktemp -d "logs/shell_logs/qv_after_task0_$(date +%Y%m%d_%H%M%S)_XXXXXX")
fi

common=(--config exps/dlora/imgr10.json
    --set "device=\"${DEVICE:-0}\""
    --set total_sessions=10 --set init_cls=20 --set increment=20
    --set use_slora=true --set use_plora=true
    --set slora_gamma=0.5 --set plora_gamma=0.75
    --set init_epoch=20 --set epochs=20 --set rank=64 --set ca=true --set ca_epochs=5
    --set dual_mask_ca_diagnostics=true --set dual_mask_private_rank=0
    --set dual_mask_competence_adaptive=true --set dual_mask_plasticity_adaptive=true
    --set dual_mask_protect_strength_mode=competence --set dual_mask_task0_gate_mode=unmasked
    --set dual_mask_anchor_reg_enabled=true --set dual_mask_anchor_reg_weight=10
    --set dual_mask_anchor_reg_task0_only=true --set dual_mask_conflict_energy_adaptive=true
    --set dual_mask_conflict_energy_ratio_floor=true --set dual_mask_conflict_ratio=0.1
    --set dual_mask_conflict_strength=0.5 --set dual_mask_conflict_old_overlap_adaptive=true
    --set dual_mask_private_conflict_mode=global --set dual_mask_conflict_merge_mode=suppress
    --set dual_mask_conflict_reg_enabled=false --set dual_mask_reg_weight=0.01
    --set dual_mask_selective_anchor_enabled=false --set dual_mask_functional_merge_calibration=false
    --set dual_mask_safe_residual_enabled=false --set dual_mask_track_w0_metrics=true --set dual_mask_vis=false
    --set experiment_tracker=wandb --set wandb_project=LoDA_ICML2026 --set "wandb_mode=${WANDB_MODE:-online}"
    --set "wandb_group=imgr10_qv_after_task0"
    --set "wandb_tags=imgr10,qv_after_task0,ca5,$task_tag")

run_stage() {
    local name=$1
    shift
    local cmd=("$PYTHON_BIN" main.py "${common[@]}" --set "seed=[$seed]" --set "prefix=${name}_$(basename "$run_dir")" "$@")
    echo "============================================================"
    echo "Starting $name"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if ! $dry_run; then
        "${cmd[@]}" 2>&1 | tee "$run_dir/$name.log"
        echo "PASS $name"
    fi
}

if [[ -z $checkpoint ]]; then
    checkpoint="$run_dir/task0_seed${seed}.pt"
    run_stage "imgr10_task0_seed${seed}" \
        --set dual_mask_qv_after_task0=false --set max_tasks=1 --set "task0_checkpoint_save=$checkpoint"
else
    echo "Reusing Task0 checkpoint: $checkpoint"
fi
run_stage "imgr10_qkv_baseline_seed${seed}" \
    --set dual_mask_qv_after_task0=false --set "max_tasks=$max_tasks" --set "task0_checkpoint_resume=$checkpoint"
run_stage "imgr10_qv_after_task0_seed${seed}" \
    --set dual_mask_qv_after_task0=true --set "max_tasks=$max_tasks" --set "task0_checkpoint_resume=$checkpoint"
if [[ $max_tasks == 10 ]]; then
    echo "Finished paired full T10 runs. Logs: $run_dir; Task0 checkpoint: $checkpoint"
else
    echo "Finished paired Task1-2 screening (NOT full T10 results). Logs/checkpoint: $run_dir"
fi
