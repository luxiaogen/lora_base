#!/usr/bin/env bash
set -euo pipefail

# Run from repository root. No cd. One script, two machine-specific comparisons.
# bash scripts/9_11_head_isolation_pair.sh 3090
# bash scripts/9_11_head_isolation_pair.sh 5090
# Data path comes from exps/dlora/imgr10.json on each machine.
# --smoke: CPU tests only. --dry-run 3090: print commands, no training.
[[ -f main.py ]] || { echo 'Run from the repository root.' >&2; exit 2; }
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONUNBUFFERED=1
tests=(test.test_cross_task_training test.test_task0_checkpoint test.test_task0_trainer test.test_head_isolation_script test.test_private_rank test.test_ca_diagnostics)
if [[ ${1:-} == --smoke ]]; then
    PYTHONWARNINGS=ignore "$PYTHON_BIN" -m unittest "${tests[@]}"
    exit 0
fi
dry_run=false
if [[ ${1:-} == --dry-run ]]; then dry_run=true; shift; fi
machine=${1:-}
case "$machine" in
    3090) candidate=task_local_head ;;
    5090) candidate=task_local_head_replay ;;
    *) echo 'Usage: bash scripts/9_11_head_isolation_pair.sh [--dry-run] 3090|5090' >&2; exit 2 ;;
esac
[[ $# == 1 ]] || { echo 'Data path is read from exps/dlora/imgr10.json; do not pass a directory.' >&2; exit 2; }
read -r -a seeds <<< "${SEEDS:-1993}"
for seed in "${seeds[@]}"; do
    [[ $seed =~ ^[0-9]+$ ]] || { echo 'SEEDS must contain space-separated integer seeds.' >&2; exit 2; }
done
if $dry_run; then
    run_dir="logs/shell_logs/head_isolation_${machine}_DRY_RUN"
else
    data_root=$("$PYTHON_BIN" -c 'import json; print(json.load(open("exps/dlora/imgr10.json"))["data_path"])')
    echo "Dataset from exps/dlora/imgr10.json: $data_root"
    [[ -d $data_root/train && -d $data_root/test ]] || { echo 'Existing train/ and test/ required; will not auto-split data.' >&2; exit 2; }
    PYTHONWARNINGS=ignore "$PYTHON_BIN" -m unittest "${tests[@]}"
    "$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); print("Visible GPU:", torch.cuda.get_device_name(int(__import__("os").environ.get("DEVICE", "0"))))'
    echo "Code revision: $(git rev-parse --short HEAD)"
    git status --short
    mkdir -p logs/shell_logs
    run_dir=$(mktemp -d "logs/shell_logs/head_isolation_${machine}_$(date +%Y%m%d_%H%M%S)_XXXXXX")
fi

common=(--config exps/dlora/imgr10.json
    --set "device=\"${DEVICE:-0}\""
    --set total_sessions=10 --set init_cls=20 --set increment=20
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
    --set "wandb_group=imgr10_head_isolation_${machine}"
    --set "wandb_tags=imgr10,head_isolation,${machine},ca5,partial_task2")

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

for seed in "${seeds[@]}"; do
    checkpoint="$run_dir/task0_seed${seed}.pt"
    run_stage "imgr10_${machine}_task0_seed${seed}" \
        --set classification_training_mode=task_local --set max_tasks=1 --set "task0_checkpoint_save=$checkpoint"
    run_stage "imgr10_${machine}_baseline_seed${seed}" \
        --set classification_training_mode=task_local --set max_tasks=3 --set "task0_checkpoint_resume=$checkpoint"
    run_stage "imgr10_${machine}_${candidate}_seed${seed}" \
        --set "classification_training_mode=$candidate" --set max_tasks=3 --set "task0_checkpoint_resume=$checkpoint"
done
echo "Finished paired Task1-2 diagnostics (NOT full T10 experiments). Logs/checkpoints: $run_dir"
