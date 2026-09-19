#!/usr/bin/env bash
set -uo pipefail
# Run from the repository root. Data paths come only from this machine's JSON.
DRY_RUN=${DRY_RUN:-0}
SMOKE_ONLY=${SMOKE_ONLY:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
BUDGET_SECONDS=${BUDGET_SECONDS:-43200}
ESTIMATE_SECONDS=${ESTIMATED_RUN_SECONDS:-3600}
LOG_DIR=${LOG_DIR:-logs/shell_logs/p_region_train_merge_5090}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SWEEP_START=$(date +%s)
FAILED=0
ATTEMPTED=0
SUCCEEDED=0
[[ "$BUDGET_SECONDS" =~ ^[0-9]+$ && "$ESTIMATE_SECONDS" =~ ^[0-9]+$ ]] || exit 2
if [[ ! -f main.py || ! -f exps/dlora/imgr10.json ]]; then
    echo "Run this script from the repository root." >&2
    exit 1
fi
if [[ "$DRY_RUN" != 1 ]]; then
    git rev-parse HEAD
    git status --short
    "$PYTHON_BIN" -m unittest test.test_p_region_training test.test_p_region_diagnostic test.test_dual_mask_core test.test_ca_diagnostics -q || exit 1
    [[ "$SMOKE_ONLY" == 1 ]] && exit 0
    "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
import torch
config = json.load(open("exps/dlora/imgr10.json"))
root = Path(config["data_path"])
print("Dataset from JSON:", root)
assert config["dataset"] == "ImageNet_R", "Expected ImageNet_R"
assert (root / "train").is_dir() and (root / "test").is_dir(), "Existing train/test required; no auto-splitting"
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "Activate the server CUDA environment first"
print("GPU:", torch.cuda.get_device_name(0))
PY
    [[ $? == 0 ]] || exit 1
fi
mkdir -p "$LOG_DIR"
STATUS_FILE="$LOG_DIR/status_${TIMESTAMP}.tsv"
printf 'run\tstatus\tseconds\n' > "$STATUS_FILE"

before_run() {
    CURRENT_RUN=$1
    local elapsed
    elapsed=$(( $(date +%s) - SWEEP_START ))
    if (( BUDGET_SECONDS > 0 && elapsed + ESTIMATE_SECONDS > BUDGET_SECONDS )); then
        echo "BUDGET STOP: attempted $ATTEMPTED/12; passed=$SUCCEEDED; deferred $((12-ATTEMPTED)); next=$CURRENT_RUN"
        echo "No running experiment was killed. This is NOT a complete 12-run sweep."
        printf '%s\tDEFERRED_BUDGET\t0\n' "$CURRENT_RUN" >> "$STATUS_FILE"
        exit "$FAILED"
    fi
    RUN_START=$(date +%s)
}

finish_run() {
    local seconds estimate
    seconds=$(( $(date +%s) - RUN_START ))
    ATTEMPTED=$((ATTEMPTED+1))
    [[ "$1" == PASS ]] && SUCCEEDED=$((SUCCEEDED+1))
    printf '%s\t%s\t%s\n' "$CURRENT_RUN" "$1" "$seconds" >> "$STATUS_FILE"
    if [[ "$DRY_RUN" != 1 && "$1" == PASS ]]; then
        estimate=$((seconds * 115 / 100))
        (( estimate > ESTIMATE_SECONDS )) && ESTIMATE_SECONDS=$estimate
    fi
    return 0
}

run_python() {
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'DRYRUN python '
        printf '%q ' "$@"
        printf '\n'
    else
        "$PYTHON_BIN" -u "$@"
    fi
}

echo "============================================================"
before_run "imgr10_baseline_seed1993"
echo "Starting imgr10_baseline_seed1993"
echo "Changed: Original baseline; extra attenuation disabled"
echo "Log: $LOG_DIR/imgr10_baseline_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    run_python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_baseline_seed1993_${TIMESTAMP}" \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_private_rank=0 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=p_region_train_merge_5090 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_p_region_diagnostic=false \
        --set dual_mask_p_region_train_mode=none \
        --set dual_mask_p_region_train_amount=0.5 \
        2>&1 | tee "$LOG_DIR/imgr10_baseline_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_baseline_seed1993"
    finish_run PASS
else
    echo "FAIL imgr10_baseline_seed1993"
    finish_run FAIL
    FAILED=1
fi

echo "============================================================"
before_run "imgr10_conflict50_seed1993"
echo "Starting imgr10_conflict50_seed1993"
echo "Changed: Extra C attenuation in training AND merge"
echo "Log: $LOG_DIR/imgr10_conflict50_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    run_python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_conflict50_seed1993_${TIMESTAMP}" \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_private_rank=0 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=p_region_train_merge_5090 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_p_region_diagnostic=false \
        --set dual_mask_p_region_train_mode=conflict \
        --set dual_mask_p_region_train_amount=0.5 \
        2>&1 | tee "$LOG_DIR/imgr10_conflict50_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_conflict50_seed1993"
    finish_run PASS
else
    echo "FAIL imgr10_conflict50_seed1993"
    finish_run FAIL
    FAILED=1
fi

echo "============================================================"
before_run "imgr10_nonconflict50_seed1993"
echo "Starting imgr10_nonconflict50_seed1993"
echo "Changed: Matched-budget U attenuation in training AND merge"
echo "Log: $LOG_DIR/imgr10_nonconflict50_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    run_python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_nonconflict50_seed1993_${TIMESTAMP}" \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_private_rank=0 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=p_region_train_merge_5090 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_p_region_diagnostic=false \
        --set dual_mask_p_region_train_mode=nonconflict \
        --set dual_mask_p_region_train_amount=0.5 \
        2>&1 | tee "$LOG_DIR/imgr10_nonconflict50_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_nonconflict50_seed1993"
    finish_run PASS
else
    echo "FAIL imgr10_nonconflict50_seed1993"
    finish_run FAIL
    FAILED=1
fi

echo "============================================================"
before_run "imgr10_conflict25_seed1993"
echo "Starting imgr10_conflict25_seed1993"
echo "Changed: Milder C attenuation"
echo "Log: $LOG_DIR/imgr10_conflict25_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    run_python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_conflict25_seed1993_${TIMESTAMP}" \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_private_rank=0 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=p_region_train_merge_5090 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_p_region_diagnostic=false \
        --set dual_mask_p_region_train_mode=conflict \
        --set dual_mask_p_region_train_amount=0.25 \
        2>&1 | tee "$LOG_DIR/imgr10_conflict25_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_conflict25_seed1993"
    finish_run PASS
else
    echo "FAIL imgr10_conflict25_seed1993"
    finish_run FAIL
    FAILED=1
fi

echo "============================================================"
before_run "imgr10_nonconflict25_seed1993"
echo "Starting imgr10_nonconflict25_seed1993"
echo "Changed: Milder matched U control"
echo "Log: $LOG_DIR/imgr10_nonconflict25_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    run_python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_nonconflict25_seed1993_${TIMESTAMP}" \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_private_rank=0 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=p_region_train_merge_5090 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_p_region_diagnostic=false \
        --set dual_mask_p_region_train_mode=nonconflict \
        --set dual_mask_p_region_train_amount=0.25 \
        2>&1 | tee "$LOG_DIR/imgr10_nonconflict25_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_nonconflict25_seed1993"
    finish_run PASS
else
    echo "FAIL imgr10_nonconflict25_seed1993"
    finish_run FAIL
    FAILED=1
fi

echo "============================================================"
before_run "imgr10_baseline_repeat_seed1993"
echo "Starting imgr10_baseline_repeat_seed1993"
echo "Changed: Same seed baseline repetition; fresh training"
echo "Log: $LOG_DIR/imgr10_baseline_repeat_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    run_python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_baseline_repeat_seed1993_${TIMESTAMP}" \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_private_rank=0 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=p_region_train_merge_5090 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_p_region_diagnostic=false \
        --set dual_mask_p_region_train_mode=none \
        --set dual_mask_p_region_train_amount=0.5 \
        2>&1 | tee "$LOG_DIR/imgr10_baseline_repeat_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_baseline_repeat_seed1993"
    finish_run PASS
else
    echo "FAIL imgr10_baseline_repeat_seed1993"
    finish_run FAIL
    FAILED=1
fi


echo "============================================================"
before_run "imgr10_baseline_seed1996"
echo "Starting imgr10_baseline_seed1996"
echo "Changed: Original baseline; extra attenuation disabled"
echo "Log: $LOG_DIR/imgr10_baseline_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    run_python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set "prefix=imgr10_baseline_seed1996_${TIMESTAMP}" \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_private_rank=0 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=p_region_train_merge_5090 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_p_region_diagnostic=false \
        --set dual_mask_p_region_train_mode=none \
        --set dual_mask_p_region_train_amount=0.5 \
        2>&1 | tee "$LOG_DIR/imgr10_baseline_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_baseline_seed1996"
    finish_run PASS
else
    echo "FAIL imgr10_baseline_seed1996"
    finish_run FAIL
    FAILED=1
fi

echo "============================================================"
before_run "imgr10_conflict50_seed1996"
echo "Starting imgr10_conflict50_seed1996"
echo "Changed: Extra C attenuation in training AND merge"
echo "Log: $LOG_DIR/imgr10_conflict50_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    run_python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set "prefix=imgr10_conflict50_seed1996_${TIMESTAMP}" \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_private_rank=0 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=p_region_train_merge_5090 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_p_region_diagnostic=false \
        --set dual_mask_p_region_train_mode=conflict \
        --set dual_mask_p_region_train_amount=0.5 \
        2>&1 | tee "$LOG_DIR/imgr10_conflict50_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_conflict50_seed1996"
    finish_run PASS
else
    echo "FAIL imgr10_conflict50_seed1996"
    finish_run FAIL
    FAILED=1
fi

echo "============================================================"
before_run "imgr10_nonconflict50_seed1996"
echo "Starting imgr10_nonconflict50_seed1996"
echo "Changed: Matched-budget U attenuation in training AND merge"
echo "Log: $LOG_DIR/imgr10_nonconflict50_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    run_python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1996]' \
        --set "prefix=imgr10_nonconflict50_seed1996_${TIMESTAMP}" \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_private_rank=0 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=p_region_train_merge_5090 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_p_region_diagnostic=false \
        --set dual_mask_p_region_train_mode=nonconflict \
        --set dual_mask_p_region_train_amount=0.5 \
        2>&1 | tee "$LOG_DIR/imgr10_nonconflict50_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_nonconflict50_seed1996"
    finish_run PASS
else
    echo "FAIL imgr10_nonconflict50_seed1996"
    finish_run FAIL
    FAILED=1
fi

echo "============================================================"
before_run "imgr10_baseline_seed1997"
echo "Starting imgr10_baseline_seed1997"
echo "Changed: Original baseline; extra attenuation disabled"
echo "Log: $LOG_DIR/imgr10_baseline_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    run_python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set "prefix=imgr10_baseline_seed1997_${TIMESTAMP}" \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_private_rank=0 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=p_region_train_merge_5090 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_p_region_diagnostic=false \
        --set dual_mask_p_region_train_mode=none \
        --set dual_mask_p_region_train_amount=0.5 \
        2>&1 | tee "$LOG_DIR/imgr10_baseline_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_baseline_seed1997"
    finish_run PASS
else
    echo "FAIL imgr10_baseline_seed1997"
    finish_run FAIL
    FAILED=1
fi

echo "============================================================"
before_run "imgr10_conflict50_seed1997"
echo "Starting imgr10_conflict50_seed1997"
echo "Changed: Extra C attenuation in training AND merge"
echo "Log: $LOG_DIR/imgr10_conflict50_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    run_python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set "prefix=imgr10_conflict50_seed1997_${TIMESTAMP}" \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_private_rank=0 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=p_region_train_merge_5090 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_p_region_diagnostic=false \
        --set dual_mask_p_region_train_mode=conflict \
        --set dual_mask_p_region_train_amount=0.5 \
        2>&1 | tee "$LOG_DIR/imgr10_conflict50_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_conflict50_seed1997"
    finish_run PASS
else
    echo "FAIL imgr10_conflict50_seed1997"
    finish_run FAIL
    FAILED=1
fi

echo "============================================================"
before_run "imgr10_nonconflict50_seed1997"
echo "Starting imgr10_nonconflict50_seed1997"
echo "Changed: Matched-budget U attenuation in training AND merge"
echo "Log: $LOG_DIR/imgr10_nonconflict50_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    run_python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1997]' \
        --set "prefix=imgr10_nonconflict50_seed1997_${TIMESTAMP}" \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_private_rank=0 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=p_region_train_merge_5090 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_p_region_diagnostic=false \
        --set dual_mask_p_region_train_mode=nonconflict \
        --set dual_mask_p_region_train_amount=0.5 \
        2>&1 | tee "$LOG_DIR/imgr10_nonconflict50_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_nonconflict50_seed1997"
    finish_run PASS
else
    echo "FAIL imgr10_nonconflict50_seed1997"
    finish_run FAIL
    FAILED=1
fi


echo "Finished $ATTEMPTED/12 runs; passed=$SUCCEEDED; FAILED=$FAILED"
echo "Logs: $LOG_DIR; status: $STATUS_FILE"
exit "$FAILED"
