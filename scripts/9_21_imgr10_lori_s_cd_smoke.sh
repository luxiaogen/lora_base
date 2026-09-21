#!/usr/bin/env bash
set -uo pipefail

if [[ ! -f main.py ]]; then
    echo "Run this script from the repository root." >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${LORI_CD_RUN_MODE:-smoke}"
TASKS="${LORI_CD_TASKS:-3}"
STAGE_EPOCHS="${LORI_CD_STAGE_EPOCHS:-2}"
CA_EPOCHS="${LORI_CD_CA_EPOCHS:-1}"
export PYTHONUNBUFFERED=1
"$PYTHON_BIN" -m py_compile models/attention.py methods/dlora.py utils/lori.py trainer.py || exit 1
"$PYTHON_BIN" -m unittest test.test_lori || exit 1

LOG_DIR="logs/shell_logs/imgr10_lori_s_cd_${RUN_MODE}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"
C_LOG="$LOG_DIR/imgr10_lori_s_C_no_dualmask_${RUN_MODE}_${TIMESTAMP}.log"
D_LOG="$LOG_DIR/imgr10_lori_s_D_dualmask_${RUN_MODE}_${TIMESTAMP}.log"

COMMON_ARGS=(
    --config exps/dlora/imgr10.json
    --set 'seed=[1993]'
    --set max_tasks="$TASKS"
    --set init_epoch="$STAGE_EPOCHS"
    --set epochs="$STAGE_EPOCHS"
    --set rank=64
    --set ca=true
    --set ca_epochs="$CA_EPOCHS"
    --set lori_s_enabled=true
    --set lori_retain_ratio=0.1
    --set lori_calibration_epochs="$STAGE_EPOCHS"
    --set lori_sparse_epochs="$STAGE_EPOCHS"
    --set lora_type=lori_s
    --set dual_mask_competence_adaptive=true
    --set dual_mask_plasticity_adaptive=true
    --set dual_mask_protect_strength_mode=competence
    --set dual_mask_task0_gate_mode=unmasked
    --set dual_mask_anchor_reg_enabled=true
    --set dual_mask_anchor_reg_weight=10.0
    --set dual_mask_anchor_reg_task0_only=true
    --set dual_mask_conflict_energy_adaptive=true
    --set dual_mask_conflict_energy_ratio_floor=true
    --set dual_mask_conflict_ratio=0.1
    --set dual_mask_conflict_strength=0.5
    --set dual_mask_conflict_old_overlap_adaptive=true
    --set dual_mask_private_conflict_mode=global
    --set dual_mask_conflict_merge_mode=suppress
    --set dual_mask_conflict_reg_enabled=false
    --set dual_mask_reg_weight=0.01
    --set dual_mask_selective_anchor_enabled=false
    --set dual_mask_functional_merge_calibration=false
    --set dual_mask_safe_residual_enabled=false
    --set dual_mask_track_w0_metrics=true
    --set dual_mask_vis=false
    --set experiment_tracker=none
)

run_case() {
    local name="$1"
    local dual_mask_enabled="$2"
    local run_log="$3"

    echo "============================================================"
    echo "Starting $name"
    echo "Changed between C/D: dual_mask_enabled=$dual_mask_enabled"
    echo "Shared: LoRI-S dense $STAGE_EPOCHS epochs + global B Top-10% + reset + sparse $STAGE_EPOCHS epochs"
    echo "Protocol: ImageNet-R T10 split, run $TASKS tasks"
    echo "Log: $run_log"
    echo "============================================================"
    if "$PYTHON_BIN" main.py "${COMMON_ARGS[@]}" \
        --set prefix="$name" \
        --set dual_mask_enabled="$dual_mask_enabled" \
        2>&1 | tee "$run_log"
    then
        echo "PASS $name"
    else
        echo "FAIL $name"
        FAILED=1
    fi
}

verify_count() {
    local expected="$1"
    local pattern="$2"
    local run_log="$3"
    local actual
    actual=$(grep -c "$pattern" "$run_log" || true)
    if [[ "$actual" -ne "$expected" ]]; then
        echo "VERIFY FAIL: expected $expected matches for '$pattern', got $actual in $run_log"
        FAILED=1
    fi
}

verify_log() {
    local run_log="$1"
    local expect_w0_metrics="$2"

    grep -q "Running $TASKS/10 tasks" "$run_log" || {
        echo "VERIFY FAIL: task limit was not confirmed in $run_log"
        FAILED=1
    }
    grep -q "Average Accuracy" "$run_log" || {
        echo "VERIFY FAIL: final metrics missing from $run_log"
        FAILED=1
    }
    if grep -E "Traceback|FAILED \(|FAIL imgr10" "$run_log" >/dev/null; then
        echo "VERIFY FAIL: runtime failure marker found in $run_log"
        FAILED=1
    fi
    if grep "\[LoRA-Stage\] Parameters to be updated" "$run_log" | grep "A.weight" >/dev/null; then
        echo "VERIFY FAIL: LoRI-S unexpectedly trained A in $run_log"
        FAILED=1
    fi

    verify_count "$TASKS" "LoRI-S dense calibration" "$run_log"
    verify_count "$TASKS" "LoRI-S global mask:.*(0.1000)" "$run_log"
    verify_count "$TASKS" "LoRI-S sparse retraining" "$run_log"
    verify_count "$TASKS" "Extrace features for merging shared component" "$run_log"
    if [[ "$expect_w0_metrics" == "true" ]]; then
        verify_count "$TASKS" "W_pre train-only competence" "$run_log"
    else
        verify_count 0 "W_pre train-only competence" "$run_log"
    fi
}

run_case "imgr10_lori_s_C_no_dualmask_${RUN_MODE}" false "$C_LOG"
run_case "imgr10_lori_s_D_dualmask_${RUN_MODE}" true "$D_LOG"

if [[ -f "$C_LOG" ]]; then
    verify_log "$C_LOG" false
fi
if [[ -f "$D_LOG" ]]; then
    verify_log "$D_LOG" true
fi

echo "============================================================"
echo "Finished C/D LoRI-S $RUN_MODE; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit "$FAILED"
