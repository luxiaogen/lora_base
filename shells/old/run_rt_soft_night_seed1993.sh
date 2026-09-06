#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CONFIG="ideas/dual_mask_branch/configs/cub10.json"
SEED=1993
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/shell_logs/rt_soft_night_seed1993/${TIMESTAMP}"
SUMMARY_FILE="${LOG_DIR}/summary.tsv"
START_TIME=$(date +%s)
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CASES=(
    baseline
    rt_only
    soft_only
    rt_soft_b1
    rt_soft_b4
)

mkdir -p "$LOG_DIR"
printf 'case\tstatus\texit_code\tduration_seconds\tlog\n' > "$SUMMARY_FILE"

if [[ "$DRY_RUN" != "1" ]]; then
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "Python executable not found: ${PYTHON_BIN}" >&2
        exit 1
    fi
    echo "Conda environment: ${CONDA_DEFAULT_ENV:-not activated}"
fi

run_experiment() {
    local case_name="$1"
    local task_relevance
    local soft_conflict
    local grad_batches
    local prefix
    local log_file
    local run_start
    local run_end
    local run_seconds
    local exit_code
    local status
    local -a cmd

    case "$case_name" in
        baseline)
            task_relevance=false
            soft_conflict=false
            grad_batches=1
            ;;
        rt_only)
            task_relevance=true
            soft_conflict=false
            grad_batches=1
            ;;
        soft_only)
            task_relevance=false
            soft_conflict=true
            grad_batches=1
            ;;
        rt_soft_b1)
            task_relevance=true
            soft_conflict=true
            grad_batches=1
            ;;
        rt_soft_b4)
            task_relevance=true
            soft_conflict=true
            grad_batches=4
            ;;
        *)
            echo "Unknown experiment case: ${case_name}" >&2
            return 2
            ;;
    esac

    prefix="night_cub10_s${SEED}_${case_name}_${TIMESTAMP}"
    log_file="${LOG_DIR}/${case_name}.log"
    cmd=(
        "$PYTHON_BIN" main.py
        --config "$CONFIG"
        --set "seed=[${SEED}]"
        --set "prefix=${prefix}"
        --set init_epoch=20
        --set epochs=20
        --set ca=true
        --set ca_epochs=10
        --set dual_mask_importance=svd
        --set dual_mask_vis=false
        --set dual_mask_track_w0_metrics=true
        --set dual_mask_metric_batches=4
        --set "dual_mask_task_relevance_enabled=${task_relevance}"
        --set dual_mask_task_coverage=0.8
        --set "dual_mask_soft_conflict_gate=${soft_conflict}"
        --set "dual_mask_grad_batches=${grad_batches}"
    )

    echo "============================================================"
    echo "Starting ${case_name}"
    echo "seed=${SEED}, Rt=${task_relevance}, soft_conflict=${soft_conflict}, grad_batches=${grad_batches}"
    echo "config=${CONFIG}"
    echo "log=${log_file}"
    printf 'command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    echo "============================================================"

    run_start=$(date +%s)
    if [[ "$DRY_RUN" == "1" ]]; then
        exit_code=0
        status="DRY_RUN"
    else
        set +e
        "${cmd[@]}" 2>&1 | tee "$log_file"
        exit_code=${PIPESTATUS[0]}
        set -e
        if [[ "$exit_code" -eq 0 ]]; then
            status="OK"
        else
            status="FAILED"
        fi
    fi
    run_end=$(date +%s)
    run_seconds=$((run_end - run_start))

    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$case_name" \
        "$status" \
        "$exit_code" \
        "$run_seconds" \
        "$log_file" \
        >> "$SUMMARY_FILE"

    printf 'Finished %s: status=%s, exit_code=%s, duration=%ds (%dh %dm %ds)\n' \
        "$case_name" \
        "$status" \
        "$exit_code" \
        "$run_seconds" \
        "$((run_seconds / 3600))" \
        "$(((run_seconds % 3600) / 60))" \
        "$((run_seconds % 60))"

    # A failed run is recorded but does not prevent the remaining night jobs.
    return 0
}

for case_name in "${CASES[@]}"; do
    run_experiment "$case_name"
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))
FAILED_COUNT=$(awk -F '\t' 'NR > 1 && $2 == "FAILED" {count++} END {print count+0}' "$SUMMARY_FILE")

echo "============================================================"
printf 'Night suite finished in %ds (%dh %dm %ds)\n' \
    "$TOTAL_SECONDS" \
    "$((TOTAL_SECONDS / 3600))" \
    "$(((TOTAL_SECONDS % 3600) / 60))" \
    "$((TOTAL_SECONDS % 60))"
echo "Failed runs: ${FAILED_COUNT}"
echo "Summary: ${SUMMARY_FILE}"
echo "Logs: ${LOG_DIR}"
echo "============================================================"

if [[ "$FAILED_COUNT" -ne 0 ]]; then
    exit 1
fi
