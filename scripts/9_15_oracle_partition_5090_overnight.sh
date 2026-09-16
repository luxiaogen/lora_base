#!/usr/bin/env bash
set -uo pipefail

[[ -f main.py ]] || { echo 'Run from the repository root.' >&2; exit 2; }
python -m unittest test.test_predicted_onehot test.test_p_conflict_diagnostics || exit 1
echo "Code revision: $(git rev-parse --short HEAD)"
git status --short

LOG_DIR=logs/shell_logs/oracle_partition_5090_overnight
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

run_job() {
    local protocol=$1
    local config=$2
    local seed=$3
    local tasks=$4
    local classes_per_task=$5
    local name="${protocol}_oracle_partition_seed${seed}"
    local log_file="$LOG_DIR/${name}_${TIMESTAMP}.log"

    echo "============================================================"
    echo "Starting $name"
    echo "Protocol: QKV, ${tasks} tasks, ${classes_per_task} classes/task, seed ${seed}"
    echo "Diagnostic: P-conflict modes including low/high-confidence oracle partition"
    echo "Log: $log_file"
    echo "============================================================"
    if
        python main.py --config "$config" \
            --set "seed=[$seed]" \
            --set prefix="$name" \
            --set max_tasks="$tasks" \
            --set total_sessions="$tasks" \
            --set init_cls="$classes_per_task" \
            --set increment="$classes_per_task" \
            --set use_slora=true \
            --set use_plora=true \
            --set lora_A_init=kaiming \
            --set init_epoch=20 \
            --set epochs=20 \
            --set ca=true \
            --set ca_epochs=5 \
            --set rank=64 \
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
            --set dual_mask_anchor_reg_weight=10 \
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
            --set wandb_group=oracle_partition_5090_overnight \
            --set wandb_tags="$protocol,qkv,seed${seed},oracle_partition,diagnostic_only" \
            2>&1 | tee "$log_file"
    then
        echo "PASS $name"
    else
        echo "FAIL $name"
        FAILED=1
    fi
}

# T10: complete the three-seed set after the current seed-1993 run.
if [[ "${ORACLE_PARTITION_SKIP_T10:-0}" != 1 ]]; then
    run_job imgr10 exps/dlora/imgr10.json 1996 10 20 || FAILED=1
    run_job imgr10 exps/dlora/imgr10.json 1997 10 20 || FAILED=1
fi

# T20 reuses this machine's ImageNet-R JSON and changes only the task split.
run_job imgr20 exps/dlora/imgr10.json 1993 20 10 || FAILED=1

echo "============================================================"
echo "Finished scheduled runs for oracle_partition_5090_overnight; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
