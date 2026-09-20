#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
[[ -f main.py ]] || exit 1
git rev-parse HEAD
python -m unittest test.test_p_input_subspace test.test_p_region_training -q || exit 1
[[ "${SMOKE_ONLY:-0}" == 1 ]] && exit 0
LOG_DIR=logs/shell_logs/p_conflict_baseline_3090
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_baseline_seed1993"
echo "Changed: Paired no-loss baseline on the same 3090; input statistics do not affect training"
echo "Log: $LOG_DIR/imgr10_baseline_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_baseline_seed1993 \
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
        --set wandb_group=p_conflict_norm_paired_screen \
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
        --set max_tasks=3 \
        --set dual_mask_p_input_rank=32 \
        --set dual_mask_p_input_weight=1 \
        --set dual_mask_p_conflict_group_diagnostic=false \
        --set dual_mask_p_input_subspace=baseline \
        2>&1 | tee "$LOG_DIR/imgr10_baseline_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_baseline_seed1993"
else
    echo "FAIL imgr10_baseline_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Finished 1 runs for p_conflict_baseline_3090; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
