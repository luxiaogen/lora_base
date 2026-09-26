#!/usr/bin/env bash
set -uo pipefail

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT_PATH")/.." || exit 1
if [[ "${1:-}" == "--dry-run" ]]; then
    sed -n '/^    python main.py/,/^        2>/p' "$SCRIPT_PATH"
    exit 0
fi
RUN_KIND=full
if [[ "${1:-}" == "--smoke" ]]; then
    RUN_KIND=smoke
    shift
    set -- "$@" --set max_tasks=2 --set init_epoch=1 --set epochs=1 --set ca_epochs=1 --set wandb_mode=offline
fi
echo "Code revision: $(git rev-parse --short HEAD)"
git status --short
LOG_DIR=logs/shell_logs/imgr10_fixed_coverage_5090
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")_${RUN_KIND}"
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_t10_coverage010_seed1993"
echo "Changed: Fixed conflict coverage 10%; beta=0.5"
echo "Log: $LOG_DIR/imgr10_t10_coverage010_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_t10_coverage010_seed1993_5090_${TIMESTAMP}" \
        --set dataset=ImageNet_R \
        --set memory_size=0 \
        --set memory_per_class=0 \
        --set fixed_memory=true \
        --set shuffle=true \
        --set init_cls=20 \
        --set increment=20 \
        --set model_name=dual_mask_branch \
        --set embd_dim=768 \
        --set num_heads=12 \
        --set total_sessions=10 \
        --set batch_size=48 \
        --set init_epoch=20 \
        --set optim=sgd \
        --set init_lr=0.02 \
        --set init_weight_decay=0 \
        --set epochs=20 \
        --set lrate=0.02 \
        --set weight_decay=0 \
        --set rank=64 \
        --set scale=20 \
        --set margin=0.1 \
        --set num_workers=8 \
        --set ca=true \
        --set ca_epochs=5 \
        --set ca_lrate=0.01 \
        --set logit_norm=0.1 \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_general_ratio=0.4 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_granularity=layer \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=true \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_competence_holdout_mod=5 \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_metric_batches=4 \
        --set dual_mask_vis=false \
        --set 'dual_mask_vis_layers=[0,5,11]' \
        --set 'dual_mask_vis_tasks=[0,1,9]' \
        --set dual_mask_vis_dir=visualizations/dual_mask_snapshots/imgr10 \
        --set dual_mask_vis_save_weight=false \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_conflict_old_overlap_adaptive=false \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=5 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_competence_metric=accuracy \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set max_tasks=10 \
        --set disable_fused_sdpa=true \
        --set dual_mask_applied_budget_log=true \
        --set task0_validation_enabled=false \
        --set dual_mask_conflict_score_mode=conflict \
        --set dual_mask_conflict_budget_multiplier=1 \
        --set dual_mask_s_conflict_enabled=true \
        --set dual_mask_p_conflict_enabled=true \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_private_rank=0 \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_update_overlap=false \
        --set wandb_group=imgr10_fixed_coverage_5090 \
        --set plora_lr_multiplier=1 \
        --set dual_mask_reg_grad_diagnostic=false \
        --set dual_mask_s_reg_enabled=true \
        --set dual_mask_p_reg_enabled=true \
        --set task0_rs_weight=0 \
        --set task0_margin=0.1 \
        --set stage_audit=true \
        --set p_step_direction=off \
        --set old_competition_weight=0 \
        --set old_competition_detach_old=false \
        --set dual_mask_conflict_exact_topk=true \
        --set ca_real_new_features=false \
        --set dual_mask_conflict_ratio=0.1 \
        "$@" \
        2>&1 | tee "$LOG_DIR/imgr10_t10_coverage010_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_t10_coverage010_seed1993"
else
    echo "FAIL imgr10_t10_coverage010_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_t10_coverage020_seed1993"
echo "Changed: Fixed conflict coverage 20%; beta=0.5"
echo "Log: $LOG_DIR/imgr10_t10_coverage020_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_t10_coverage020_seed1993_5090_${TIMESTAMP}" \
        --set dataset=ImageNet_R \
        --set memory_size=0 \
        --set memory_per_class=0 \
        --set fixed_memory=true \
        --set shuffle=true \
        --set init_cls=20 \
        --set increment=20 \
        --set model_name=dual_mask_branch \
        --set embd_dim=768 \
        --set num_heads=12 \
        --set total_sessions=10 \
        --set batch_size=48 \
        --set init_epoch=20 \
        --set optim=sgd \
        --set init_lr=0.02 \
        --set init_weight_decay=0 \
        --set epochs=20 \
        --set lrate=0.02 \
        --set weight_decay=0 \
        --set rank=64 \
        --set scale=20 \
        --set margin=0.1 \
        --set num_workers=8 \
        --set ca=true \
        --set ca_epochs=5 \
        --set ca_lrate=0.01 \
        --set logit_norm=0.1 \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_general_ratio=0.4 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_granularity=layer \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=true \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_competence_holdout_mod=5 \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_metric_batches=4 \
        --set dual_mask_vis=false \
        --set 'dual_mask_vis_layers=[0,5,11]' \
        --set 'dual_mask_vis_tasks=[0,1,9]' \
        --set dual_mask_vis_dir=visualizations/dual_mask_snapshots/imgr10 \
        --set dual_mask_vis_save_weight=false \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_conflict_old_overlap_adaptive=false \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=5 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_competence_metric=accuracy \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set max_tasks=10 \
        --set disable_fused_sdpa=true \
        --set dual_mask_applied_budget_log=true \
        --set task0_validation_enabled=false \
        --set dual_mask_conflict_score_mode=conflict \
        --set dual_mask_conflict_budget_multiplier=1 \
        --set dual_mask_s_conflict_enabled=true \
        --set dual_mask_p_conflict_enabled=true \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_private_rank=0 \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_update_overlap=false \
        --set wandb_group=imgr10_fixed_coverage_5090 \
        --set plora_lr_multiplier=1 \
        --set dual_mask_reg_grad_diagnostic=false \
        --set dual_mask_s_reg_enabled=true \
        --set dual_mask_p_reg_enabled=true \
        --set task0_rs_weight=0 \
        --set task0_margin=0.1 \
        --set stage_audit=true \
        --set p_step_direction=off \
        --set old_competition_weight=0 \
        --set old_competition_detach_old=false \
        --set dual_mask_conflict_exact_topk=true \
        --set ca_real_new_features=false \
        --set dual_mask_conflict_ratio=0.2 \
        "$@" \
        2>&1 | tee "$LOG_DIR/imgr10_t10_coverage020_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_t10_coverage020_seed1993"
else
    echo "FAIL imgr10_t10_coverage020_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_t10_coverage100_seed1993"
echo "Changed: Fixed conflict coverage 100%; beta=0.5"
echo "Log: $LOG_DIR/imgr10_t10_coverage100_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_t10_coverage100_seed1993_5090_${TIMESTAMP}" \
        --set dataset=ImageNet_R \
        --set memory_size=0 \
        --set memory_per_class=0 \
        --set fixed_memory=true \
        --set shuffle=true \
        --set init_cls=20 \
        --set increment=20 \
        --set model_name=dual_mask_branch \
        --set embd_dim=768 \
        --set num_heads=12 \
        --set total_sessions=10 \
        --set batch_size=48 \
        --set init_epoch=20 \
        --set optim=sgd \
        --set init_lr=0.02 \
        --set init_weight_decay=0 \
        --set epochs=20 \
        --set lrate=0.02 \
        --set weight_decay=0 \
        --set rank=64 \
        --set scale=20 \
        --set margin=0.1 \
        --set num_workers=8 \
        --set ca=true \
        --set ca_epochs=5 \
        --set ca_lrate=0.01 \
        --set logit_norm=0.1 \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set use_slora=true \
        --set use_plora=true \
        --set lora_A_init=kaiming \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_general_ratio=0.4 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_granularity=layer \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=true \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_competence_holdout_mod=5 \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_metric_batches=4 \
        --set dual_mask_vis=false \
        --set 'dual_mask_vis_layers=[0,5,11]' \
        --set 'dual_mask_vis_tasks=[0,1,9]' \
        --set dual_mask_vis_dir=visualizations/dual_mask_snapshots/imgr10 \
        --set dual_mask_vis_save_weight=false \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_conflict_old_overlap_adaptive=false \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=5 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_competence_metric=accuracy \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set max_tasks=10 \
        --set disable_fused_sdpa=true \
        --set dual_mask_applied_budget_log=true \
        --set task0_validation_enabled=false \
        --set dual_mask_conflict_score_mode=conflict \
        --set dual_mask_conflict_budget_multiplier=1 \
        --set dual_mask_s_conflict_enabled=true \
        --set dual_mask_p_conflict_enabled=true \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_private_rank=0 \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_update_overlap=false \
        --set wandb_group=imgr10_fixed_coverage_5090 \
        --set plora_lr_multiplier=1 \
        --set dual_mask_reg_grad_diagnostic=false \
        --set dual_mask_s_reg_enabled=true \
        --set dual_mask_p_reg_enabled=true \
        --set task0_rs_weight=0 \
        --set task0_margin=0.1 \
        --set stage_audit=true \
        --set p_step_direction=off \
        --set old_competition_weight=0 \
        --set old_competition_detach_old=false \
        --set dual_mask_conflict_exact_topk=true \
        --set ca_real_new_features=false \
        --set dual_mask_conflict_ratio=1 \
        "$@" \
        2>&1 | tee "$LOG_DIR/imgr10_t10_coverage100_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_t10_coverage100_seed1993"
else
    echo "FAIL imgr10_t10_coverage100_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Finished 3 runs for imgr10_fixed_coverage_5090; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
