#!/usr/bin/env bash
# Expanded from scripts/sweeps/imgr10_pair_separation_5090.json.
set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."
MODE="${1:---run}"
case "$MODE" in
    --run|--smoke|--full|--dry-run) ;;
    *) echo "Usage: bash $0 [--smoke|--full|--dry-run]" >&2; exit 2 ;;
esac
TIMESTAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="logs/shell_logs/imgr10_pair_separation_5090/${TIMESTAMP}_${MODE#--}"
FAILED=0

# Smoke checks execution only, never accuracy; all full candidates run in order.
if [[ "$MODE" == "--run" ]]; then
    bash "$SCRIPT" --smoke || exit $?
fi
if [[ "$MODE" != "--dry-run" ]]; then
    mkdir -p "$LOG_DIR"
fi
echo "Code revision: $(git rev-parse HEAD 2>/dev/null || true)"
echo "Mode: $MODE; full results and one-epoch smoke results are separate."

run_experiment() {
    local setting prefix=""
    for setting in "$@"; do
        if [[ "$setting" == prefix=* ]]; then prefix="${setting#prefix=}"; fi
    done
    local -a command=(python main.py "$@" --set "prefix=${prefix}_${TIMESTAMP}_${MODE#--}")
    if [[ "$MODE" == "--smoke" ]]; then
        command+=(--set max_tasks=2 --set init_epoch=1 --set epochs=1
                  --set ca_epochs=1 --set wandb_mode=offline --set save_task_weights=false)
    fi
    printf 'Command: '
    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ "$MODE" != "--dry-run" ]]; then
        "${command[@]}"
    fi
}
record_log() {
    if [[ "$MODE" == "--dry-run" ]]; then cat; else tee "$1"; fi
}

echo "============================================================"
echo "Starting imgr10_pair_separation_5090_p_w005_t3_seed1993"
echo "Changed: Current-task pair margin .1; extra gradient to p; weight 0.05; Task0 unchanged; Task0-2"
echo "Log: $LOG_DIR/imgr10_pair_separation_5090_p_w005_t3_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    run_experiment --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_pair_separation_5090_p_w005_t3_seed1993 \
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
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_granularity=layer \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=true \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_conflict_energy_adaptive=true \
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
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_competence_metric=accuracy \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set max_tasks=3 \
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
        --set ca_real_new_features=false \
        --set dual_mask_conflict_exact_topk=false \
        --set head_balance_weight=0 \
        --set ca_boundary_shadow=false \
        --set history_audit=false \
        --set plora_train_a=false \
        --set incremental_holdout=false \
        --set incremental_holdout_mod=5 \
        --set old_model_distill_temperature=2 \
        --set old_model_distill_weight=0 \
        --set head_start_init=random \
        --set head_start_epochs=0 \
        --set head_start_stage=pre \
        --set slora_lr_multiplier=1 \
        --set wandb_group=imgr10_pair_separation_5090 \
        --set save_task_weights=false \
        --set ca_cross_task_margin_weight=0 \
        --set ca_cov_shrinkage=0 \
        --set dual_mask_anchor_reg_weight=5 \
        --set ca_stats_transport=false \
        --set ca_stats_transport_mean_only=false \
        --set plora_a_init_batches=4 \
        --set plora_a_init_mode=off \
        --set pair_separation_weight=0.05 \
        --set pair_separation_scope=p \
        --set pair_separation_margin=0.1 \
        2>&1 | record_log "$LOG_DIR/imgr10_pair_separation_5090_p_w005_t3_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_pair_separation_5090_p_w005_t3_seed1993"
else
    echo "FAIL imgr10_pair_separation_5090_p_w005_t3_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_pair_separation_5090_sp_w0025_t3_seed1993"
echo "Changed: Current-task pair margin .1; extra gradient to sp; weight 0.025; Task0 unchanged; Task0-2"
echo "Log: $LOG_DIR/imgr10_pair_separation_5090_sp_w0025_t3_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    run_experiment --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_pair_separation_5090_sp_w0025_t3_seed1993 \
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
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_granularity=layer \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=true \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_conflict_energy_adaptive=true \
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
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_competence_metric=accuracy \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set max_tasks=3 \
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
        --set ca_real_new_features=false \
        --set dual_mask_conflict_exact_topk=false \
        --set head_balance_weight=0 \
        --set ca_boundary_shadow=false \
        --set history_audit=false \
        --set plora_train_a=false \
        --set incremental_holdout=false \
        --set incremental_holdout_mod=5 \
        --set old_model_distill_temperature=2 \
        --set old_model_distill_weight=0 \
        --set head_start_init=random \
        --set head_start_epochs=0 \
        --set head_start_stage=pre \
        --set slora_lr_multiplier=1 \
        --set wandb_group=imgr10_pair_separation_5090 \
        --set save_task_weights=false \
        --set ca_cross_task_margin_weight=0 \
        --set ca_cov_shrinkage=0 \
        --set dual_mask_anchor_reg_weight=5 \
        --set ca_stats_transport=false \
        --set ca_stats_transport_mean_only=false \
        --set plora_a_init_batches=4 \
        --set plora_a_init_mode=off \
        --set pair_separation_weight=0.025 \
        --set pair_separation_scope=sp \
        --set pair_separation_margin=0.1 \
        2>&1 | record_log "$LOG_DIR/imgr10_pair_separation_5090_sp_w0025_t3_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_pair_separation_5090_sp_w0025_t3_seed1993"
else
    echo "FAIL imgr10_pair_separation_5090_sp_w0025_t3_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_pair_separation_5090_sp_w005_t10_seed1993"
echo "Changed: Current-task pair margin .1; extra gradient to sp; weight 0.05; Task0 unchanged; Task0-9"
echo "Log: $LOG_DIR/imgr10_pair_separation_5090_sp_w005_t10_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    run_experiment --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_pair_separation_5090_sp_w005_t10_seed1993 \
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
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_granularity=layer \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=true \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_conflict_energy_adaptive=true \
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
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_anchor_reg_enabled=true \
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
        --set ca_real_new_features=false \
        --set dual_mask_conflict_exact_topk=false \
        --set head_balance_weight=0 \
        --set ca_boundary_shadow=false \
        --set history_audit=false \
        --set plora_train_a=false \
        --set incremental_holdout=false \
        --set incremental_holdout_mod=5 \
        --set old_model_distill_temperature=2 \
        --set old_model_distill_weight=0 \
        --set head_start_init=random \
        --set head_start_epochs=0 \
        --set head_start_stage=pre \
        --set slora_lr_multiplier=1 \
        --set wandb_group=imgr10_pair_separation_5090 \
        --set save_task_weights=false \
        --set ca_cross_task_margin_weight=0 \
        --set ca_cov_shrinkage=0 \
        --set dual_mask_anchor_reg_weight=5 \
        --set ca_stats_transport=false \
        --set ca_stats_transport_mean_only=false \
        --set plora_a_init_batches=4 \
        --set plora_a_init_mode=off \
        --set pair_separation_weight=0.05 \
        --set pair_separation_scope=sp \
        --set pair_separation_margin=0.1 \
        2>&1 | record_log "$LOG_DIR/imgr10_pair_separation_5090_sp_w005_t10_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_pair_separation_5090_sp_w005_t10_seed1993"
else
    echo "FAIL imgr10_pair_separation_5090_sp_w005_t10_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Finished 3 runs for imgr10_pair_separation_5090; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
