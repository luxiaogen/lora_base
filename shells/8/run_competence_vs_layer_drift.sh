#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/shell_logs/competence_vs_layer_drift"
WANDB_GROUP="competence_vs_layer_drift_seed1993_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "  CUB10 competence baseline"
echo "  seed=1993, Energy-50%, alpha=competence"
echo "========================================="
python main.py \
  --config exps/dlora/cub10.json \
  --set 'seed=[1993]' \
  --set prefix=cub10_competence_seed1993 \
  --set dual_mask_protect_strength_mode=competence \
  --set dual_mask_competence_adaptive=true \
  --set dual_mask_competence_all_seen=false \
  --set dual_mask_competence_metric=accuracy \
  --set dual_mask_plasticity_adaptive=true \
  --set dual_mask_conflict_energy_adaptive=true \
  --set dual_mask_conflict_ratio=0.1 \
  --set dual_mask_conflict_strength=0.5 \
  --set dual_mask_conflict_reg_enabled=true \
  --set dual_mask_conflict_old_overlap_adaptive=false \
  --set dual_mask_private_conflict_mode=global \
  --set dual_mask_task0_gate_mode=protect_only \
  --set dual_mask_track_w0_metrics=true \
  --set dual_mask_vis=false \
  --set experiment_tracker=wandb \
  --set wandb_project=LoDA_ICML2026 \
  --set wandb_mode=online \
  --set "wandb_group=${WANDB_GROUP}" \
  --set wandb_tags=competence,energy50,cub10,seed1993 \
  2>&1 | tee "${LOG_DIR}/cub10_competence_seed1993_${TIMESTAMP}.log"

echo "========================================="
echo "  CUB10 layer-drift ablation"
echo "  seed=1993, Energy-50%, alpha=layer_drift"
echo "========================================="
python main.py \
  --config exps/dlora/cub10.json \
  --set 'seed=[1993]' \
  --set prefix=cub10_layer_drift_seed1993 \
  --set dual_mask_protect_strength_mode=layer_drift \
  --set dual_mask_competence_adaptive=true \
  --set dual_mask_competence_all_seen=false \
  --set dual_mask_competence_metric=accuracy \
  --set dual_mask_plasticity_adaptive=true \
  --set dual_mask_conflict_energy_adaptive=true \
  --set dual_mask_conflict_ratio=0.1 \
  --set dual_mask_conflict_strength=0.5 \
  --set dual_mask_conflict_reg_enabled=true \
  --set dual_mask_conflict_old_overlap_adaptive=false \
  --set dual_mask_private_conflict_mode=global \
  --set dual_mask_task0_gate_mode=protect_only \
  --set dual_mask_track_w0_metrics=true \
  --set dual_mask_vis=false \
  --set experiment_tracker=wandb \
  --set wandb_project=LoDA_ICML2026 \
  --set wandb_mode=online \
  --set "wandb_group=${WANDB_GROUP}" \
  --set wandb_tags=layer_drift,energy50,cub10,seed1993 \
  2>&1 | tee "${LOG_DIR}/cub10_layer_drift_seed1993_${TIMESTAMP}.log"

echo "========================================="
echo "  ImageNet-R10 competence baseline"
echo "  seed=1993, Energy-50%, alpha=competence"
echo "========================================="
python main.py \
  --config exps/dlora/imgr10.json \
  --set 'seed=[1993]' \
  --set prefix=imgr10_competence_seed1993 \
  --set dual_mask_protect_strength_mode=competence \
  --set dual_mask_competence_adaptive=true \
  --set dual_mask_competence_all_seen=false \
  --set dual_mask_competence_metric=accuracy \
  --set dual_mask_plasticity_adaptive=true \
  --set dual_mask_conflict_energy_adaptive=true \
  --set dual_mask_conflict_ratio=0.1 \
  --set dual_mask_conflict_strength=0.5 \
  --set dual_mask_conflict_reg_enabled=true \
  --set dual_mask_conflict_old_overlap_adaptive=false \
  --set dual_mask_private_conflict_mode=global \
  --set dual_mask_task0_gate_mode=protect_only \
  --set dual_mask_track_w0_metrics=true \
  --set dual_mask_vis=false \
  --set experiment_tracker=wandb \
  --set wandb_project=LoDA_ICML2026 \
  --set wandb_mode=online \
  --set "wandb_group=${WANDB_GROUP}" \
  --set wandb_tags=competence,energy50,imgr10,seed1993 \
  2>&1 | tee "${LOG_DIR}/imgr10_competence_seed1993_${TIMESTAMP}.log"

echo "========================================="
echo "  ImageNet-R10 layer-drift ablation"
echo "  seed=1993, Energy-50%, alpha=layer_drift"
echo "========================================="
python main.py \
  --config exps/dlora/imgr10.json \
  --set 'seed=[1993]' \
  --set prefix=imgr10_layer_drift_seed1993 \
  --set dual_mask_protect_strength_mode=layer_drift \
  --set dual_mask_competence_adaptive=true \
  --set dual_mask_competence_all_seen=false \
  --set dual_mask_competence_metric=accuracy \
  --set dual_mask_plasticity_adaptive=true \
  --set dual_mask_conflict_energy_adaptive=true \
  --set dual_mask_conflict_ratio=0.1 \
  --set dual_mask_conflict_strength=0.5 \
  --set dual_mask_conflict_reg_enabled=true \
  --set dual_mask_conflict_old_overlap_adaptive=false \
  --set dual_mask_private_conflict_mode=global \
  --set dual_mask_task0_gate_mode=protect_only \
  --set dual_mask_track_w0_metrics=true \
  --set dual_mask_vis=false \
  --set experiment_tracker=wandb \
  --set wandb_project=LoDA_ICML2026 \
  --set wandb_mode=online \
  --set "wandb_group=${WANDB_GROUP}" \
  --set wandb_tags=layer_drift,energy50,imgr10,seed1993 \
  2>&1 | tee "${LOG_DIR}/imgr10_layer_drift_seed1993_${TIMESTAMP}.log"

echo "========================================="
echo "  ImageNet-A competence baseline"
echo "  seed=1993, Energy-50%, alpha=competence"
echo "========================================="
python main.py \
  --config exps/dlora/imga10.json \
  --set 'seed=[1993]' \
  --set prefix=imga10_competence_seed1993 \
  --set dual_mask_protect_strength_mode=competence \
  --set dual_mask_competence_adaptive=true \
  --set dual_mask_competence_all_seen=false \
  --set dual_mask_competence_metric=accuracy \
  --set dual_mask_plasticity_adaptive=true \
  --set dual_mask_conflict_energy_adaptive=true \
  --set dual_mask_conflict_ratio=0.1 \
  --set dual_mask_conflict_strength=0.5 \
  --set dual_mask_conflict_reg_enabled=true \
  --set dual_mask_conflict_old_overlap_adaptive=false \
  --set dual_mask_private_conflict_mode=global \
  --set dual_mask_task0_gate_mode=protect_only \
  --set dual_mask_track_w0_metrics=true \
  --set dual_mask_vis=false \
  --set experiment_tracker=wandb \
  --set wandb_project=LoDA_ICML2026 \
  --set wandb_mode=online \
  --set "wandb_group=${WANDB_GROUP}" \
  --set wandb_tags=competence,energy50,imga10,seed1993 \
  2>&1 | tee "${LOG_DIR}/imga10_competence_seed1993_${TIMESTAMP}.log"

echo "========================================="
echo "  ImageNet-A layer-drift ablation"
echo "  seed=1993, Energy-50%, alpha=layer_drift"
echo "========================================="
python main.py \
  --config exps/dlora/imga10.json \
  --set 'seed=[1993]' \
  --set prefix=imga10_layer_drift_seed1993 \
  --set dual_mask_protect_strength_mode=layer_drift \
  --set dual_mask_competence_adaptive=true \
  --set dual_mask_competence_all_seen=false \
  --set dual_mask_competence_metric=accuracy \
  --set dual_mask_plasticity_adaptive=true \
  --set dual_mask_conflict_energy_adaptive=true \
  --set dual_mask_conflict_ratio=0.1 \
  --set dual_mask_conflict_strength=0.5 \
  --set dual_mask_conflict_reg_enabled=true \
  --set dual_mask_conflict_old_overlap_adaptive=false \
  --set dual_mask_private_conflict_mode=global \
  --set dual_mask_task0_gate_mode=protect_only \
  --set dual_mask_track_w0_metrics=true \
  --set dual_mask_vis=false \
  --set experiment_tracker=wandb \
  --set wandb_project=LoDA_ICML2026 \
  --set wandb_mode=online \
  --set "wandb_group=${WANDB_GROUP}" \
  --set wandb_tags=layer_drift,energy50,imga10,seed1993 \
  2>&1 | tee "${LOG_DIR}/imga10_layer_drift_seed1993_${TIMESTAMP}.log"