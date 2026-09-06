#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/shell_logs/protect_strength_mapping"
WANDB_GROUP="protect_strength_mapping_seed1993_${TIMESTAMP}"

mkdir -p "$LOG_DIR"

echo "========================================="
echo "  Starting CUB legacy-linear strength"
echo "  alpha = 0.30 + 0.60 * C_control"
echo "  Log: ${LOG_DIR}/cub10_legacy_linear_seed1993_${TIMESTAMP}.log"
echo "========================================="
python main.py \
  --config ideas/dual_mask_branch/configs/cub10.json \
  --set 'seed=[1993]' \
  --set prefix=cub10_strength_legacy_linear \
  --set dual_mask_protect_strength_mode=legacy_linear \
  --set "wandb_group=${WANDB_GROUP}" \
  --set wandb_tags=protect_strength_mapping,cub10,legacy_linear,seed1993 \
  2>&1 | tee "${LOG_DIR}/cub10_legacy_linear_seed1993_${TIMESTAMP}.log"

echo "========================================="
echo "  Starting CUB competence-direct strength"
echo "  alpha = C_control"
echo "  Log: ${LOG_DIR}/cub10_competence_seed1993_${TIMESTAMP}.log"
echo "========================================="
python main.py \
  --config ideas/dual_mask_branch/configs/cub10.json \
  --set 'seed=[1993]' \
  --set prefix=cub10_strength_competence \
  --set dual_mask_protect_strength_mode=competence \
  --set "wandb_group=${WANDB_GROUP}" \
  --set wandb_tags=protect_strength_mapping,cub10,competence,seed1993 \
  2>&1 | tee "${LOG_DIR}/cub10_competence_seed1993_${TIMESTAMP}.log"

echo "========================================="
echo "  Starting ImageNet-R legacy-linear strength"
echo "  alpha = 0.30 + 0.60 * C_control"
echo "  Log: ${LOG_DIR}/imgr10_legacy_linear_seed1993_${TIMESTAMP}.log"
echo "========================================="
python main.py \
  --config ideas/dual_mask_branch/configs/imgr10.json \
  --set 'seed=[1993]' \
  --set prefix=imgr10_strength_legacy_linear \
  --set dual_mask_protect_strength_mode=legacy_linear \
  --set "wandb_group=${WANDB_GROUP}" \
  --set wandb_tags=protect_strength_mapping,imgr10,legacy_linear,seed1993 \
  2>&1 | tee "${LOG_DIR}/imgr10_legacy_linear_seed1993_${TIMESTAMP}.log"

echo "========================================="
echo "  Starting ImageNet-R competence-direct strength"
echo "  alpha = C_control"
echo "  Log: ${LOG_DIR}/imgr10_competence_seed1993_${TIMESTAMP}.log"
echo "========================================="
python main.py \
  --config ideas/dual_mask_branch/configs/imgr10.json \
  --set 'seed=[1993]' \
  --set prefix=imgr10_strength_competence \
  --set dual_mask_protect_strength_mode=competence \
  --set "wandb_group=${WANDB_GROUP}" \
  --set wandb_tags=protect_strength_mapping,imgr10,competence,seed1993 \
  2>&1 | tee "${LOG_DIR}/imgr10_competence_seed1993_${TIMESTAMP}.log"

echo "========================================="
echo "  Starting ImageNet-A legacy-linear strength"
echo "  alpha = 0.30 + 0.60 * C_control"
echo "  Log: ${LOG_DIR}/imga10_legacy_linear_seed1993_${TIMESTAMP}.log"
echo "========================================="
python main.py \
  --config ideas/dual_mask_branch/configs/imga10.json \
  --set 'seed=[1993]' \
  --set prefix=imga10_strength_legacy_linear \
  --set dual_mask_protect_strength_mode=legacy_linear \
  --set "wandb_group=${WANDB_GROUP}" \
  --set wandb_tags=protect_strength_mapping,imga10,legacy_linear,seed1993 \
  2>&1 | tee "${LOG_DIR}/imga10_legacy_linear_seed1993_${TIMESTAMP}.log"

echo "========================================="
echo "  Starting ImageNet-A competence-direct strength"
echo "  alpha = C_control"
echo "  Log: ${LOG_DIR}/imga10_competence_seed1993_${TIMESTAMP}.log"
echo "========================================="
python main.py \
  --config ideas/dual_mask_branch/configs/imga10.json \
  --set 'seed=[1993]' \
  --set prefix=imga10_strength_competence \
  --set dual_mask_protect_strength_mode=competence \
  --set "wandb_group=${WANDB_GROUP}" \
  --set wandb_tags=protect_strength_mapping,imga10,competence,seed1993 \
  2>&1 | tee "${LOG_DIR}/imga10_competence_seed1993_${TIMESTAMP}.log"

echo "========================================="
echo "  Finished 6 protect-strength experiments"
echo "  Logs: ${LOG_DIR}"
echo "  W&B group: ${WANDB_GROUP}"
echo "========================================="