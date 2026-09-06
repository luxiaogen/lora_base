#!/bin/bash
set -euo pipefail

# 创建日志目录（如果不存在）
mkdir -p logs/shell_logs

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================="
echo "  Starting Cub200 T10 Experiment"
#echo "  Log: logs/shell_logs/cifar100_${TIMESTAMP}.log"
echo "========================================="
#python main.py --config exps/dlora/cifar10.json 2>&1 | tee logs/shell_logs/cifar100_${TIMESTAMP}.log
python main.py \
    --config exps/dlora/cub10.json \
    --set 'seed=[1993]' \
    --set prefix=dual_mask_simplified_v1 \
    --set init_epoch=20 \
    --set epochs=20 \
    --set ca=true \
    --set dual_mask_vis=false \
    --set dual_mask_track_w0_metrics=true \
    --set dual_mask_threshold_calibration=false \
    --set dual_mask_lora_readapt=false \
    --set experiment_tracker=wandb \
    --set wandb_project=LoDA_ICML2026 \
    --set wandb_mode=online

echo "========================================="
echo "  Starting ImageNet-R T10 Experiment"
#echo "  Log: logs/shell_logs/imgr10_${TIMESTAMP}.log"
echo "========================================="
#python main.py --config exps/dlora/imgr10.json 2>&1 | tee logs/shell_logs/imgr10_${TIMESTAMP}.log

python main.py \
    --config exps/dlora/imgr10.json \
    --set 'seed=[1993]' \
    --set prefix=dual_mask_simplified_v1 \
    --set init_epoch=20 \
    --set epochs=20 \
    --set ca=true \
    --set dual_mask_vis=false \
    --set dual_mask_track_w0_metrics=true \
    --set dual_mask_threshold_calibration=false \
    --set dual_mask_lora_readapt=false \
    --set experiment_tracker=wandb \
    --set wandb_project=LoDA_ICML2026 \
    --set wandb_mode=online

#echo "========================================="
#echo "  Starting ImageNet-R T20 Experiment ca=true"
#echo "  Log: logs/shell_logs/imgr20_${TIMESTAMP}.log"
#echo "========================================="
#python main.py --config exps/dlora/imgr20.json 2>&1 | tee logs/shell_logs/imgr20_${TIMESTAMP}.log
#
#echo "========================================="
#echo "  Starting Cub T10 Experiment"
#echo "  Log: logs/shell_logs/cub10_${TIMESTAMP}.log"
#echo "========================================="
#python main.py --config exps/dlora/cub10.json 2>&1 | tee logs/shell_logs/cub10_${TIMESTAMP}.log

echo "========================================="
echo "  All experiments finished successfully!"
echo "========================================="
