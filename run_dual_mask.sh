#!/bin/bash

# 创建日志目录（如果不存在）
mkdir -p logs/shell_logs

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
##
#echo "========================================="
#echo "  [1/2]Starting Cub T10 Experiment"
echo "  Log: logs/shell_logs/cub10_${TIMESTAMP}.log"
echo "========================================="
python main.py --config /home/shengqin/lys/baseline/LoDA_ICML2026/ideas/dual_mask_branch/configs/cub10.json 2>&1 | tee logs/shell_logs/cub10_${TIMESTAMP}.log

#echo "========================================="
#echo "  [2/3]Starting CIFAR-100 T10 Experiment"
#echo "  Log: logs/shell_logs/cifar100_${TIMESTAMP}.log"
#echo "========================================="
#python main.py --config /home/shengqin/lys/baseline/LoDA_ICML2026/ideas/dual_mask_branch/configs/cifar10.json 2>&1 | tee logs/shell_logs/cifar100_${TIMESTAMP}.log

#echo "========================================="
#echo "  [2/2]Starting ImageNet-R T10 Experiment"
#echo "  Log: logs/shell_logs/imgr10_${TIMESTAMP}.log"
#echo "========================================="
#python main.py --config /home/shengqin/lys/baseline/LoDA_ICML2026/ideas/dual_mask_branch/configs/imgr10.json 2>&1 | tee logs/shell_logs/imgr10_${TIMESTAMP}.log



echo "========================================="
echo "  All experiments finished successfully!"
echo "========================================="
