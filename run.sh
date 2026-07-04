#!/bin/bash

# 创建日志目录（如果不存在）
mkdir -p logs/shell_logs

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================="
echo "  Starting CIFAR-100 T10 Experiment"
echo "  Log: logs/shell_logs/cifar100_${TIMESTAMP}.log"
echo "========================================="
python main.py --config exps/dlora/cifar10.json 2>&1 | tee logs/shell_logs/cifar100_${TIMESTAMP}.log

echo "========================================="
echo "  Starting ImageNet-R T10 Experiment"
echo "  Log: logs/shell_logs/imgr10_${TIMESTAMP}.log"
echo "========================================="
python main.py --config exps/dlora/imgr20.json 2>&1 | tee logs/shell_logs/imgr10_${TIMESTAMP}.log

echo "========================================="
echo "  Starting Cub T10 Experiment"
echo "  Log: logs/shell_logs/cub10_${TIMESTAMP}.log"
echo "========================================="
python main.py --config exps/dlora/cub10.json 2>&1 | tee logs/shell_logs/cub10_${TIMESTAMP}.log

echo "========================================="
echo "  All experiments finished successfully!"
echo "========================================="
