#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs/shell_logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

run_with_conflict() {
    local name="$1"
    local config="$2"

    echo "========================================="
    echo "Starting ${name} with conflict mask"
    echo "conflict_ratio=0.1, conflict_strength=0.5, conflict_reg=true"
    echo "========================================="

    python main.py \
        --config "$config" \
        --set prefix=idea3_wpre_adaptive_with_conflict \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=true \
        2>&1 | tee "logs/shell_logs/${name}_with_conflict_${TIMESTAMP}.log"
}

#run_with_conflict cub10 ideas/dual_mask_branch/configs/cub10.json
#run_with_conflict cifar100 ideas/dual_mask_branch/configs/cifar10.json
run_with_conflict imgr10 ideas/dual_mask_branch/configs/imgr10.json

echo "========================================="
echo "All with-conflict experiments finished successfully!"
echo "========================================="



##!/bin/bash
#set -euo pipefail
#
## 创建日志目录（如果不存在）
#mkdir -p logs/shell_logs
#
## 生成时间戳
#TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
###
#echo "========================================="
#echo "  [1/3]Starting Cub T10 Experiment ca=true"
#echo "  Log: logs/shell_logs/cub10_${TIMESTAMP}.log"
#echo "========================================="
#python main.py --config /home/shengqin/lys/baseline/LoDA_ICML2026/ideas/dual_mask_branch/configs/cub10.json 2>&1 | tee logs/shell_logs/cub10_${TIMESTAMP}.log
#
#echo "========================================="
#echo "  [2/3]Starting CIFAR-100 T10 Experiment ca=true"
#echo "  Log: logs/shell_logs/cifar100_${TIMESTAMP}.log"
#echo "========================================="
#python main.py --config /home/shengqin/lys/baseline/LoDA_ICML2026/ideas/dual_mask_branch/configs/cifar10.json 2>&1 | tee logs/shell_logs/cifar100_${TIMESTAMP}.log
#
#echo "========================================="
#echo "  [3/3]Starting ImageNet-R T10 Experiment ca=true"
#echo "  Log: logs/shell_logs/imgr10_${TIMESTAMP}.log"
#echo "========================================="
#python main.py --config /home/shengqin/lys/baseline/LoDA_ICML2026/ideas/dual_mask_branch/configs/imgr10.json 2>&1 | tee logs/shell_logs/imgr10_${TIMESTAMP}.log
#
#
#
#echo "========================================="
#echo "  All experiments finished successfully!"
#echo "========================================="
