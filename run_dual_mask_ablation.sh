#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs/shell_logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

run_no_conflict() {
    local name="$1"
    local config="$2"
    python main.py \
        --config "$config" \
        --set prefix=idea3_wpre_adaptive_no_conflict \
        --set dual_mask_conflict_ratio=0 \
        --set dual_mask_conflict_strength=0 \
        --set dual_mask_conflict_reg_enabled=false \
        2>&1 | tee "logs/shell_logs/${name}_no_conflict_${TIMESTAMP}.log"
}

run_no_conflict cub10 ideas/dual_mask_branch/configs/cub10.json
run_no_conflict cifar100 ideas/dual_mask_branch/configs/cifar10.json
run_no_conflict imgr10 ideas/dual_mask_branch/configs/imgr10.json