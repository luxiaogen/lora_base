#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs/shell_logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [ "$#" -eq 0 ]; then
    set -- exps/dlora/cub10.json
fi

for config in "$@"; do
    name=$(basename "$config" .json)
    echo "Starting ${name} with W_pre directional gradient hook"
    python main.py \
        --config "$config" \
        --set prefix=idea3_wpre_adaptive_directional \
        --set dual_mask_directional_conflict=true \
        --set dual_mask_directional_strength=0.2 \
        --set dual_mask_directional_temperature=0.1 \
        --set dual_mask_directional_replace_magnitude=false \
        2>&1 | tee "logs/shell_logs/${name}_directional_${TIMESTAMP}.log"
done
