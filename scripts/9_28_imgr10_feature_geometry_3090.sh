#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Default is the exact anchor2.5 saved baseline from the3090 log, not a latest-file glob.
checkpoint="${1:-logs/ImageNet_R/10_tasks/ca/imgr10_anchor2p5_save_t10_3090_anchor2p5_20260928_133705_194333_/checkpoints/20260928_133709_314001/task_09.pt}"
out="${2:-logs/feature_geometry/anchor2p5_3090_$(date +%Y%m%d_%H%M%S)}"
echo "Checkpoint: $checkpoint"
echo "Output: $out"
# Import plotting dependencies before spending GPU time. No training test gate.
python3 scripts/plot_feature_geometry.py --help >/dev/null
python3 scripts/extract_feature_geometry.py --checkpoint "$checkpoint" --out "$out"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/plot_feature_geometry.py --cache "$out"
