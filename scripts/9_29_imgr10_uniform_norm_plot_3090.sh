#!/usr/bin/env bash
# Run only after the separate CUDA smoke has succeeded.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_ID="$(date +%Y%m%d_%H%M%S)"
CANDIDATE="logs/9_29_imgr10_uniform_norm_full_${RUN_ID}.log"
BASELINE="logs/9_28_imgr10_anchor2p5_save_t10_3090.log"
bash scripts/9_29_imgr10_uniform_norm_3090.sh 2>&1 | tee "$CANDIDATE"
python scripts/plot_matched_conflict_norm.py \
    --baseline "$BASELINE" --candidate "$CANDIDATE" \
    --out "logs/figures/matched_conflict_norm_${RUN_ID}"
