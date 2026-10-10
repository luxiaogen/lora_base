#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# 单组：覆盖0.90、rank64/64固定；增量保护用C_new，冲突强度用R_old。
exec "${LODA_PYTHON:-python}" scripts/run_tail_update.py --machine 3090 \
  --spec scripts/sweeps/imgr10_ncm_direct_strengths_3090.json --hours 3 "$@"
