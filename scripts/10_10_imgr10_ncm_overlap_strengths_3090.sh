#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# 单组：保护强度C_new，冲突强度恢复min(0.5*(1+R_old),1)。
exec "${LODA_PYTHON:-python}" scripts/run_tail_update.py --machine 3090 \
  --spec scripts/sweeps/imgr10_ncm_overlap_strengths_3090.json --hours 3 "$@"
