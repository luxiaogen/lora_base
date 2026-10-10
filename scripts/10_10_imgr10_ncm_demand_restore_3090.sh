#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# 单组D_t对照：保护C_new*(1-D_t)，原冲突公式；覆盖0.90与rank64/64固定。
exec "${LODA_PYTHON:-python}" scripts/run_tail_update.py --machine 3090 \
  --spec scripts/sweeps/imgr10_ncm_demand_restore_3090.json --hours 3 "$@"
