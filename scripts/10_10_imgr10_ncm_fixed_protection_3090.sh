#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# 单组：保护强度固定0.5，冲突沿用0.5*(1+R_old)，增量阶段跳过D_t。
exec "${LODA_PYTHON:-python}" scripts/run_tail_update.py --machine 3090 \
  --spec scripts/sweeps/imgr10_ncm_fixed_protection_3090.json --hours 3 "$@"
