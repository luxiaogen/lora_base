#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# 先短测两个端点，随后依次完整训练；P硬权限、冲突公式和Task0不变。
exec "${LODA_PYTHON:-python}" scripts/run_tail_update.py --machine 3090 \
  --spec scripts/sweeps/imgr10_ncm_protection_boundary_3090.json --hours 3 "$@"
