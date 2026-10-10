#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# 只恢复覆盖、保护强度和 P 秩；先短测，再执行唯一一组完整 T10。
exec "${LODA_PYTHON:-python}" scripts/run_tail_update.py --machine 3090 \
  --spec scripts/sweeps/imgr10_ncm_controller_restore_3090.json --hours 3 "$@"
