#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# 所有配置先短测，seed1993固定顺序；十二小时后不再启动新组。
exec "${LODA_PYTHON:-python}" scripts/run_tail_update.py --machine 3090 \
  --spec scripts/sweeps/imgr10_protection_search_3090.json --hours 12 "$@"
