#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# 两组先全部短测；正式按alpha=0.25、0.75顺序运行，不重跑0.5。
exec "${LODA_PYTHON:-python}" scripts/run_tail_update.py --machine 3090 \
  --spec scripts/sweeps/imgr10_ncm_protection_sensitivity_3090.json --hours 3 "$@"
