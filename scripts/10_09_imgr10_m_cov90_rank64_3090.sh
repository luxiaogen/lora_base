#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${LODA_PYTHON:-python}" scripts/run_tail_update.py --machine 3090 \
  --spec scripts/sweeps/imgr10_m_cov90_rank64_3090.json --modes M_seed1993 "$@"
