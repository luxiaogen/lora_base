#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${LODA_PYTHON:-python}" scripts/run_tail_update.py --machine 3090 --spec scripts/sweeps/tail_update_imga10_3090.json "$@"
