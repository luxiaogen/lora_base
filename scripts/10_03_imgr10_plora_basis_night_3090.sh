#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/run_plora_weight_basis.py --spec scripts/sweeps/imgr10_plora_basis_night_3090.json --tasks 10 "$@"
