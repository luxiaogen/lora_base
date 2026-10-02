#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/run_p_direction_score.py --spec scripts/sweeps/imgr10_p_functional_score_3090.json "$@"
