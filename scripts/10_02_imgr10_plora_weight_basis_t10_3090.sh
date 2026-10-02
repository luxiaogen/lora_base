#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# All three matching GPU smokes and Task0-2 runs have completed; no baseline rerun.
python3 scripts/run_plora_weight_basis.py --tasks 10 --mode t10 --only gradient weight_prior "$@"
