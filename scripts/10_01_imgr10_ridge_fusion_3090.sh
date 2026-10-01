#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:---run}"
exec python3 scripts/run_ridge_fusion.py --machine 3090 --mode="${mode#--}"
