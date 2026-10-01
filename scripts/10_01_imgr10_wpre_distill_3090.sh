#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/run_wpre_distill.py --machine 3090 --mode "${1:-run}"
