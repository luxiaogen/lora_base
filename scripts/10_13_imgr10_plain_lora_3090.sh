#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${LODA_PYTHON:-python}" scripts/run_plain_lora_control.py --machine 3090 --hours 2 "$@"
