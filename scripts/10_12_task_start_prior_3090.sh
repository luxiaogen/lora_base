#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${LODA_PYTHON:-python}" scripts/run_task_start_prior.py --machine 3090 --hours 8 "$@"
