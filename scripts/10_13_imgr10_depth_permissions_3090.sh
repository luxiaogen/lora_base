#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${LODA_PYTHON:-python}" scripts/run_depth_permissions.py --machine 3090 --hours 10 "$@"
