#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/run_permission_release_night.py --machine 3090 "$@"
