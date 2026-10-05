#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u scripts/run_compact_structure_night.py --machine 5090 "$@"
