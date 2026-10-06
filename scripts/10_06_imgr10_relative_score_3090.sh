#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u scripts/run_relative_score.py --machine 3090 "$@"
