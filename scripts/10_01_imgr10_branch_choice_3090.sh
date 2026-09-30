#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
python scripts/run_branch_choice_night.py 3090 "$@"
