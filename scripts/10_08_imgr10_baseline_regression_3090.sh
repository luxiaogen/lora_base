#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${PYTHON:-python}" scripts/run_baseline_suite.py --modes imgr10_seed1993 "$@"
