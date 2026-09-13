#!/usr/bin/env bash
set -euo pipefail

# From repository root. Optional TASK0_CHECKPOINT reuses this machine's Task0 run.
# lrun scripts/9_13_qv_after_task0_t10.sh ./logs/9_13_qv_after_task0_t10.log
MAX_TASKS=10 exec bash scripts/9_13_qv_after_task0.sh "$@"
