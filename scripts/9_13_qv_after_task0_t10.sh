#!/usr/bin/env bash
set -euo pipefail

# From repository root. Both runs train Task0 with QKV from scratch.
# Task1-9 compare QKV against QV; no checkpoint is saved or resumed.
# lrun scripts/9_13_qv_after_task0_t10.sh ./logs/9_13_qv_after_task0_t10.log
MAX_TASKS=10 exec bash scripts/9_13_qv_after_task0.sh "$@"
