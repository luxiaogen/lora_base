#!/usr/bin/env bash
set -euo pipefail

# Run from repository root: lrun scripts/9_11_head_isolation_3090.sh ./logs/9_11_head_isolation_3090.log
exec bash scripts/9_11_head_isolation_pair.sh "$@" 3090
