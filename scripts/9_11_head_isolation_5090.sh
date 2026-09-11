#!/usr/bin/env bash
set -euo pipefail

# Run from repository root: lrun scripts/9_11_head_isolation_5090.sh ./logs/9_11_head_isolation_5090.log
exec bash scripts/9_11_head_isolation_pair.sh "$@" 5090
