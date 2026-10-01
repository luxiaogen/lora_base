#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:---run}" in
    --run|--smoke|--t10|--dry-run|--cache-only) ;;
    *) echo "Usage: bash $0 [--run|--smoke|--t10|--dry-run|--cache-only] [--cache PATH]" >&2; exit 2 ;;
esac
mode="${1:---run}"
if (( $# > 0 )); then shift; fi
exec python3 scripts/run_expert_legal.py --machine 5090 --mode="$mode" "$@"
