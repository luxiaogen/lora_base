#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/run_core_evidence_night.py --machine 5090 "$@"
