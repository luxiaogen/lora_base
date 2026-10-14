#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LODA_PYTHON="${LODA_PYTHON:-python}"
exec "$LODA_PYTHON" "$SCRIPT_DIR/run_ca_drift.py" "$@"
