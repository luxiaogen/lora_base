#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec bash "$REPO_ROOT/scripts/9_23_imgr10_lori_night_common.sh" 5090 "$@"
