#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec bash "$REPO_ROOT/scripts/9_24_imgr10_mask_comparison_common.sh" 3090 "$@"
