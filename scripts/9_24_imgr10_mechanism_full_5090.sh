#!/usr/bin/env bash
# Only for a machine that has NOT already run the three-score comparison.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
if [[ "${1:-}" != "--from-scratch" ]]; then
    echo "Three-score experiment already running/completed? Run 9_24_imgr10_fixed_rho_5090.sh (4 new runs)."
    echo "To explicitly run all 7: bash $0 --from-scratch"
    exit 2
fi
FAILED=0
bash "$SCRIPT_DIR/9_24_imgr10_conflict_score_t10_seed1993_5090.sh" || FAILED=1
bash "$SCRIPT_DIR/9_24_imgr10_fixed_rho_5090.sh" || FAILED=1
exit "$FAILED"
