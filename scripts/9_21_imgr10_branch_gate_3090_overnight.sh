#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

echo "Code revision: $(git rev-parse --short HEAD)"
echo "Dataset path is read only from exps/dlora/imgr10.json."
echo "Run order: A1993 -> B1993 -> C1993 -> D1993 -> B1996 -> B1997"

python -m unittest test.test_branch_gate_sweep || exit 1
python -m unittest \
    test.test_dual_mask_core.MaskSelectionTests.test_private_none_mode_leaves_only_plastic_mask \
    test.test_dual_mask_core.LoRALifecycleTests.test_balanced_strength_is_uniform_inside_protect_mask \
    test.test_dual_mask_core.LoRALifecycleTests.test_unprotected_positions_are_not_suppressed \
    || exit 1

echo "Running one-epoch B route smoke; this is not a performance result."
bash scripts/9_21_imgr10_p_conflict_off_smoke_3090.sh || exit 1

echo "Running four complete seed-1993 mechanism runs."
bash scripts/9_21_imgr10_branch_gate_seed1993_3090.sh || exit 1

echo "Running the pre-registered B follow-up on seeds 1996 and 1997."
bash scripts/9_21_imgr10_p_conflict_off_followup_3090.sh
