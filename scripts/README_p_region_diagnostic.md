# P conflict region diagnostic, based on 102b65c

Both machines run `scripts/9_19_p_region_diagnostic.sh` from repository root.
One fresh ImageNet-R T10 seed1993 QKV training per machine, 20 epochs, CA5,
rank64, adaptive private rank, Task0 unmasked+anchor10, global suppress,
Energy50/floor10, protection regularizer .01, extra conflict regularizer off.
Each machine reads its own `exps/dlora/imgr10.json` data_path. No checkpoint,
no additional routing, no permanent merge filtering, no old-image training replay.

For CURRENT task only, capture the actual gamma-scaled, suppressed safe P delta
at merge, split into conflict C and nonconflict U. Per layer remove
`r = 0.5 * min(||C||_F, ||U||_F)` from either region by proportional scaling.
Thus neither region is amplified; zero norm on either side makes both no-op.
This controls removed-update Frobenius norm, not coordinate count or functional
effect. U is proportional shrinkage, NOT random coordinate selection.

At Task1/2/9: after merge, before statistics/CA, evaluate each layer and joint
removal on every fifth example of CURRENT train data in deterministic eval view.
This is an in-sample diagnostic, NOT an independent holdout or selector.
At Task1 through Task9: after normal CA evaluate baseline/joint C/joint U on
all-seen test data. Task0 has no P and is a no-op. Statistics and CA are NOT
recomputed for temporary interventions. These are local sensitivity results,
NOT three independently trained methods or proof of permanent-filter gains.

All weights restored by copy, RNG and module modes restored; later tasks always
train from original baseline. Only current task tensors retained; released after
diagnostic. Test labels only compute metrics, never choose a mask or gate.
`P-region norms` records actual per-layer budget; `P-region diagnostic` JSON
records task/source/layer, all/old/new accuracy, task accuracy, corrected/broken,
and local/cross-task margin change (candidate MINUS baseline, positive=improvement).
Cross-task margin uses ALL other tasks, not only latest-vs-old.
The normal final CNN summary is the unmodified baseline, not the diagnostic modes.

First run CPU correctness checks (no dataset/GPU needed):
`python -m unittest test.test_p_region_diagnostic -v`
Then run full script; it also checks existing train/test before training.
Do not compare raw 3090 and 5090 scores as intervention gains: compare C/U to
each machine's own baseline. Repeated favorable C vs U and consistent margin
evidence justify designing a future train/merge change; they do not prove it.
