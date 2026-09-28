# 3090: CA diagonal covariance shrinkage

## Question and status

Approved single-candidate experiment, not a demonstrated improvement. Test whether weakening
estimated cross-feature correlations in CA Gaussian pseudo-features improves the existing
anchor2.5 recipe. This is separate from the running 5090 statistics-transport experiment.

## Single change

At CA sampling only, use `(1-alpha) * covariance + alpha * diag(diag(covariance))`,
with `ca_cov_shrinkage=0.5`. Default is 0 (original sampling path). Apply equally to
all seen classes, not only new classes. Preserve means, marginal variances, trace,
stored raw statistics, sample counts, sampling/shuffle RNG calls, optimizer and CE.
Task0 has no CA. No backbone, mask, merge, or inference change. No runtime assertions added.
This may discard useful correlations; it is not evidence that covariance noise is the bottleneck.

## Recipe and comparator

- Original project and branch `codex/mask-budget-comparison-20260924`.
- ImageNet-R T10, seed1993, full training set, no holdout; official test reporting.
- Task0 anchor2.5, 20 training epochs, CA5, math-SDPA, original layer dual gates.
- Cross-task margin weight0, statistics transport false; other candidates disabled.
- One candidate; no repeated baseline. Per-task evaluation weight saving enabled.
- Machine JSON owns `data_path` and `device`.
- Reference: `/Users/luxiaogen/Desktop/loda_logs/9-28/9_28_imgr10_anchor2p5_save_t10_3090.log`.
  Average87.281, Last82.67, Task0 97.10. Older same-recipe run was87.214; small differences
  must be interpreted against this observed repeat variability, not as reliable gains.
- Expected full-run time around80–90 minutes from recent same-machine79-minute runs.

## Launch

Optional GPU smoke, separate from full training (two tasks, one epoch each):

```bash
bash scripts/9_28_imgr10_ca_cov_shrinkage_3090.sh --smoke
```

Full candidate (no automatic smoke or test gate):

```bash
bash scripts/9_28_imgr10_ca_cov_shrinkage_3090.sh
```

The launcher prints commit and dirty status to stdout (captured by the user's outer `lrun`
log); the per-candidate log records exact CLI and exit code. A unique timestamp is used
for logs and checkpoint prefixes. `--dry-run` prints without training.

## Interpretation

First confirm full10 tasks, finite metrics, save messages and `CACovShrinkage` telemetry for
Task1–9. Compare Task0 and pre-first-CA metrics for unexpected reproduction drift. Then compare
Average/Last, stage-average and final Old/New, Forgetting and error transitions. Do not call
a New-only gain an improvement. This is single-seed exploratory evidence, not a paper claim.
No extra shrinkage values or combined transport experiment are scheduled.

## Verification

Tests execute the actual CA method with small CPU tensors: alpha0/default equality,
alpha0.5/1 formula, unchanged means/diagonal/raw statistics/RNG consumption, positive
definiteness and classifier-only changes. Script tests compare the complete effective recipe
with the saved-weight baseline and verify full/smoke dry runs from a different directory.
Local tests do not establish GPU end-to-end success or accuracy gain.

Local verification on 2026-09-28: 14 focused tests passed; full discovery ran291 entries,
289 passed and2 modules could not import because local `easydict` is missing
(`test_ca_diagnostics`, `test_task0_margin_screen`). Python compilation, Bash syntax and
full/smoke dry-run checks passed. Independent code review approved; its separate run of
the2 new tests passed. No server/GPU training has been started by the assistant.
