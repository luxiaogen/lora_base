# P update direction: ImageNet-R T10 seed1993

This is a predeclared single-seed method screen, not a JANUS/DGS reproduction.
Both GPUs run the same three configurations in sequence for within-machine
comparisons. Cross-GPU agreement is a reproducibility check, not two seeds.

| Run | Training update |
| --- | --- |
| baseline | Original SGD; all candidate calculations are read-only |
| norm | P B displacement right-preconditioned by damped A Gram inverse; match effective Q/K/V step norms |
| conflict | Same candidate pool and norm bounds; additionally bound conflict-region effective step norms |

Shared recipe: ImageNet-R T10, seed1993, Task0 and subsequent tasks20 epochs,
CA5, math SDPA, layer masks, original adaptive settings, S/P gates retained,
mask regularizer0.01 for both branches, independent Task0 anchor10,
P LR0.02, Task0 margin0.1, representation-steering disabled. Dataset path is
read from each machine's `exps/dlora/imgr10.json`. No dataset path overrides.
Old dense-update overlap logging and the redundant mask-reg gradient probe
are disabled equally in all three runs. The new direction probe is enabled
equally. No replay, stored historical feature bases, or extra inference pass.

## Exact update rule and limits

Task0 uses the original optimizer. Task1 onward, every fifth batch (0,5,10,...)
we capture P B and the classification gradient, then run the original SGD
step including momentum and any extra training loss. With frozen A, define
the *actual* SGD displacement `dB = B_after - B_before`. The proposal is
`dB @ inverse(A A^T + .01*mean_diag(A A^T)*I) * mean_diag(A A^T)`.
The Gram inverse is cached per task/layer. This is ordinary damped factor
preconditioning, not a newly derived optimal or guaranteed-safe direction.

Try fixed mixtures1/.5/.25 of the proposal and original displacement. Match
each Q/K/V fixed-gate effective norm, then RECOMPUTE each candidate's actual
dynamic gate. Accept only if actual effective step norms are between98% and
100.01% of the original SGD step, per projection, and the fixed-gate
classification first-order descent estimate improves and is positive.
Choose the best feasible first-order estimate in this finite pool. If none
qualifies, keep the exact original B. For `conflict`, the update on the union
of before/SGD-after/candidate-after conflict masks must additionally be no
larger than the original update on the SAME union (tolerance0.01%).

S, A, classifier SGD updates and momentum buffers are not overwritten by the
filter. P momentum remains the raw SGD accumulator; this is a post-SGD
displacement filter. Current first-order classification gradients are used
only at scheduled steps. No test/old labels are used to choose updates.

Mask rules/nominal budgets stay the same, but adaptive masks and their counts
may change with the trajectory. We log actual counts; do not claim identical
coordinate counts at every step. A norm cap on a proxy conflict region is
NOT a guarantee of retaining old-class predictions.

## Diagnostics

Every scheduled step contributes to `PStepSummary` acceptance/fallback counts.
First batches of epochs1/10/20 record each layer and Q/K/V as `PStepDirection`:
actual norm, common-region conflict norm, raw B-induced vs mask-switch step,
mask Jaccard/counts, and first-order predicted gain. The exact decomposition is
`G0*(W1-W0) + (G1-G0)*W1`, with branch gamma included. Components need not be
orthogonal; their norms do not sum to the effective norm.

At these detailed samples, `PStepProbe` evaluates all P proposals together on
the SAME TRAINING batch in eval mode; S and heads are held at the common
post-SGD state. It restores selected B, RNG and module modes. Actual batch
loss/accuracy/margin are diagnostics only and never select updates. These are
not holdout validation or evidence of old-feature damage. Baseline also gets
the same candidate work/probes so compute and diagnostic paths are comparable.

## Commands (run from the existing checkout)

```bash
# Each full launcher first runs a three-arm Task0-1 one-epoch GPU smoke.
# A failed/incomplete smoke or a candidate never applied stops the full queue.
bash scripts/9_25_imgr10_p_direction_3090.sh
bash scripts/9_25_imgr10_p_direction_5090.sh

# Optional smoke only; low short-run accuracy is expected.
bash scripts/9_25_imgr10_p_direction_3090.sh --smoke
```

Each GPU executes three full T10 runs, not three seeds. Recent original runs
were about76min on3090 and84-103min on5090 with other diagnostics. The new
candidate route has no measured full CUDA timing yet. Reserve roughly6-9h
per machine including smoke; the9h target is an estimate, not a measured
guarantee. Check the first full run's duration before extrapolating. Do not
silently truncate training or change interval halfway through a comparison.

At the end, each launcher automatically exports into its own log directory:
`figures_<timestamp>_full/`: CSV, provenance manifest, PNG and PDF figures.
To render downloaded outer logs later (one machine at a time):

```bash
python scripts/plot_p_step_direction.py YOUR_3090_LOG --out output/p_direction_3090
python scripts/plot_p_step_direction.py YOUR_5090_LOG --out output/p_direction_5090
```

Plots: accuracy/Old/New trajectories; stage-mean and final Old-New scatter;
effective/conflict norm constraint audit; layer-by-QKV mask-switch heatmap;
actual same-batch counterfactual losses. No error bars or significance claims
from one seed. Incomplete runs are excluded from performance figures.

## Predeclared interpretation

- First check smoke, completed10 tasks, matched Task0, configuration and
  nonzero candidate acceptance. Unit tests are not CUDA/full-run evidence.
- Conflict vs baseline: aim for Average+0.15pp, Last+0.20pp, stage-mean
  New+0.30pp, stage-mean Old no worse than-0.10pp, Forgetting no worse than
  +0.10pp. These are practical screening thresholds, not significance tests.
- Compare conflict vs norm at matched effective step sizes. If norm is just
  as good, attribute any gain to generic factor preconditioning; do not claim
  a benefit from conflict localization.
- If acceptance is nearly zero, predicted gains do not translate into actual
  batch gains, or T10 trades away Old, report that failure and stop this recipe.
- No learning-rate/ridge/mix/interval sweep or new seeds in this queue. Freeze
  settings before running; test curves are for reporting this predeclared
  comparison, not choosing new settings. Any positive single-seed result
  remains provisional.
