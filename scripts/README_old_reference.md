# Old-score stop-gradient comparison

Question: does removing the direct old-score gradient from the extra competition
hinge retain New gains with less Old cost? This is a hypothesis, not a safety
guarantee. Suppressing an incorrect old score on a new image is not inherently
harmful. No old-sample constraint is introduced by stop-gradient.

`old_competition_detach_old=true` detaches the maximum old cosine score before
subtracting the correct new cosine. At identical parameters the loss and active
fraction are unchanged, but the gradient of this extra term flows only through
the positive score. The current head and shared trainable features still receive
gradients; old scores may change on later steps. The default is false and preserves
the existing implementation. Weight0/Task0 still bypass the extra loss.

## Two-machine schedule

Each machine runs ONE new full ImageNet-R T10 seed1993 candidate concurrently:

- 3090: `bash scripts/9_26_imgr10_old_reference_3090.sh`
- 5090: `bash scripts/9_26_imgr10_old_reference_5090.sh`

Both use anchor5, 20 epochs, CA5, math SDPA, layer double gates, P LR .02,
mask regularization .01 and competition weight .1. JSON data_path stays local.
No seed sweep and no strength tuning. Never compare absolute scores across GPUs.

First the launcher runs focused tests and two one-epoch Task0–1 smoke runs
(competition off versus detached competition). It reuses the existing smoke
checker: all stages complete, Task0 equal, candidate loss positive. Smoke logs
go to the existing old_competition machine directory with a fresh timestamp;
full candidate logs go to old_reference. `--smoke` runs only these smoke checks;
`--dry-run` displays the single full-run command without loading data.

Reuse each machine's completed `9_26_imgr10_old_competition_<GPU>.log` baseline
and non-detached candidate only after comparing effective settings and source
changes against commit 78e0bda. The only method change is this flag. Preserve the
same software environment, data, pretrained weights and local source edits.
If any of those changed, rerun a matched control rather than claiming a clean
historical comparison. Full Task0 should match its prior same-machine run;
matching Task0 alone does not prove full trajectory equivalence.

Approximate time from recent logs: 3090 full ~80 min; 5090 full ~120–124 min,
plus short smoke/tests. Scripts do not connect to servers or launch remotely.

## Readout

Keep the existing three-stage audit and CSV output. Report Average, Last,
Forgetting, stage-mean and final Old/New, and new-to-old / old-to-new error rates
(counts divided by partition size). True-partition oracle metrics remain
diagnostics only, never deployed inference. No automatic winner selection.

A promising result reduces new-to-old errors without material Old loss and
improves overall accuracy. If New rises at Old's expense with no net benefit,
do not retain it. If it matches the original candidate, this gradient path has
not emerged as a useful optimization lever. One seed is preliminary evidence.
Any later hyperparameter tuning requires training holdout, not these test reports.

## Local verification (2026-09-26)

15 focused tests pass, including identical forward losses, changed feature
gradients, positive-path gradients retained, old-head gradients absent, inactive
hinge, Task0/weight0 skip, hook wiring, and all generated script overrides.
Python compilation, shell syntax and diff whitespace checks pass. Full discovery
passed 193 tests; two additional modules could not import because the local Python
environment lacks easydict. CUDA smoke and performance are not yet verified.
Both supplied old_competition GPU logs have two full runs whose explicit common
training overrides match the new specs (excluding run/group names).
