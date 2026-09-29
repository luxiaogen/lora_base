# 2026-09-29 fresh same-machine ablations

No training-method changes. Both queues read data_path from local imgr10.json.
Common: ImageNet-R T10, seed1993, anchor2.5, 20 epochs, CA5, math-SDPA,
layer baseline, original S/P gates and regularizers. No checkpoints or uniform
norm matching, late averaging, frozen masks, staged training, or CA candidates.

3090 queue (6 full runs): layer/budget1 baseline, projection, model, budget0.5,
budget1.5, magnitude1. The first run serves both granularity and budget controls.
Retain original adaptive coverage/old-overlap strength. Global modes redistribute
the local-reference count on their own evolving updates. Magnitude1 matches the
conflict-reference count on the same current update; different runs can still
have different realized adaptive density. Do not describe final cross-run counts
as strictly identical. Soft suppression, not weight pruning.

5090 queue (6 full runs): fixed10 baseline, fixed0,20,40,60,100 percent of each
branch's full layer QKV matrix. Exact Top-K; energy coverage and old-overlap
strength adaptation disabled; beta=.5. S protection and P plastic restrictions
remain, including at 0%. This is not the 3090 adaptive baseline. Keep GPU results
separate. All coverage points are now fresh and on the same 5090 machine.

Each queue first runs all six variants for Task0–1, one epoch each, CA1, offline;
if smoke fails, the full queue does not start. Smoke log names are distinct.
Each full run has a separate timestamped log; full queue records failures and
continues remaining variants. No numerical runtime guards added to the method.

Run from original repository after pulling:

```bash
# 3090: approximately 8–9 h including smoke, from recent ~80 min/full run.
bash scripts/9_29_imgr10_granularity_budget_refresh_3090.sh
# 5090: historically ~5–6 h if idle; shared GPU can exceed 8 h.
bash scripts/9_29_imgr10_coverage_refresh_5090.sh
```

These are manual night queues; no start time or scheduler is installed. Inspect
current GPU processes before starting; do not overlap another experiment.
Report runtime, realized masks, Avg/Last, stage and final Old/New, forgetting.
Old runs are context only, not the new paired controls. No test-driven early
stopping or candidate filtering: all requested full T10 variants are scheduled.
