# Fixed conflict-suppression coverage (planned; not performance evidence)

ImageNet-R T10 seed1993, anchor5, 20 epochs, CA5, math-SDPA. Both S/P
conflict gates, S protection, P plastic mask and regularizers remain enabled.
CA is the original Gaussian version (`ca_real_new_features=false`); competition
and P direction correction remain off. Machine-local JSON supplies data_path.

This sweep fixes beta=0.5, disables energy-adaptive selection and old-overlap
strength adaptation, and enables `dual_mask_conflict_exact_topk=true`.
Default false preserves previous selection behavior and existing experiments.
Only coverage changes WITHIN this sweep; the current adaptive CA baseline is
not its fixed-10% reference.

| Machine | Sequential coverage | Full T10 runs | Estimated additional time |
| --- | --- | --- | --- |
| 3090 | 10%, 0%, 40%, 60% | 4 | 5.3–6 hours |
| 5090 | 10%, 20%, 100% | 3 | 6–6.5 hours |

Timings extrapolate recent runs, not measured fixed-coverage runtimes. Each
machine has its own 10% reference; cross-machine absolute values must not be
joined as a single performance curve. All runs use seed1993, not multi-seed.

## Selection semantics

For each layer and each S/P branch separately, K=floor(ratio*N), where N is
the full QKV update matrix coordinate count. Exact Top-K uses returned indices
rather than a threshold, so tied values cannot inflate K. Zero/all-tied scores
also follow that count. Tied-coordinate identities need not match across devices;
selection consumes no random numbers. The mask is recomputed from the current
update, not frozen for an entire task. Forward and merge use the same rule.

0% removes conflict-gate suppression, NOT the regularizer or S/P base gates.
100% multiplies all otherwise-permitted updates by 0.5. Task0 stays unmasked.
This is update suppression coverage, NOT model sparsity. P still intersects the
selected mask with its plastic region; its effective coverage can differ from
the nominal full-matrix ratio. Existing applied-budget logs record actual masks,
P overlap and suppression; StageAudit records pre/post-merge and post-CA metrics.
Changing the effective update can also change the value of the retained
regularization loss; it is not a pure optimizer-independent scaling experiment.

## Running

Full launch directly runs the experiments. No automatic tests, preflight,
accuracy-equality assertions, or hidden short runs. A failed Python run is still
reported as failed; subsequent variants run, and the script returns failure.

Tests run separately:

```bash
python -m unittest test.test_fixed_coverage test.test_dual_mask_core
```

Optional GPU smoke (one epoch each, Task0–1; not performance evidence):

```bash
bash scripts/9_26_imgr10_fixed_coverage_3090.sh --smoke
```

Use `--dry-run` to print full commands without training. Smoke/full logs have
separate timestamped names. Do not pass smoke overrides to the full experiment.

The user's Fish function is `lwait PID "COMMAND"`. Wait for the entire current
CA Bash queue, not its first `python main.py` child. The function waits only for
exit, not successful completion. Keep its terminal/session alive.

Find the current outer queue PID on the appropriate machine:

```fish
pgrep -af '[b]ash .*9_26_imgr10_ca_real_new_3090.sh'
pgrep -af '[b]ash .*9_26_imgr10_ca_real_new_5090.sh'
```

Use only the corresponding machine's command below, replacing PID with the
observed numeric outer Bash PID. Pull AFTER the running CA queue exits, to avoid
changing source while a multi-run queue is active. A pull conflict stops the
next launch; preserve local edits rather than forcing checkout/reset.

```fish
lwait PID "git pull --ff-only origin codex/mask-budget-comparison-20260924 && lrun scripts/9_26_imgr10_fixed_coverage_3090.sh ./logs/9_26_imgr10_fixed_coverage_3090.log"
lwait PID "git pull --ff-only origin codex/mask-budget-comparison-20260924 && lrun scripts/9_26_imgr10_fixed_coverage_5090.sh ./logs/9_26_imgr10_fixed_coverage_5090.log"
```

## Analysis after completion

Report Average/Last, stage-mean and final Old/New, Forgetting, new-to-old and
old-to-new errors. Plot within-machine coverage curves and differences relative
to fixed10%, Old–New tradeoffs, and actual suppressed update norms.
An intermediate ratio beating 100% supports selective suppression under these
settings, not superiority of conflict scores at matched suppressed norm.
Hard-zero sparsity and random/magnitude/conflict comparisons are not part of
this sweep. No method selection based on the highest test point; any tuning
claim needs training holdout and subsequent independent confirmation.

## Local verification

203 lightweight tests passed, including 6 new fixed-coverage tests and a CPU
Attention training/merge check at all six ratios. Python compilation, shell
syntax and dry-run/spec comparisons passed. Two existing modules
(`test_ca_diagnostics`, `test_task0_margin_screen`) were excluded because the
local environment lacks easydict. No CUDA/data smoke or server launch has been
performed; these checks do not establish runtime or performance on either GPU.
