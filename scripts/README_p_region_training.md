# 5090D: training-time P-region attenuation (12 planned runs)

Branch is descended from 102b65c via the read-only diagnostic commits. Main is
untouched. Run from repository root in the server conda environment:

```bash
SMOKE_ONLY=1 bash scripts/9_19_p_region_train_merge_5090_overnight.sh
lrun scripts/9_19_p_region_train_merge_5090_overnight.sh ./logs/9_19_p_region_train_merge_5090_overnight.log
```

Fish users can use `env SMOKE_ONLY=1 bash ...` for the first command.
SMOKE_ONLY runs CPU unit/tiny optimizer+merge tests, not full dataset training.
The main script repeats these checks, then requires CUDA and existing train/test.
Dataset path comes ONLY from local exps/dlora/imgr10.json. Do not copy another
machine's JSON or apply old code stashes. No auto-split, checkpoint, task router,
inheritance, old-image replay or test-based model selection.

## Fixed protocol

ImageNet-R T10 (20+20 classes), full Task0..9, QKV, rank64, adaptive P rank,
fresh initialization each run, 20 epochs/task, CA5, Task0 unmasked+anchor10,
global suppress Energy50/floor10, original protection regularizer .01, extra
conflict regularizer OFF. S branch unchanged. Original post-CA read-only region
diagnostic is OFF in all runs; each row is a separately trained model with its
own statistics and CA. Normal Average/Last/Old/New/Forgetting are its scores.

## Mechanism

For t>0, each P layer first computes original safe update D. With the ORIGINAL
conflict mask M, C=D*M and U=D*(1-M). The additional removal budget is
`r = amount * min(norm(C), norm(U))` (Frobenius norm). Conflict arm uses
`D - r/norm(C)*C`, nonconflict arm uses `D - r/norm(U)*U`. Either zero norm means
both candidate removals are zero; zero B initialization keeps finite gradients.
Norm coefficients/mask are detached; gradients flow through the effective delta.
Mask/budget are recomputed as learning proceeds, not frozen from an old model.
Amount=.5 is half the SMALLER norm budget, not always deleting half of C or U.

Training P forward uses gamma-scaled BA and the same _safe_delta as final merge.
Task0 has no P intervention; S is untouched; merged delta is written once and
LoRA units cleared as before. No conditional inference; CA uses the actually
merged model. Existing importance regularizer still applies to effective safe
updates, so its value changes with the gate. This is NOT a CE-only intervention.
Consistent training/merge gating already exists in the baseline; only the EXTRA
region attenuation is new.

Norm matching is exact between hypothetical C/U interventions on ONE delta.
Independent runs learn different deltas; their realized budgets can diverge.
`P-region training task=... layer=...` logs both norms, target/actual removed norm,
and fractions at every merge. Check these before attributing outcomes to region.

## Execution order

1. seed1993 baseline
2. seed1993 conflict .5
3. seed1993 nonconflict .5
4. seed1993 conflict .25
5. seed1993 nonconflict .25
6. seed1993 baseline repeat (same settings, new process)
7-9. seed1996 baseline / conflict .5 / nonconflict .5
10-12. seed1997 baseline / conflict .5 / nonconflict .5

The two JSON specs record the exact matrix, expanded using the research-skill
sweep generator and combined with a budget/preflight wrapper. Script tests
verify the expanded commands match both specs.

## Time and outcome interpretation

Recent 5090D diagnostic run: 59.52 min including extra evaluation. Estimate
10-12 hours for 12 runs, NOT measured for this new training variant. Default
budget 43200 seconds; before each run reserve initially 3600 seconds, increased
to max observed successful duration *1.15. If insufficient budget remains,
prints BUDGET STOP / DEFERRED_BUDGET and leaves unstarted runs for later. Does not
kill a running experiment, so this is a soft budget, not guaranteed hard deadline.
`BUDGET_SECONDS=0` disables the guard; not recommended for a fixed reservation.
Failed jobs are recorded, later jobs continue. Logs and timestamped status TSV
are in logs/shell_logs/p_region_train_merge_5090. No completed job is silently
called a complete 12-run sweep when others were budget-deferred.

Compare within seed on this machine, not against best historical 87.14.
- C beats baseline AND U with Old/New retained: supports targeted attenuation.
- C and U similarly improve: generic magnitude effect, not evidence of good conflict localization.
- Old improves but New falls: tradeoff remains after adaptation; not a clean gain.
- .25 helps where .5 hurts: evidence of strength sensitivity, not grounds for a large grid search.
- Gains comparable to baseline repeat or inconsistent across seeds: no robust win.
Do not derive new thresholds or gates from test labels; no automated winner selection.
