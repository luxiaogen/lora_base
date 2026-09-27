# 3090: distillation holdout screen, Task0–2 only

Three runs in order: weight0 (short holdout control), weight0.25, weight1.
This is not a repeat of the full-data T10 baseline: each class reserves
every fifth training sample, including Task0. All train-source requests
use only the remaining samples, including W0 competence/prototypes and
CA class mean/covariance estimation. No test sample enters the holdout.
Historical held-out training images are accessed for evaluation only,
never optimization or teacher input. This is a validation protocol,
not the formal memory-free training run or a replay method.

Keep ImageNet-R T10 class order/increment20, seed1993, max_tasks3,
20 training epochs, CA5, anchor10, math-SDPA, frozen P-A, original gates.
Temperature2 is fixed. Task0 has no distillation. No defaults or local
data_path are changed. Fixed per-class source order makes the split
repeatable on the same dataset snapshot, not a random stratified split.

After each task's CA (Task0 has none), log `IncrementalHoldout` with
all-seen-class Total/Old/New accuracy, cosine CE, and class margin.
Evaluation preserves RNG and train/eval modes. Test scores still appear
in legacy reporting, but must not be used to pick weights. No extra
per-epoch validation or automatic T10 continuation is added.

Interpret Task1 and Task2 separately and their equal-weight mean. Advance
at most one candidate only if mean Total improves by at least 0.20pp,
mean Old and New do not decline, and neither task loses over 0.30pp in
either group. Check matched Task0, sample hashes and counts first. These
are screening rules, not statistical significance; tiny differences or
an Old-for-New tradeoff do not justify T10. If none pass, stop this sweep.
Do not modify these thresholds after viewing the results. A pass does not
prove long-horizon benefit, and weaker KD recovering New alone is not enough.

Estimated queue runtime: 1–1.5h from previous 80/96min full T10 timings;
holdout reduces training samples but adds evaluation. GPU time is unmeasured.

```bash
# Optional one-epoch Task0–1 route check, not performance evidence:
bash scripts/9_27_imgr10_distill_holdout_t3_3090.sh --smoke
# Full short screen, three runs:
lrun scripts/9_27_imgr10_distill_holdout_t3_3090.sh ./logs/9_27_imgr10_distill_holdout_t3_3090.log
# Print only validation metrics:
python3 scripts/summarize_distill_holdout.py ./logs/9_27_imgr10_distill_holdout_t3_3090.log
```

No runtime assertions/test gates and no server auto-launch. CPU checks
do not establish GPU execution or improved accuracy.
