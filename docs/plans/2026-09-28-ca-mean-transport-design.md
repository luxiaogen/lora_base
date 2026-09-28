# CA mean-only transport: approved two-machine experiment

## Hypothesis and single change

Full statistics transport was near-flat on the 5090 (Average87.141 vs87.092 reference,
Last82.62 vs82.60). Test whether changing covariance offsets a useful mean correction.
This is an unproven hypothesis, not an established optimization. The failed diagonal
shrinkage experiment is not proof that affine covariance transport is harmful.

Reuse the existing before/after current-training-feature capture, diagonal scale and offset
fit, and task lifecycle. With `ca_stats_transport=true` and
`ca_stats_transport_mean_only=true`, update only:

```python
mapped_means = means * scale + offset
mapped_covariances = covariances
```

The new flag defaults false: existing full transport and disabled baselines are unchanged.
Old means are updated cumulatively each task. Each old class keeps the covariance recorded
when it was introduced. New class statistics are still computed normally after transport.
Task0 is excluded by the existing transport lifecycle. This is not offset-only translation.
No old images, test labels, new loss, mask change or inference change. No runtime validation
or automatic test gate is added. Logs append `mode=mean_only` to `CATransport` telemetry.

## Exactly two candidates, no baseline reruns

| Machine | Anchor | Scope | Archived reference |
|---|---:|---|---|
| 3090 | 2.5 | ImageNet-R T10 seed1993 | `9_28_imgr10_anchor2p5_save_t10_3090.log`, Average87.281/Last82.67 |
| 5090 | 5 | ImageNet-R T10 seed1993 | `9_27_imgr10_head_balance_5090.log`, off group87.092/82.60; `diag_transport.log`87.141/82.62 |

Both use full training data, 20 epochs, CA5, math-SDPA, existing dual gates/losses/LRs,
and per-task evaluation weight saving. Cross-margin and covariance shrinkage are explicitly
zero. Machine JSON owns data_path/device. No checkpoint resume is implied: each trains from
Task0. The 3090 has no matching anchor2.5 full-transport run, so it only tests net gain over
its baseline, not the separate causal contribution of covariance transport.

Do not compare absolute performance across machines/anchors. Check pre-first-CA differences
and configuration/environment drift before attributing small changes. Require net
Average/Last gains without sacrificing Old to New. Near-flat results do not justify a
coefficient sweep. One seed on two machines is not multi-seed validation.

## Launch and evidence

Scripts: `scripts/9_28_imgr10_ca_mean_transport_3090.sh` and
`scripts/9_28_imgr10_ca_mean_transport_5090.sh`. Each has exactly one candidate.
Use `bash SCRIPT --smoke` for an optional separate two-task, one-epoch GPU check;
`bash SCRIPT` for full training, or `--dry-run` to inspect CLI without training.
No automatic smoke or tests precede full training. All runs use unique timestamp prefixes.
Outer stdout records commit/dirty status; inner logs record command/output/exit code.

Budget: allow roughly1.5–2h on3090 including paired feature extraction; shared5090's last
full-transport run took4h53m, which is a reference, not a promised duration.

Test the actual transport hook and helper: identical mean mapping to full transport,
unchanged covariance object/content over repeated tasks, unchanged input tensors/RNG,
default compatibility, correct machine recipes, full/smoke commands and local data paths.
CPU checks do not establish CUDA success or accuracy gain. The assistant does not launch
server training; the user runs the delivered commands.

## Local verification (2026-09-28)

16 related tests passed. Full discovery: 293 entries, 291 passed, two existing modules
(`test_ca_diagnostics`, `test_task0_margin_screen`) could not import because local
`easydict` is missing. Python compilation, Bash syntax, and both full/smoke dry-run
recipes passed. Default full transport was also compared against HEAD's original helper
on identical tensors: returned means and covariances were bitwise equal. No CUDA training
or server process was started.
Independent code review approved; all nine transport tests were rerun and passed.
