# S/P mask-regularizer isolation, 5090

Three sequential ImageNet-R T10 seed1993 runs: A both penalties, C S-only,
D P-only. A is repeated on this GPU for a matched comparator. The existing
3090 neither-penalty run is not a matched fourth cell for a formal 2x2 claim.
If a complete factorial analysis is needed later, run neither on this GPU too.

New switches (default true): `dual_mask_s_reg_enabled`,
`dual_mask_p_reg_enabled`. Each zeros that branch's entire mask penalty
(protection + conflict), not its LoRA branch, forward gate or merge gate.
Zeroed entries remain in the original mean, so the other branch does not
double in weight. Task0 anchor is separate and unchanged. Defaults reproduce
the previous loss expression. No JSON data paths or defaults are modified.

Common settings match the 3090 mask-reg pair: weight0.01, CA5,20epochs,
math-SDPA, layer granularity, P LR0.02, original adaptive settings. Gradient
diagnostic is identical: Task1–9 epochs1/10/20 first batch, S/P B gradients.

In the existing checkout and training environment, run with Bash:

```bash
# Smoke first; these are not performance results.
bash scripts/9_25_imgr10_branch_reg_5090.sh --set max_tasks=2 --set init_epoch=1 --set epochs=1 --set ca_epochs=1 --set wandb_mode=offline

# Full three-run experiment.
bash scripts/9_25_imgr10_branch_reg_5090.sh
```

Per-run timestamped logs and fingerprints:
`logs/shell_logs/imgr10_branch_reg_5090/`.
Estimate about2.5–3hours total using earlier5090 T10 runs of roughly47–60min;
new diagnostic overhead has not been measured on that GPU.

Report C−A and D−A on this machine: Average/Last, Task1–9 mean Old/New and
Forgetting. Check Task0 matches and config differences first. Do not equate
zero regularizer gradient with an orthogonal direction (cosine is undefined).
This is an ablation, not a gradient-correction method or a demonstrated gain.
