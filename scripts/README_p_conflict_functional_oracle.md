# P-conflict functional oracle diagnostic

This experiment keeps the original ImageNet-R T10 partition and trains only
Task0-2. It never changes training, merge, CA, or the final prediction rule.

For the current task's retained P-conflict update, each attention layer is split
into Q/K/V and eight consecutive LoRA-rank groups. On a deterministic half of
four test samples per seen class, the diagnostic measures the class-margin
change caused by temporarily removing one component. Components whose removal
improves both Old and New margins form a label-selected diagnostic plan. The
combined plan is evaluated on the other half of the samples, together with
component-energy-matched magnitude and seeded-random controls.

The reported `functional_oracle_holdout` is not a formal label-free CIL score:
it uses true labels to select a diagnostic intervention. Temporary removals are
restored exactly, and all captured factors are discarded after the diagnostic.

Run from the repository root:

```bash
lrun \
  scripts/9_20_p_conflict_functional_oracle_3090.sh \
  ./logs/9_20_p_conflict_functional_oracle_3090.log
```

The script reads `data_path` only from the machine-local
`exps/dlora/imgr10.json`.
