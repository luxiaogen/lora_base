# P-conflict SVD functional diagnostic

This is the final read-only harmfulness diagnostic for the current P-conflict
direction. It keeps the ImageNet-R T10 protocol and trains only Task0-2.
Training, merge, CA, and the final prediction rule are unchanged.

For each attention layer, the diagnostic takes the actual retained masked
P-conflict update `C_P`, separates Q/K/V, and decomposes each matrix by SVD.
The singular directions are placed into eight non-empty, near-equal-energy
orthogonal groups. This replaces the earlier arbitrary LoRA-rank grouping.

A deterministic half of four samples per seen class selects groups whose
temporary removal improves both Old and New class margins. The other half
evaluates that plan against the unchanged baseline and energy-matched magnitude
and seeded-random controls. Temporary removals are restored exactly.

`functional_oracle_holdout` uses true labels to select components, so it is a
diagnostic upper bound, not a formal label-free CIL score. If it produces no
real corrected samples, or does not beat the matched controls without harming
New, stop the P-conflict harmfulness-screening direction.

Run from the repository root:

```bash
lrun \
  scripts/9_20_p_conflict_svd_functional_oracle_3090.sh \
  ./logs/9_20_p_conflict_svd_functional_oracle_3090.log
```

The script reads `data_path` only from the machine-local
`exps/dlora/imgr10.json`.
