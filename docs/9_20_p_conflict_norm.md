# P-conflict norm screen

This screen asks whether the earlier random-subspace gain can be reproduced by
an ordinary penalty on the effective P conflict update. It does not collect
features, estimate covariance, or store an old-task subspace.

For an effective conflict component `C_P` and the existing comparison rank `r`,
the added loss is `(r / d) * ||C_P||_F^2`. The `r / d` factor matches the
expected scale of projecting `C_P` onto a random rank-`r` input subspace, so the
comparison changes the direction information rather than the nominal loss
scale.

The supplied script runs one ImageNet-R seed-1993 experiment under the original
T10 split, stopping after Task2. All other DualMask settings match the previous
input-subspace screen. The machine-local `data_path` is read from
`exps/dlora/imgr10.json` and is never overridden.
