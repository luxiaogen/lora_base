# Incremental P A+B comparison (3090)

Run in the existing project:

```bash
bash scripts/9_27_imgr10_plora_train_a_3090.sh
```

The queue runs only one complete ImageNet-R T10 seed1993 candidate:
`trainable` (current P A and B trained from Task1). Reuse the existing
3090 frozen-A baseline with the matching recipe; do not rerun it.
The candidate uses anchor=10, 20 epochs, CA5, math-SDPA, the existing
layer masks and adaptive-rank rule. Task0 and S training are unchanged.
No data path is passed; each machine keeps its own JSON path.

This is an exploratory capacity ablation, not a demonstrated improvement
or a reproduction of LoRA-FA. Only `plora_train_a` changes relative to the baseline.
The default is false and no default JSON is edited. The new A parameters
enter the existing LoRA optimizer at the same learning rate as B.

The existing parameter log reports trainable scalar counts; the new
`P-A training` line reports additional A scalars per task. For layer l,
the added count is `rank_P,l * d_in,l`, summed over layers. Adaptive ranks
may subsequently differ because learning changes. More trainable parameters
and activation/gradient memory are expected; inference still uses one merged
model. This is not a same-trainable-parameter-budget comparison.

At zero-B initialization, the first A gradient can be zero; later steps
must update A. Tests cover this, frozen S A, unchanged initialization,
and masked-forward/merge equivalence. There are no runtime assertions or
automatic test gates in the queue. Optional `--smoke` runs two tasks with
one epoch per stage; optional `--dry-run` only prints commands.

Compare same-machine Average, Last, stage-average Old/New and Forgetting,
plus time and parameter counts. Task0 is unchanged by design and should
be checked as a comparability indicator. Do not accept a New gain alone
if Old and total accuracy deteriorate. GPU execution remains to be run
on the server; local CPU unit tests do not establish performance.
