# LoRA inheritance experiment (2026-09-13)

Branch: `codex/lora-inherit`, based on `57032f8`. Main is not changed.

## S-only / P-only follow-up (2026-09-14)

The new `dual_mask_lora_inherit_branches=both|s|p` selects which branch inherits.
The existing `dual_mask_lora_inherit` remains the master switch; false always means
fresh initialization. Omitting the new option preserves the previous both-branch behavior.

| Condition | Master switch | Branch selector | S initialization | P initialization |
|---|---|---|---|---|
| fresh | false | both | fresh each task | fresh each task |
| s_only | true | s | inherit from Task1 | fresh each task |
| p_only | true | p | fresh each task | first active at Task1; inherit from Task2 |

Both S and P still train normally. Task0 remains fresh QKV with anchor10. Selected
branches alone cache their factors, and retain the centered `BA - BA_start` update.
Adaptive rank resizing, frozen A after Task0, weight decay and merge semantics are
the same as the original inheritance experiment described below.

Run these scripts from the repository root in the existing training environment:

```bash
git fetch origin
git switch codex/lora-inherit
git pull --ff-only origin codex/lora-inherit

# 5090D: ImageNet-R and ImageNet-A; 18 full T10 runs
lrun scripts/9_14_lora_inherit_branches_5090.sh ./logs/9_14_lora_inherit_branches_5090.log

# 3090: CUB; 9 full T10 runs
lrun scripts/9_14_lora_inherit_branches_3090.sh ./logs/9_14_lora_inherit_branches_3090.log
```

For each dataset, seeds 1993/1996/1997 each run fresh, s_only, then p_only.
All 27 runs start independently from pretrained weights, train Task0-9 for 20
epochs/task and CA5, and never load a Task0 checkpoint. Dataset paths are read from
the machine's existing JSON configs. The scripts do not change directories.
The run count is 1.5 times the original sweep, so budget about 1.5 times its
observed wall time; this is not a ten-hour cutoff. To run only ImageNet-R:

```bash
bash scripts/9_14_lora_inherit_branches.sh imgr10
```

Use `DRY_RUN=1 bash scripts/9_14_lora_inherit_branches_5090.sh` to inspect all
commands without loading models or checking dataset paths. The launcher preflight
runs the inheritance and script tests; synthetic 2-epoch output belongs to those
tests, followed by the real 20-epoch experiments.

Verification: 23 focused tests pass locally, including both/s/p three-task CPU
smokes, nonselected-branch initialization/RNG, zero initial delta, gradients,
exactly-once merge, branch-selective caching, and comparison of every generated
CLI override with the previous sweep specifications. This verifies execution,
not GPU performance. Compare same-machine fresh against s_only and p_only on
Average/Last, Old/New and Forgetting, with all three seeds reported.

## Exact change

`dual_mask_lora_inherit=false` is the unchanged fresh-initialization baseline.
`true` caches the last active S/P A and B before clearing merged adapters, then
copies them into the next task. Forward, regularization and merge all use
`delta = B @ A - B_start @ A_start`. Gates apply to this new delta. The accumulated
backbone therefore receives only newly learned changes, never the inherited BA twice.

Task0 is fresh QKV with the original initialization and task0 anchor10. All later
tasks also use QKV. S starts inheriting at Task1. P is first trained at Task1 and
starts inheriting at Task2 (the unused Task0 P is not inherited). A remains frozen
after Task0 and B remains trainable, following the original method.

The controller still chooses P rank. On a rank change, copy the first
`min(previous_rank, current_rank)` A rows/B columns; extra A rows retain the usual
random initialization and extra B columns are zero. A rank decrease drops trailing
directions; it does not compress the old update using SVD. Log lines report both
ranks and copied rank. This tests overlap inheritance, not exact full-rank retention.

Important interpretation: with frozen A and zero weight decay, the new delta is
`(B-B_start) @ A`. Thus inheriting B provides no independent functional warm start;
ImageNet-R mainly tests retaining the learned/frozen A subspace. S retains its
Task0-learned directions; P retains the random frozen directions from its first
active task, except for rank resizing. No optimizer state is inherited. CUB and
ImageNet-A retain their configured weight decay (5e-4), which acts on the full
inherited B, not only `B-B_start`. This effect is deliberately included in this
inheritance experiment; do not attribute any gain purely to basis reuse there.

## Night schedule

| Machine | Dataset | Seeds | Conditions | Full runs |
|---|---|---|---|---:|
| 5090D | ImageNet-R T10, rank64 | 1993/1996/1997 | fresh / inherit | 6 |
| 5090D | ImageNet-A T10, rank32 | 1993/1996/1997 | fresh / inherit | 6 |
| 3090 | CUB T10, rank32 | 1993/1996/1997 | fresh / inherit | 6 |

Each seed runs fresh then inherit. All 18 runs start from pretrained weights and
train Task0-9 independently, without saving/loading any Task0 checkpoint. Both
conditions use 20 epochs, CA5, anchor10 only at Task0, global+suppress,
Energy50/floor10 and adaptive P rank. Diagnostic-only extra CA evaluations are off.
Dataset-specific learning rates, batch size, margins and weight decay come from
the local JSON. Branch scales are S=.5/P=.75 for R/CUB, S=.5/P=1 for A.
The scripts do not change directories or override data_path.

Expected time: roughly 9-11 hours on 5090D and 6-8 hours on 3090, not a hard cutoff.
This uses recent R timings (~1.2-1.3h on 5090D, ~2.6-2.7h on 3090) and the 9_6.log
ratio for A/CUB (~22min/~33min vs R~78min). The CUB estimate on 3090 is extrapolated;
inheritance speed is unmeasured. Runs finish normally even if the estimate is exceeded.

```bash
# Activate the machine's existing training environment, from repository root.
# 5090D
lrun scripts/9_13_lora_inherit_5090.sh ./logs/9_13_lora_inherit_5090.log
# 3090
lrun scripts/9_13_lora_inherit_3090.sh ./logs/9_13_lora_inherit_3090.log
```

Both scripts first run the focused tests (including synthetic three-task CPU
training), print code/GPU/data-path identity, then train the full runs. Failures are
recorded, remaining runs continue, and the final exit code is nonzero if any failed.
Prefixes and per-run logs include timestamps. Review effective JSON settings and
logged overrides when comparing to historical runs.

## Verification and scope

```bash
PYTHONWARNINGS=ignore python -m unittest test.test_lora_inherit
```

Checks include unchanged Task0/RNG, inherited factors, zero initial effective delta,
trainable gradients, P rank resizing, sequential initialization, weight-decay
semantics, immutable W_pre, and pre/post-merge equivalence through Task2.
The CPU smoke uses the real Learner with a tiny ViT and synthetic images; it does
not establish GPU accuracy or a successful full dataset run. Full lightweight
tests on local PyTorch 2.0.1 show an existing entropy assertion off by 1.19e-7;
the same test fails at unmodified 57032f8. It is not included in the launch preflight.
Fresh-model state_dict reconstruction of inheritance buffers is not implemented;
these scripts explicitly disable task0 checkpoint save/resume.

Compare paired Average, Last, Old/New and Forgetting for all three seeds on each
machine. Report Task0 variation separately. Do not treat a single high Task0 or
a small single-seed difference as proof of an inheritance benefit.

Sweep specs in scripts/sweeps/lora_inherit_{3090,5090}.json generated the explicit
commands. Their scripts additionally contain root/preflight checks and timestamped
prefixes; preserve those additions if regenerating with generate_sweep.py.
