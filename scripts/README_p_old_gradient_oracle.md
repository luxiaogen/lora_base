# Privileged old-gradient reference: 5090, ImageNet-R Task0-2

This is explicitly an **old-training-data Oracle diagnostic**, not a new
exemplar-free method and not a mathematical upper bound. The user permits old
training images only for this reference. Test images/labels never choose an
update. No true test task ID, task router, second inference pass, saved dense
historical update, teacher copy, or new checkpoint is used.

## One candidate, fixed recipe

Run seed1993, full current-task training data, Task0-2 under the T10 class
partition, 20 epochs per task, CA5, math-SDPA, anchor2.5, layer masks, the
original adaptive S/P gates, mask regularizer0.01, original LR0.02 and frozen
incremental A. All recent candidate switches are off. Task0 is untouched.

Only `p_old_gradient_oracle=true` activates the privileged P update filter.
The default/missing switch keeps the ordinary training path. The launcher
does not rerun a full baseline; compare with an existing **5090** run only
after checking the effective configuration and Task0 result. An earlier
anchor2.5 T3 log has Task0=96.66; different Task0/configuration means the
performance comparison is not strictly paired. Never subtract 3090 scores.

## Exact rule

Every fifth Task1+ batch (0,5,10,...):

1. Capture current P B before the original optimizer step.
2. Run the original loss/SGD, including S/head updates and momentum.
3. Hold S and heads at their **common post-SGD state**, put P back at its
   pre-step value, and compute global-class CE on an old-training batch
   (`scale=20`). Differentiate only with respect to current P B.
4. Let `d` be the actual P SGD displacement and `g_old` that old gradient.
   If their dot product is positive, remove the component of `d` along
   `g_old`: `d_candidate = d - max(dot(g_old,d),0)/norm(g_old)^2 * g_old`.
   Projection uses the concatenated P B space across layers, not arbitrary
   dense QKV coordinates. A, masks and trainable parameter count are unchanged.
5. Re-run the actual dynamic-gate forward. Accept only if selector old CE
   does not increase (absolute numerical tolerance1e-6) and current-batch
   new CosFace loss decreases (more than1e-7), relative to **no P step with
   the same post-SGD S/heads**. Otherwise retain the exact original P SGD.

This removes a first-order harmful component, not a guaranteed optimum.
P B step norm can shrink: a gain cannot be attributed exclusively to direction
without a later matched-size control. Effective masked QKV step norms need
not be matched. Old protection is local to the selector batch and P step;
S/head updates may still harm Old. New may learn less than the raw SGD step.
Momentum buffers remain raw SGD accumulators, as in the earlier direction
screen; this is a post-SGD displacement filter.

## Data and diagnostic isolation

At each incremental task, select the first eight training images per old
class in dataset order. Even positions (four/class) form the old gradient
selector; odd positions (four/class) form a disjoint read-only old probe.
Four current-class training images/class form the read-only new probe.
All use deterministic test **transforms** but `source='train'`. Selector
batches shuffle with a private generator; nothing consumes training RNG.
Only image paths/subset indices are retained, not feature tensors or copied
images. Reloading a batch uses the current extractor, avoiding stale features.
Pools and iterators are released after LoRA training, before CA/inference.

Probes are disjoint from the old selector but **not unseen holdout data**:
old images could have trained earlier tasks and current probes could occur
in ordinary current-task training. They never select or accept a direction.
Only normal all-seen-class test evaluation measures reported performance.

Logs:

- `POldGradientOracleData`: privilege, pool sizes, index hash and split rules.
- `POldGradientOracleStep`: first scheduled batch each epoch; no-P/raw-SGD/
  projected/selected selector losses, local accuracy and P B step norms.
- `POldGradientOracleSummary`: sampled/conflicting/applied counts per epoch.
- `POldGradientOracleProbe`: epochs1/10/20; raw-SGD vs selected on the SAME
  independent probe images and all seen classes, including margin/errors.

RNG, per-module modes and original gradients are preserved by auxiliary
forwards; temporary parameter swaps are restored even when a probe fails.
There are no new training-time input validation guards/assertions.

## Run from the existing project

```bash
bash scripts/9_30_imgr10_p_old_gradient_oracle_5090.sh --dry-run
bash scripts/9_30_imgr10_p_old_gradient_oracle_5090.sh --smoke
bash scripts/9_30_imgr10_p_old_gradient_oracle_5090.sh
```

The default command runs one Task0-1, one-epoch GPU smoke, then one complete
Task0-2 candidate if smoke exits successfully. Smoke scores are not formal
results. Dataset/pretrained paths and device come from the machine-local
JSON. Weight saving remains off. Logs have unique run/smoke directories.

Earlier 5090 anchor2.5 T3 took roughly55min under shared load. This Oracle
adds forward/backward work; full GPU timing is unmeasured. Allow roughly1-2h
including smoke, longer under heavier sharing, and revise from first epochs.

## Interpretation

Check completion, matched recipe/Task0, and nonzero applied counts separately
from performance. Small selector gains alone do not show generalization.
Seek Old/New joint improvement on independent probes and reported T1/T2,
with Average/Last/Forgetting not worse. A result trading New for Old is not
the hoped-for breakthrough. Zero acceptance or no gain only rejects this
sampled projection/pool setting, not all possible privileged update rules.
Do not interpret a positive Oracle as evidence that its benefit can already
be recovered without old images. No extra seeds/T10 sweep is in this queue.
