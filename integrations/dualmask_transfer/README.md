# DualMask → CL-LoRA / SD-LoRA (experimental)

Independent integration; no edits to LoDA training, masks, configs or main branch.
Run **from the lora_base root**. Data path comes exclusively from the current
machine's `exps/dlora/imgr10.json`. Images are never moved or split.

## Scope and exact mapping

This is a **fixed-policy port of DualMask's update-control framework**, NOT an
exact reproduction of all LoDA components. Host losses, ranks (10), optimizers,
learning rates, classifiers, Q/V targets and magnitude/block-weight mechanisms
stay with each upstream. No LoDA CA, anchor loss, competence controller, adaptive
rank, extra S/P parameters, router, oracle or functional merge filter is added.
The framework is parameterized in weight space, not by averaging LoRA factors.

- Immutable pretrained Q/V projection → top-32 SVD row/column importance, using
  singular values as in LoDA's hard-SVD formula. Unlike LoDA's joint QKV tensor,
  each host Q/V projection is scored separately.
- Protect mask: 50% **importance energy**; plastic mask is its complement.
- Conflict score: normalized importance × normalized absolute current update.
  Conflict selection: 50% score energy plus Top-10% coordinate floor.
- Protection strength and conflict suppression strength are both fixed at 0.5.
  Selection masks are detached; factor gradients still flow through the update.
- Task0: upstream unmasked training, unchanged; no added anchor loss.

CL-LoRA: original shared adapters are in the first six layers, private adapters
in the last six. For shared layers, retain an effective task-start kernel and a
raw task-start kernel. Apply soft protection and conflict suppression only to
`raw_now - raw_start`, then add `effective_start`. At task boundaries consolidate
once, before upstream teacher copies. Private adapters use plastic restriction
and conflict suppression. Upstream block weights remain outside this gate.

SD-LoRA: no native S/P branches. The new direction is partitioned by complementary
protect/plastic masks, with **tied original factors**: protected contribution is
softly attenuated; plastic contribution remains available; conflict suppression
applies to the sum. Old directions retain their saved gate, and the original
learnable scalar magnitudes and raw-factor norm denominators remain unchanged.
Old normalized masked kernels are cached once per task; their scalar-weighted sum
is evaluated with the current kernel. This avoids sorting all historical masks
every batch. It introduces dense-kernel storage/compute; it is not free LoRA.

The official training-loop evaluation is supported. Upstream SD's separate
`eval=True`/`load_eval_vit` API is NOT supported for plugin checkpoints; it must
not be used for reporting this experiment. Official code already has separate
behavior in that API. The runner never invokes it.

## Provenance / protocol

Official sources (MIT, retained in their downloaded repos):

- CL-LoRA: https://github.com/JiangpengHe/CL-LoRA
  `df4e74efe589ca1aa872d3843315ab6851192cf3`, `exps/inr.json`.
- SD-LoRA: https://github.com/WuYichen-97/SD-LoRA-CL
  `8bacded6eb44786db071f66fb90a87dd660d94ea`, `exps/sdlora_inr.json`.

Sources download under ignored `.external/`. Each run checks the exact revision
and tracked-file cleanliness. No patches are written into official sources.
Runtime hooks modify only the experiment process. All outputs and SD factor files
are isolated in unique `logs/dualmask_transfer/...` directories; no old checkpoint
is reused. Config, upstream SHA, dependency versions and dataset path/label
manifest digest are recorded. The manifest is NOT a byte-content audit.

Common protocol overrides: seed1993, ImageNet-R 200 classes, 20 initial + 20/class
increment = T10. CL's provided example is 5+5 (T40); both CL arms are changed to
20+20. SD's example seed1995 is changed to1993. All four full runs start fresh,
20 epochs/task, using each host's own training logic (not our CA5 protocol).
Do not compare their absolute accuracy to the historical LoDA 87.144 target as
if backbones/classifiers/training settings were matched.

## Environments and launch

No automatic pip/conda changes are made. Use the existing environment only if
imports and GPU smoke pass. Upstream CL lists timm0.6.12; SD lists timm1.0.9 and
torch2.4.1. Two separate environments may be required. Do not overwrite a working
LoDA environment. `CL_PYTHON` and `SD_PYTHON` may be absolute interpreter paths.
Both need Torch/Torchvision, timm, numpy, scipy, sklearn, tqdm, einops and easydict.
Model weights may download on first use, independently in each environment.

```bash
# Short preflight: both methods, both arms, T0–1, one epoch/task.
env SMOKE_ONLY=1 bash scripts/9_19_dualmask_transfer_3090.sh

# Full queue; repeats the smoke checks before full training.
lrun scripts/9_19_dualmask_transfer_3090.sh ./logs/9_19_dualmask_transfer_3090.log
```

Queue: CL off/on smoke → CL off/on T10 → SD off/on smoke → SD off/on T10.
Four full experiments, four short smoke experiments; no 3-seed sweep.
A smoke failure skips that method's full pair, continues the other method, and
returns nonzero. A full-run failure is recorded; remaining runs continue.
No guaranteed 10/12-hour runtime is claimed: these hosts have different evaluation
costs from LoDA and have not been timed on the user's 3090. There is no timed kill.

## Verification and interpretation

Local unit tests exercise official Adapter and SD wrapper class bodies on tiny
CPU tensors. Tests cover task0 exact equality, RNG preservation, shared
consolidation, teacher immutability, private plastic support, save/reload, and
SD identity-gate output/gradient equivalence. They do NOT establish CUDA end-to-end
compatibility or accuracy. Mandatory server smoke must finish before full pairs.

Compare **within each host**, off vs on, for Average/Last, old/new accuracy,
Forgetting and runtime; inspect upstream `CNN` grouped accuracy and curves.
An improvement at seed1993 is preliminary, not evidence of general transfer.
Keep main unchanged until matched multi-seed results support the framework.
