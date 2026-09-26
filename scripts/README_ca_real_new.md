# CA real-new feature ablation

Only new-class CA input changes. Default `ca_real_new_features=false` preserves
legacy Gaussian CA. Candidate `true` caches the vectors already extracted from
current training images (`source=train, mode=test`) for class statistics after
merge. It does not read test images, re-extract old images, or add a backbone pass.
The cache is CPU float32, current task only, and cleared after CA. Task0/CA-off do
not cache. Approximate peak cache bytes: current training count × feature dim ×4.

CA still optimizes all seen classifier heads only, for 5 epochs, 256 samples per
class, with unchanged optimizer, logits normalization and loss. Current real
vectors are uniformly sampled WITH replacement using a private CPU generator
seeded by torch.initial_seed()+task. Old Gaussian draws and global shuffle RNG
remain unchanged: candidate deliberately still draws and discards the new-class
Gaussian samples. This preserves the prior RNG trajectory, at a small temporary
sampling overhead; it is not an efficiency improvement claim.

Current task mean decay is exactly1 under the equal-size T10 protocol. Real
features are used unmodified before the existing classifier normalization.
Old statistics can still be stale; this does not fix drift or guarantee Old safety.

## Runs

Each GPU runs a fresh baseline then candidate, ImageNet-R T10 seed1993. Fixed:
anchor5 Task0-only,20 epochs,CA5,math-SDPA,layer dual gates,P LR.02,reg.01.
Both competition variants and P direction correction are off. No JSON data paths
are changed. Scripts log effective config/revision and stage-audit CSVs.

3090: `bash scripts/9_26_imgr10_ca_real_new_3090.sh`

5090: `bash scripts/9_26_imgr10_ca_real_new_5090.sh`

Add `--smoke` for only the paired one-epoch Task0–1 smoke, or `--dry-run` to inspect
full commands. Full launch directly runs the two full experiments; smoke is optional.
There is no runtime accuracy-equality assertion or automatic test/preflight gate.
Run tests separately with `python -m unittest test.test_ca_real_new test.test_stage_audit`.
Local tests separately verify identical default weights/RNG versus e78251c, paired
old samples/global RNG, retained class counts, cache lifetime, and unchanged backbone.

These are independent matched full runs, not an in-memory checkpoint branch.
Do not claim later-stage classifier weights remain identical: prior CA has changed
them intentionally. No automatic checkpoint choice or test-feedback selection.
If post-merge features or Task0 unexpectedly differ, investigate before attributing
changes to CA. Data/software/local changes must match within each machine.

Recent timings suggest two full runs plus smoke:3090 ~3h,5090 ~4–5h. Runtime and
real-feature benefits are not verified until server runs. No remote launch included.

## Decision

Compare against the fresh SAME-machine baseline: Average/Last, final and stage-mean
Old/New, forgetting, old→new/new→old AND within-partition errors. Improvement only in
New at Old's expense is not a success. Keep no claim of better representations,
bidirectional protection, or Gaussian-model failure until evidence supports it.
This is a fixed exploratory ablation, not holdout-tuned: no mixture/weight sweeps,
no selecting epochs with test accuracy. Default remains unchanged regardless of
this exploratory test result; later hyperparameter selection needs training holdout.

Local verification: 19 focused tests passed; 197 tests passed in the lightweight
suite. Two dependency-heavy modules (test_ca_diagnostics/test_task0_margin_screen)
were excluded because local easydict is unavailable. CPU CA integration uses the
actual extracted method, not a reimplementation; CUDA/data smoke is pending.
