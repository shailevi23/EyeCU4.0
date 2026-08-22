# B2 preflight record — Tesla T4, 2026-08-16

Execution evidence for the run below. The predeclared design lives in
`experiment_B2_spec.md` and is **deliberately not edited by this file** — the
spec is what was frozen before results, this is what happened when it was run.

```
run          b2_20260816T082543Z
git HEAD     c5ce3e23ac382937c0cd36fb39775fa7c4cb699e
GPU          Tesla T4, 14.56 GiB
donor        yolo26s.pt  sha256 646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
output       /content/drive/MyDrive/EyeCU_B2_runs/b2_20260816T082543Z
```

No model had been trained and no experimental result had been observed at the
time of the repair described below.

## The memory probe was defective and was repaired before either arm ran

The first version of the memory-preflight cell raised before producing any
measurement:

```
ValueError: outputs must be a Tensor or an iterable of Tensors
```

**Cause.** YOLO26's training-mode forward output is nested. The probe scanned
only the top level for tensors, so the synthetic loss collapsed to a plain
`int` and `GradScaler.scale()` rejected it.

**Repair.** The probe now recurses through nested `dict`/`list`/`tuple`
outputs, keeps floating-point tensors that require grad, builds a scalar from
their squared means, runs an AMP forward + backward, and records peak CUDA
allocation. It raises rather than continuing if no differentiable tensor is
found, so the same class of defect cannot silently return a bogus measurement.

**Scope.** Only the probe implementation changed. Untouched: model
architecture, dataset, initialization, training hyperparameters, evaluation,
thresholds, experimental arms, success criteria, and the B2 hypothesis.

**Why this is a preflight repair, not a post-result change.** The defect
prevented the probe from producing a number at all, and it was corrected
before either arm was trained. Nothing about the experiment's outcome was
known — or knowable — when the fix was made. The scientific content of
`experiment_B2_spec.md` is therefore unaffected and was not rewritten.

## What the probe is, and is not

It is an **indicative activation-memory preflight**. Its only job is to decide
whether one frozen batch can serve both arms.

It is **not** the real YOLO detection loss, not a training result, not an
evaluation result, and not a statement about detector performance. Its peak
figures must never be quoted as B2 evidence.

## Measurement — batch=3, imgsz 960, AMP, forward + backward

| arm | batch 3 | indicative peak |
|---|---|---|
| matched baseline | **FITS** | 3.79 GiB |
| B2 width-matched P2 | **FITS** | 6.01 GiB |

B2's peak is 1.59× the baseline's, against a 1.70× compute ratio and a 4.05×
prediction-cell count — consistent with activations, not weights, being the
cost of the added stride-4 level. Both sit well inside 14.56 GiB.

## Batch decision — CLOSED

The predeclared rule (`both fit -> 3 for both`) resolves cleanly:

```
FROZEN_BATCH            = 3      for BOTH arms
accumulate              = 21     round(nbs 64 / batch 3)
effective weight decay  = 0.0004921875    = 0.0005 * 3 * 21 / 64
```

No reduction is required and none is authorised. Do not revisit batch size
unless the real trainer raises an actual CUDA OOM.

### The fallback guard stays armed

`ultralytics/engine/trainer.py` 8.4.116 lines 511–535 still halves batch on a
first-epoch OOM — `3 // 2 == 1`, not 2 — rebuilding the optimizer and
scheduler and continuing on a warning. After each real run, `args.yaml` is read
back and asserted to record `batch = 3`, and the log is checked for
`Reducing to batch`. **A run that fell back is INVALID: stop, report it, and do
not analyse it as B2 evidence.**

## Frozen configuration for both arms

```
dataset          823 EyeCU TRAIN / 208 EyeCU VAL, no Stage-A data
epochs           80          patience         0
imgsz            960         batch            3
optimizer        AdamW       lr0              0.00125
momentum         0.9         weight_decay     0.0005
warmup_bias_lr   0.0         seed             0
deterministic    True        accumulate       21
effective wd     0.0004921875
```

## Weight transfer — measured against the official donor

```
matched baseline   99.969%   total parameters
B2                 58.2366%  total parameters
B2 backbone         100%     hard gate, passed
B2 Detect head        0%     expected, never a failure
```

## Arms and order

```
1. MATCHED BASELINE   stock YOLO26s @960          yolo26s.pt -> baseline
2. B2                 width-matched P2 @960       yolo26s.pt -> B2
```

Both initialize independently from the same donor file. **B2 is never
initialized from the trained baseline.** Nothing changes between the runs
except the model topology.

Primary readout: **FINAL EPOCH / `last.pt`**. Secondary: `best.pt`.

## Status

Memory preflight **CLOSED**. Both arms ready. Experiment configuration
unchanged. EyeCU sealed TEST not accessed.
