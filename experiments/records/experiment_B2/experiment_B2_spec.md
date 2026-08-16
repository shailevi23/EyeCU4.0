# Experiment B2 — width-matched P2 detection level

> Predeclared plan for the B2/P2 architecture experiment, written before any B2
> model was trained and before any B2 result was seen. Every measured figure
> below was derived on CPU from ultralytics 8.4.116 in this repository, not
> quoted from an earlier document.
>
> _Written 2026-08-16. Governing brief: CLAUDE_B2_P2_RESEARCH_AND_EXECUTION_BRIEF.md_

## Research question

Does adding a stride-4 (P2) detection level to YOLO26s @960 improve
small-football detection, when data, initialization donor, training recipe,
evaluation path, thresholds and environment are held constant?

This is a controlled architecture experiment. It is not a dataset experiment,
not a threshold study, and not an architecture search.

## Two arms

Both arms are trained fresh from the same COCO donor. Neither is initialized
from the other, and neither is initialized from `best_A_960.pt`,
`best_C_960.pt`, `eyecu_football_v1.pt` or any Stage-A weight.

```
yolo26s.pt  ->  MATCHED BASELINE
yolo26s.pt  ->  B2
```

### MATCHED BASELINE

Stock `yolo26s.yaml` (ultralytics 8.4.116 `cfg/models/26/yolo26.yaml`, scale s),
`nc=4`, `end2end=True`, `reg_max=1`.

### B2

`models/yolo26s-p2-widthmatched.yaml` — the stock `yolo26-p2.yaml` with one
line changed.

**Why a matched baseline rather than the historical A@960.** A was trained
under `optimizer=auto`, `patience=20`, early-stopped at epoch 65, and its
published numbers come from `best.pt`. B2 uses the pinned recipe below with
`patience=0` and a final-epoch readout. Comparing B2 to A would confound the
architecture with the training policy and the checkpoint-selection rule. The
historical A@960 and B@1280 figures remain secondary reference points.

**A B2 result that beats historical A but not the matched baseline is not
evidence that P2 caused anything.**

## The architectural change

One line, in the P2/4-xsmall branch (head index 8, model layer 19):

```yaml
- [-1, 2, C3k2, [128, True]]      # stock
- [-1, 2, C3k2, [256, True]]      # B2
```

At width scale 0.5 this takes the P2 branch output from 64 to 128 channels.

**Why the width match is mandatory.** Ultralytics derives every Detect branch
width from `ch[0]`, the first input. Stock P2 presents a 64-channel branch
first, which collapses `c3 = max(ch[0], min(nc,100))` from 128→64 and
`c2 = max(16, ch[0]//4, reg_max*4)` from 32→16 **at every level**, including
the P3/P4/P5 levels that serve player, goalkeeper and referee. Stock P2 is not
"add P2"; it is "add P2 and narrow the whole head" — two variables and a likely
human-class regression. The measured signature is that stock P2 has *fewer*
parameters than the baseline despite adding a detection level.

## Measured topology — derived on CPU, ultralytics 8.4.116

| | baseline | stock P2 (rejected) | **B2** |
|---|---|---|---|
| strides | 8/16/32 | 4/8/16/32 | 4/8/16/32 |
| Detect input channels | [128,256,512] | [64,128,256,512] | **[128,128,256,512]** |
| box branch width | 32 | 16 ✗ | **32** ✓ |
| cls branch width | 128 | 64 ✗ | **128** ✓ |
| params (nc=4) | 9,950,960 | 9,665,024 | **10,389,376** |
| ratio vs baseline | — | 0.971× | **1.044×** (+438,416) |
| GFLOPs @960 | 51.9 | 60.6 | **88.2 (1.70×)** |
| head params | 934,960 | 482,496 | 1,100,864 |
| prediction cells | 10,710 | 43,350 | **43,350 (4.05×)** |

Prediction cells are counted at EyeCU's real geometry: source 640×360
letterboxed to 960×544. P2 240×136 = 32,640, P3 120×68 = 8,160,
P4 60×34 = 2,040, P5 30×17 = 510.

P2 activation: **136×240×128 = 4,177,920 elements per image**, ~8.4 MB at fp16,
recurring through every tensor in the P2 branch and its concatenations. The
memory cost of B2 is activations, not weights.

## Pre-training architecture assertions — fail closed

```
strides                == [4, 8, 16, 32]
Detect input channels  == [128, 128, 256, 512]
box branch width       == 32
cls branch width       == 128
params (nc=4)          == 10,389,376
GFLOPs @960            == 88.2 (+/- 0.1)
prediction cells @960x544 == 43,350
```

The parameter assertion is not redundant with the width assertions. Ultralytics
infers the model scale from the **filename stem** — a yaml copied to a name
without the `s` in it silently builds at scale `n` (2,617,568 params) while
every stride and channel-count check still looks plausible. The parameter count
is what catches it.

## Dataset — frozen

```
TRAIN  data/dataset_baseline/images/train   823 images
VAL    data/dataset_baseline/images/val     208 images
yaml   data/dataset_baseline/football.yaml  (unchanged)
```

Not used: the 1,030-image Stage-A set, Stage-A ACTIVE or NON_ACTIVE additions,
keremberke source VALID, keremberke source TEST, EyeCU sealed TEST.

No new annotations. No data collection. B2 does not touch Stage-A artifacts.

Because the data differs from Stage-A, **B2 arms are not comparable to Stage-A
arms.** That is intended: B2 isolates architecture.

## Initialization

Official COCO-pretrained `yolo26s.pt`, obtained from the Ultralytics assets
release and hashed before use.

```
sha256  646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
bytes   20,422,725
source  github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
```

This is byte-identical to the file Ultralytics downloaded during the Stage-A
run, independently verified.

## Weight transfer — measured against the official donor

Matching is by state-dict key **and** tensor shape, mirroring Ultralytics'
`intersect_dicts`. Measured by `tools/verify_weight_transfer.py`:

| | tensors | parameters |
|---|---|---|
| **MATCHED BASELINE** | 696/708 = 98.31% | 9,985,962/9,989,058 = **99.97%** |
| backbone (0–10) | 240/240 | 100.00% |
| neck | 228/228 | 100.00% |
| Detect head | 228/240 | 99.67% |
| **B2** | 360/902 = 39.91% | 6,075,238/10,431,985 = **58.2366%** |
| backbone (0–10) | 240/240 | **100.00%** |
| neck + new P2 path | 120/342 | 15.97% |
| Detect head | 0/320 | **0.00%** |

Tensor percentage understates transfer badly — the head contributes many small
tensors while the backbone holds a few very large convolutions. **The parameter
figure is the meaningful one, and the module breakdown is more informative than
either.**

### Transfer gate

```
backbone parameter transfer == 100.00%      HARD. Below this, STOP.
B2 total parameter transfer == 58.2366%     frozen; notebook must reproduce
Detect head transfer        == 0.00%        EXPECTED, never a failure
```

Backbone is the hard gate because layers 0–10 are topology-independent between
`yolo26` and `yolo26-p2`: anything under 100% means the donor, the scale or the
yaml is wrong, not merely suboptimal.

### Disclosed asymmetry — the two arms do not inherit equally

The baseline inherits **99.97%** of its parameters; B2 inherits **58.24%**.
The donor is identical and the loading path is identical, but adding a
detection level necessarily re-initialises the neck and the entire head.

This is **inherent to the research question and cannot be engineered away** —
there is no version of "add P2" that preserves the head.

**Ruling, 2026-08-16: proceed with the measured transfer.** Do not add a longer
schedule, do not invent a custom head initialization, and do not initialize B2
from the matched baseline. The asymmetry is accepted as a property of the
question, which fixes what B2 is:

> **B2 is an ADOPTION TEST UNDER A FIXED 80-EPOCH BUDGET.**
>
> Can width-matched P2, using the realistic transfer available from official
> `yolo26s.pt`, outperform the matched standard YOLO26s under the same
> 80-epoch budget?

This determines how each outcome may be written up.

A **positive** result is unaffected by the asymmetry — a win achieved from a
strictly worse starting point.

A **negative** result supports exactly one conclusion:

```
B2 DID NOT JUSTIFY ADOPTION UNDER THE FROZEN TRAINING BUDGET.
```

It does **not** support "P2 cannot work", and that claim must not be made.

A longer-schedule follow-up may be proposed **only as a new experiment**, and
only if the B2 learning curves show concrete evidence that the model was still
materially improving at epoch 80. Absent that evidence in `results.csv`, the
branch closes.

## Training recipe — identical for both arms

```
epochs           = 80          patience         = 0
imgsz            = 960         optimizer        = AdamW
lr0              = 0.00125     momentum         = 0.9
weight_decay     = 0.0005      warmup_bias_lr   = 0.0
lrf              = 0.01        warmup_epochs    = 3.0
warmup_momentum  = 0.8         nbs              = 64
seed             = 0           deterministic    = True
amp              = True        mosaic           = 1.0
close_mosaic     = 10          hsv_h/s/v        = 0.015 / 0.7 / 0.4
fliplr           = 0.5         scale            = 0.5
translate        = 0.1         erasing          = 0.4
```

Forbidden: `optimizer=auto`, `batch=-1`, hyperparameter tuning, architecture
search, alternate imgsz, alternate augmentation, alternate seed, early
stopping, different recipes between arms.

**Primary readout: FINAL EPOCH.** `best.pt` is retained as a secondary artifact
and must not be substituted for the primary result.

## Batch size and the OOM fallback

### The trainer will silently rewrite the experiment

Verified in `ultralytics/engine/trainer.py` at 8.4.116, lines 511–535. On a
first-epoch CUDA OOM, single-GPU, up to 3 retries:

```python
self.args.batch = self.batch_size = max(self.batch_size // 2, 1)
...
self._build_train_pipeline()   # rebuilds dataloaders, optimizer, scheduler
```

`3 // 2 == 1`. A single OOM takes batch from **3 straight to 1** — not to 2 —
and rebuilds the optimizer and scheduler with it, which silently changes
`accumulate` and the effective weight decay. It logs a warning and continues.

For a controlled experiment this is unacceptable. It must be prevented, and
detected if it occurs anyway.

### Policy

1. Run an explicit memory preflight on both topologies at batch=3, imgsz 960,
   AMP, forward + backward, before either real run.
2. **If both fit: batch = 3 for both arms. If B2 does not fit: batch = 2 for
   both arms.** Never baseline at 3 and B2 at 2 — that is a second variable.
3. Do not drop to batch=1 without explicit reconsideration; it degrades
   BatchNorm statistics materially.
4. After each run, assert the batch recorded in `args.yaml` equals the frozen
   batch, and scan the log for `Reducing to batch`. If a fallback occurred the
   run is **invalid** and must be reported and discarded, not analysed.

### Batch-dependent weight decay — record whichever applies

| | batch 3 | batch 2 |
|---|---|---|
| accumulate = round(nbs/batch) | 21 | 32 |
| effective wd = wd·batch·accumulate/nbs | 0.0004921875 | 0.0005000000 |
| optimizer steps/epoch ≈ | 13.1 | 12.9 |
| iterations = ceil(823/max(batch,nbs))·80 | 1040 | 1040 |

Both arms share whichever batch is frozen, so the architecture comparison
stays defensible either way. The figure is recorded, never hidden.

## Evaluation — frozen paths, three layers

Production path unchanged throughout: `ONE_TO_ONE_END2END`, candidate threshold
0.10, high threshold 0.25, ball dedupe IoU 0.70. **No threshold retuning after
seeing results.**

### A. Standard 208-image EyeCU VAL

Per class — player, goalkeeper, referee, ball — precision, recall, mAP50,
mAP50-95, plus all-class metrics.

Limitation carried forward: this VAL has no useful multi-ball structure, so
ball precision changes must not be over-interpreted. B2 does not change the
ontology, so this bites less here than in Stage-A, but it still bounds reading.

### B. 104-frame temporal ball benchmark

`tools/eval_temporal_val.py`, unaltered. `MATCH_IOU = 0.5`, pre-existing.

Historical reference: A raw ball recall **0.5714** (44/77); coverage with the
frozen selector **0.6234** (48/77).

### C. Small-ball diagnostics — reported separately, never pooled

- **women_1 w1** — 6 GT, ~7–11 px, historical A recall ~0.167. The regime the
  P2 hypothesis targets.
- **youth w1** — 12 GT, ~18.9–24.2 px, historical A recall 0.000. The audit
  states this is **not primarily a resolution problem** (contact, occlusion,
  clutter). P2 is not expected to fix it and failure here is not evidence
  against P2.
- **Zero-proposal diagnostic** — of 17 misses across those windows, 12 had no
  usable proposal even at confidence 0.01. Re-run identically and report how
  many of those 12 now receive a usable B2 proposal.

Layer C is **diagnostic**. It carries no hard threshold and none may be
invented after results are seen.

## Predeclared success criteria

Preserved verbatim from `docs/research/ball_architecture_audit.md` §5. All
must hold:

```
1. temporal-val raw ball recall  >  0.5714
2. coverage with frozen selector >= 0.7143
3. 208-VAL player recall         >= 0.90
4. 208-VAL referee recall        >= 0.75
5. measured ms/frame             <= 263.3
```

The audit prose says "all four" while listing five thresholds; it counts
player and referee as one clause. The threshold set is identical either way.

The qualifier *"with the gain concentrated in the small-ball windows"* was
never assigned a number. It is **NOT PREDECLARED** and is reported as a
diagnostic. No threshold may be attached to it now.

### Criterion 5 — PREDECLARED ESTIMATED RUNTIME CEILING = 263.3 ms/frame

**Ruling, 2026-08-16.** `263.3` is **not** a measured historical B@1280 runtime
and must never be described as one. It appears exactly once in the repository —
in the audit line asserting it — and in no results or evidence file. The
repository holds two distinct timing conventions:

| convention | A@960 | B@1280 | C@960 |
|---|---|---|---|
| detector inference | 57.7 FPS = 17.3 ms | 32.6 FPS = 30.7 ms | — |
| pipeline ms/frame | 154.7 | **not recorded** | 164.4 |

`154.7 × 1.702 = 263.3`. The value is therefore **A's pipeline ms/frame scaled
by the B2 compute ratio** — an estimated ceiling, not evidence.

It is **retained** because it was specified before any B2 result was seen, and
it is **frozen**: the number does not change after results.

```
PREDECLARED ESTIMATED RUNTIME CEILING = 263.3 ms/frame
```

### Runtime is decided by the matched comparison, not by the ceiling

The primary runtime evidence is **MATCHED BASELINE vs B2**, measured on the
same T4, in the same runtime session, under the same timing protocol, at the
same input size, through the same pipeline scope. Report all four:

```
baseline measured ms/frame
B2 measured ms/frame
B2 / baseline runtime ratio
estimated 263.3 ceiling            PASS / FAIL
```

The ceiling is a weak guard against gross regression. The ratio is the finding.

## Matched-control interpretation layer

Beyond the historical criteria, report `MATCHED BASELINE -> B2` deltas for:
raw temporal ball recall, selector coverage, 208-VAL ball metrics, player
recall, referee recall, inference time, the women_1 small-ball diagnostic, and
the zero-proposal diagnostic.

The matched-control comparison is the causal one. The historical criteria are
the adoption bar.

## Guardrails

```
player recall   >= 0.90     HARD
referee recall  >= 0.75     HARD
ms/frame        <= 263.3    HARD (see disclosure above)
```

Goalkeeper metrics are reported; no hard threshold was historically frozen for
goalkeeper and none is invented here.

If a human guardrail fails, **B2 fails even if ball recall improves.** The
width match exists precisely to protect these classes, and its failure would
mean the protection did not work. The system does not trade player or referee
detection for a small ball gain without an explicit, separate redesign
decision.

## What a positive result would and would not establish

**Would support:** finer spatial feature resolution helps the current
small-ball proposal/detection bottleneck under EyeCU's present data and
evaluation conditions. Sufficient to continue with P2 as detector architecture.

**Would not establish:** that the ball problem is solved; that all misses are
resolution-related; that every camera regime benefits; that contact/occlusion
failures are addressed; that the dataset is complete; that the temporal
selector matters less.

## What a negative result would mean

Classify the failure before acting on it. Do not respond by collecting data,
adding attention modules, trying more heads, raising imgsz, or tuning
thresholds.

A clean negative — B2 no better than the matched baseline on ball and temporal
diagnostics, guardrails intact — closes the resolution/P2 branch and moves the
project to the next system-level bottleneck in the roadmap (calibration /
geometry / downstream reasoning).

A guardrail failure is a different outcome from a ball-null and must be
reported as such.

## Sealed TEST

`EyeCU sealed TEST` is not accessed for training, validation, debugging,
architecture selection, threshold tuning or final comparison. B2 is
development-stage architecture research.

## Result persistence

Per arm, never overwriting a previous run: `last.pt`, `best.pt`, `results.csv`,
`args.yaml`, plots, logs, git HEAD, model yaml, parameter/FLOP report, donor
SHA256, weight-transfer report, dataset identity and counts, training config,
the batch decision and its effective weight decay, GPU, CUDA, torch and
Ultralytics versions, timing results, temporal evaluation output, small-ball
diagnostics.
