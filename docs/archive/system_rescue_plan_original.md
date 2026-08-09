> **ARCHIVED — superseded.**
> The original fact-checked system-rescue plan. Most of it has been executed or
> overtaken by measurement. The still-useful research and architectural
> rationale were distilled into
> [../research/system_rescue_research.md](../research/system_rescue_research.md),
> which also tracks the status of every recommendation below. Remaining work is
> in [../coursework/COURSEWORK_PLAN.md](../coursework/COURSEWORK_PLAN.md).

---

# EyeCU 4.0 — Fact-Checked System Rescue TODO for Claude

> **Role:** Act as a senior computer-vision / sports-video engineer.
>
> **Goal:** Spend the minimum engineering + annotation time needed to materially improve EyeCU as a football-video analysis system, then move on to the next phases.
>
> **Critical constraint:** The frozen TEST set must remain untouched for model/system selection. Use TRAIN + VAL only until the final configuration is frozen.

---

## 0. Executive decision

The current production detector candidate remains:

- **A = YOLO26s @ 960**
- A is currently the speed/production winner.
- B @1280 produced a real but modest ball-resolution benefit, but not enough to justify its ~1.73× inference cost under the current setup.
- Do **not** state “resolution is not the bottleneck.” The supported conclusion is narrower:

> Increasing input resolution from 960 to 1280 did not produce a sufficiently large or robust improvement in ball detection to justify its computational cost under the current training setup.

Do **not** run more A/B detector training yet.

The next experiment is a **video-system rescue ablation**, not another model-size experiment.

---

# 1. FACT CHECK — reference repository

Reference:

https://github.com/abdullahtarek/football_analysis

## 1.1 Training recipe — CONFIRMED

The public notebook:

`training/football_training_yolo_v5.ipynb`

downloads Roboflow project:

`football-players-detection-3zvbc`, version 1

and trains:

```bash
yolo task=detect mode=train model=yolov5x.pt data=... epochs=100 imgsz=640
```

Source:

https://github.com/abdullahtarek/football_analysis/blob/main/training/football_training_yolo_v5.ipynb

## 1.2 Dataset size — use precise wording

The Roboflow project exposes **372 source images**.

Dataset v1 contains **663 generated images**:

- 612 train
- 38 validation
- 13 test

v1 uses augmentation including:

- horizontal flip
- saturation changes
- brightness changes

Source:

https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1

## 1.3 “One source video” is NOT verified

Public repository + Roboflow metadata do **not** prove that all 372 source images came from exactly one video.

Therefore:

- do not write “trained on one video” as a verified fact;
- until provenance is proven, call it a **small, visually narrow football dataset**.

## 1.4 The demo pipeline uses substantial post-processing — CONFIRMED

`trackers/tracker.py` runs inference at:

```python
conf=0.1
```

It initializes:

```python
sv.ByteTrack()
```

It converts goalkeeper detections to player detections before tracking:

```python
if cls_names[class_id] == "goalkeeper":
    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
```

Source:

https://github.com/abdullahtarek/football_analysis/blob/main/trackers/tracker.py

**Do NOT copy this destructive merge into EyeCU.**

EyeCU must keep these semantic classes distinct end-to-end:

- player
- goalkeeper
- referee
- ball

A temporary class-agnostic **human association view** is allowed, but raw semantic classes must be preserved.

## 1.5 The reference repo does not actually ByteTrack the ball — CONFIRMED

Player/referee detections go through ByteTrack.

Ball detections are read directly from detector outputs and stored under a fixed ID:

```python
tracks["ball"][frame_num][1] = {"bbox": bbox}
```

Then `interpolate_ball_positions()`:

- creates a dataframe of ball boxes;
- interpolates missing values;
- applies `bfill()`.

`main.py` explicitly calls:

```python
tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
```

Sources:

https://github.com/abdullahtarek/football_analysis/blob/main/trackers/tracker.py

https://github.com/abdullahtarek/football_analysis/blob/main/main.py

Implication:

The reference demo can display smooth ball locations even when the detector did not hit the ball on every frame.

Its interpolation also has no explicit maximum-gap limit in that code, and `bfill()` can fabricate positions before the first observed ball.

EyeCU should use **bounded, provenance-aware temporal recovery only**.

## 1.6 The reference repo contains source-specific hard-coding — CONFIRMED

Perspective transform uses fixed image vertices:

```python
[[110, 1035],
 [265, 275],
 [910, 260],
 [1640, 915]]
```

Team assignment contains:

```python
if player_id == 91:
    team_id = 1
```

Sources:

https://github.com/abdullahtarek/football_analysis/blob/main/view_transformer/view_transformer.py

https://github.com/abdullahtarek/football_analysis/blob/main/team_assigner/team_assigner.py

Therefore the demo is **not** a clean generalization benchmark for EyeCU.

---

# 2. FACT CHECK — research and official guidance

## 2.1 Low-confidence detections can help tracking

ByteTrack’s core idea is to retain and associate low-score detections rather than throwing them all away, because some low-score boxes are real partially visible/occluded objects.

Paper:

https://arxiv.org/abs/2110.06864

Current Ultralytics ByteTrack defaults include:

```yaml
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.25
```

Source:

https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml

**Important limitation:** this supports the principle of low-confidence association for MOT. It does not prove ByteTrack itself is the right tracker for a tiny, fast, irregularly moving football.

For the ball, keep low-confidence proposals but evaluate a dedicated temporal selector.

## 2.2 Temporal information is a legitimate soccer-ball direction

Soccer-ball research has shown that frame history / spatio-temporal correlation can help during temporary disappearance and partial occlusion.

Source:

https://arxiv.org/abs/1909.02406

This supports temporal recovery.

It does **not** justify unlimited interpolation.

## 2.3 SAHI is legitimate for small-object detection, but not free

SAHI performs sliced inference by splitting a high-resolution image into overlapping regions and running detection on slices.

The original paper reports small-object AP gains on VisDrone/xView without requiring detector fine-tuning.

Sources:

https://arxiv.org/abs/2202.06934

https://docs.ultralytics.com/guides/sahi-tiled-inference

Caveats:

- evidence is not football-specific;
- multiple slice passes usually increase total latency/compute;
- use SAHI as an **ablation / conditional rescue**, not a guaranteed production win.

## 2.4 Dedicated small-ball treatment is scientifically plausible

FootAndBall is a football-specific detector using high-resolution/FPN features to improve ball detection.

Source:

https://arxiv.org/abs/1912.05445

This supports the general idea that the ball can deserve specialized treatment.

It does **not** justify migrating EyeCU to FootAndBall now.

## 2.5 Modern football analysis separates tracking and semantic identity subtasks

SoccerNet Game State Reconstruction treats the pipeline as linked subtasks such as:

- person detection
- re-identification/tracking
- role classification
- team affiliation
- jersey recognition
- pitch localization/camera calibration

Sources:

https://github.com/SoccerNet/sn-gamestate

https://arxiv.org/abs/2404.11335

This supports an EyeCU design where tracking association and semantic role classification are related but not forced to be identical operations.

## 2.6 YOLO26 official small-dataset guidance

Ultralytics currently recommends considering, for datasets below ~1,000 images:

```text
mosaic=0.5
mixup=0.0
copy_paste=0.0
lr0=0.001
epochs=50
patience=20
```

and optionally freezing backbone layers.

Source:

https://docs.ultralytics.com/guides/yolo26-training-recipe

Important:

When `optimizer="auto"`, manually supplied LR/momentum can be ignored.

Source:

https://docs.ultralytics.com/guides/finetuning-guide

Do not claim a manual `lr0` experiment unless the optimizer configuration actually honors it.

---

# 3. MAJOR METHODOLOGICAL CORRECTION

The current 208-image VAL set is excellent for **per-frame detector evaluation**.

It is **not sufficient by itself for temporal recovery or track-continuity evaluation**, because the dataset frames were sampled sparsely from video.

Do not treat sparse validation images several seconds apart as adjacent frames.

Before evaluating temporal logic, build a tiny fixed **continuous validation micro-benchmark** from VAL source videos only.

This is the highest-priority correction.

---

# 4. HARD CONSTRAINTS

1. Do **not** touch frozen TEST for:
   - thresholds
   - model selection
   - tracker tuning
   - rescue logic
   - SAHI settings
   - training decisions

2. Preserve four semantic classes end-to-end:
   - player
   - goalkeeper
   - referee
   - ball

3. A temporary `human` association representation may combine player/GK/referee for tracking, but:
   - never overwrite `raw_class`;
   - goalkeeper must remain a distinct final semantic class.

4. Preserve raw detector outputs.

5. Mark every ball state as one of:
   - observed
   - recovered_low_conf
   - interpolated_short_gap
   - unknown

6. Never fabricate a ball indefinitely:
   - no unlimited interpolation
   - no unconditional forward-fill
   - no unconditional backward-fill

7. Do not add VAL frames to TRAIN.

8. Hard-example mining must use TRAIN or new explicitly TRAIN-designated sources only.

9. Make new behavior reversible behind feature flags.

10. Measure before/after. Pretty rendered video is not evidence.

---

# 5. PRIORITY 0 — Freeze and audit baseline

Before modifying production behavior:

- [ ] confirm local path + checksum/hash for A `best.pt`
- [ ] record detector:
  - imgsz
  - global/per-class confidence handling
  - NMS
- [ ] record tracker:
  - algorithm
  - high/low/new thresholds
  - buffer
  - match threshold
- [ ] record:
  - `skip_frames`
  - source FPS
  - effective FPS
- [ ] ensure reports can distinguish:
  - raw detector output
  - tracker-associated output
  - temporally recovered output
- [ ] add one reproducible baseline command
- [ ] save baseline JSON per VAL match

## Add detector diagnostics on existing 208 VAL images

- [ ] standard per-class precision/recall
- [ ] **any-human recall**
- [ ] human-role confusion matrix
- [ ] ball precision/recall at production threshold
- [ ] per-match ball metrics

### any-human recall definition

For GT classes:

```text
player / goalkeeper / referee
```

count localization as successful if **any human-role prediction** matches the GT bbox at the fixed IoU threshold.

Purpose:

separate:

```text
human missed entirely
```

from:

```text
human localized but wrong semantic role
```

---

# 6. PRIORITY 1 — Build a tiny CONTINUOUS temporal VAL benchmark

Use ONLY:

- `austin_fc_vs__club_tijuana`
- `bayern_munich_3-1_chelsea`
- `women_1`
- `youth_premier_league`

## Target size

Approximately **80–120 continuous diagnostic frames total**.

Low-effort suggestion:

- 1–2 short windows/match
- ~4–6 seconds/window
- fixed 5 FPS diagnostic sampling, or another deterministic cadence
- preserve source frame number + timestamp

For the first temporal benchmark, manually label at least:

- ball bbox when visible
- explicit no-visible-ball state when appropriate

Do not use local model output as final GT.

## TODO

- [ ] implement `tools/build_temporal_val.py`
- [ ] select/document deterministic windows
- [ ] preserve source/timestamp provenance
- [ ] create annotation ZIP
- [ ] mark split as VAL-only
- [ ] add leakage guard preventing temporal-val from entering train/test

This benchmark is for **validation only**.

Do not train on it.

---

# 7. PRIORITY 2 — Human association without destroying roles

## Hypothesis

Some player/GK/referee instability is semantic jitter rather than localization failure.

A temporary class-agnostic human association may stabilize track continuity.

## Correct architecture

```text
RAW DETECTION
bbox
raw_class = player/GK/referee
raw_confidence

        ↓

ASSOCIATION VIEW
association_class = human

        ↓

TRACKER
track_id

        ↓

SEMANTIC LAYER
track_role = player/GK/referee
```

Never do:

```text
goalkeeper -> player permanently
```

## TODO

First audit:

- [ ] Is current association class-aware?
- [ ] Can class changes split an existing track?
- [ ] Can different human roles accidentally share/lose IDs?
- [ ] Is ball isolated from human tracking?

Then prototype:

- [ ] optional `human_class_agnostic_tracking`
- [ ] merge only player/GK/referee for association
- [ ] keep ball separate
- [ ] preserve raw class/confidence on each observation
- [ ] preserve distinct final goalkeeper output

## Conservative track-role smoothing

Prototype one simple method only:

- confidence-weighted vote OR EMA
- short rolling history
- minimum evidence before changing a stable role
- hysteresis against one-frame flips
- naturally resets on a new track

Do not permanently lock role forever.

Do not let an ID switch poison a long track.

## Evaluate

On existing 208 VAL images:

- raw role confusion
- smoothed role confusion where correspondence is available

On continuous VAL windows:

- unique IDs
- median/mean track length
- fragmentation proxy
- role flips/track
- player↔referee↔GK switches

Do **not** call these IDF1/HOTA without identity GT.

## Adoption rule

Keep only if:

- any-human localization does not worsen;
- role instability decreases;
- fragmentation does not worsen;
- result is not carried by only one match.

---

# 8. PRIORITY 3 — Ball low-confidence candidate study

Do not lower the threshold globally for every class.

Create a ball-specific candidate path.

Main detector stays:

```text
A @960
```

Run one small predeclared VAL-only threshold grid:

```text
0.05
0.10
0.15
0.20
0.25
```

For each threshold report:

- precision
- recall
- false positives/image
- per-match results

Purpose:

answer whether the model already produces useful low-confidence ball proposals that are currently discarded.

Freeze the candidate threshold after this small study.

No repeated threshold fishing.

---

# 9. PRIORITY 4 — Conservative temporal ball selector

Implement a separate module, e.g.:

```text
BallTemporalSelector
```

## State provenance

Every output must be explicitly labeled:

```text
observed
recovered_low_conf
interpolated_short_gap
unknown
```

## Candidate scoring may use

Keep v1 simple:

- detector confidence
- distance from predicted location
- recent velocity
- gap duration
- bbox-size plausibility
- optional reliable camera-motion compensation

A constant-velocity/Kalman-style predictor + gating is enough for the first ablation.

## Safety rules

- [ ] no history anchor -> no invented ball
- [ ] long gap -> unknown
- [ ] implausibly large jump -> reject/reset
- [ ] camera cut/discontinuity -> reset state
- [ ] no backward filling before first observation
- [ ] no indefinite forward filling
- [ ] provenance stored per box
- [ ] gap limit expressed in **seconds**, converted using real FPS/effective FPS

## Evaluate ONLY on continuous temporal VAL

Report:

- raw ball detector recall
- recovered-low-conf recall
- temporal trajectory coverage
- false recovery rate
- longest false recovery
- unknown-frame count
- per-match results

Do not report interpolation as raw detector recall.

---

# 10. PRIORITY 5 — Conditional ROI / SAHI ball rescue

Only do this if low-confidence + temporal rescue still leaves obvious gaps.

## 10A. ROI rescue — preferred first

Trigger only when:

- a ball trajectory was recently established;
- current full-frame A@960 has no accepted ball.

Run a second detector pass on a crop around the predicted ball location.

Requirements:

- configurable crop size
- ROI expands with gap
- crop clamps to frame
- reset after camera cut
- never use stale position after long gap

Measure:

- rescue activation rate
- GT balls recovered
- false rescues
- average end-to-end latency/FPS

## 10B. SAHI ablation

Run A + SAHI on the existing 208-image detector VAL.

Use only 1–2 sensible slicing configs.

Measure:

- ball precision
- ball recall
- ball mAP
- duplicate rate
- latency

If it helps substantially, consider SAHI only as **conditional rescue**.

Do not default to SAHI on every frame unless the measured gain clearly justifies latency.

---

# 11. PRIORITY 6 — Hard-100 active-learning package

Only after system diagnostics reveal remaining errors.

Target:

**~100–150 high-value TRAIN frames**, not another huge random batch.

## Source rule

Allowed:

- current TRAIN source videos
- new explicitly TRAIN-designated sources

Forbidden:

- VAL frames
- TEST frames

## Mine examples such as

### Ball

- A miss / teacher sees plausible ball
- low-confidence ball band
- A/B disagreement
- frames around known TRAIN ball detections
- blurred / tiny / airborne / partially occluded balls
- hard-negative false-ball triggers

### Human roles

- player↔referee disagreement
- player↔GK disagreement
- unusual kit colors
- known kit-confusion patterns
- closeups + wide shots
- underrepresented domains

## Diversity

Cap examples per source/window.

Do not allow one match to dominate.

## Export

Create:

```text
data/export/football_hard100_for_annotation.zip
```

Include:

- images
- optional draft labels
- provenance manifest
- source/match ID
- selection reason per frame

Manual correction mandatory.

---

# 12. PRIORITY 7 — One final detector retrain after Hard-100

If Hard-100 is completed:

## First retrain

Use:

```text
YOLO26s @960
```

Keep training recipe close to A.

Reason:

measure the data effect separately.

Do not simultaneously change:

- dataset
- optimizer
- LR
- mosaic
- architecture

unless intentionally running a second ablation.

## Optional ONE recipe ablation

If time remains, separately test official small-dataset guidance:

```text
mosaic=0.5
mixup=0.0
copy_paste=0.0
epochs=50
patience=20
```

If testing:

```text
lr0=0.001
```

make sure the selected optimizer actually honors it.

Do not leave `optimizer=auto` and then claim manual LR was tested.

No broad hyperparameter sweep.

---

# 13. DO NOT DO NOW

Do not:

- retrain A with more epochs
- retrain B
- jump to YOLO26l/x
- jump to 1600px
- migrate to SoccerNet/TrackLab
- replace production with FootAndBall
- build P2 before cheap rescue tests
- annotate 500 random frames
- add VAL failures directly to TRAIN
- tune on TEST
- judge by video appearance only
- count interpolated balls as detector hits
- normalize GK to player
- trust speed/distance while metric calibration is unresolved

---

# 14. TESTS REQUIRED

## Human

- [ ] raw class metadata survives association
- [ ] goalkeeper remains goalkeeper semantically
- [ ] only human roles can share human association
- [ ] ball never enters human association
- [ ] one weak role observation cannot flip stable track under smoothing
- [ ] feature flag OFF reproduces baseline

## Ball

- [ ] no anchor -> no fabricated ball
- [ ] short bounded gap can be recovered
- [ ] long gap -> unknown
- [ ] no backward fill before first observation
- [ ] camera-cut reset clears state
- [ ] deterministic multi-candidate selection
- [ ] provenance preserved
- [ ] feature flag OFF reproduces baseline

## Split safety

- [ ] temporal-val cannot enter TRAIN
- [ ] Hard-100 selector refuses VAL/TEST
- [ ] evaluation tools refuse TEST during development
- [ ] TEST remains untouched through all tasks here

---

# 15. METRICS — keep meanings separate

## Detector

- precision
- recall
- mAP50
- mAP50-95
- per class
- per match

## Human system

- any-human recall
- role confusion
- role flips/track
- unique track IDs
- track-length stats
- fragmentation proxy

Do not call proxy metrics HOTA/IDF1.

## Ball system

### Raw detector hit
Real normal-threshold detector prediction matches GT.

### Recovered low-confidence hit
A real low-confidence prediction is accepted by temporal logic.

### Interpolated short-gap coverage
No detector hit existed; a bounded temporal estimate is generated.

### Unknown
System explicitly has insufficient evidence.

Never combine these into one misleading “detector recall.”

---

# 16. STOP RULE

Stop detector/system rescue and continue the project when:

1. A@960 remains a strong real-time detector;
2. human tracking is stable enough for downstream analysis;
3. role jitter is materially reduced or honestly documented;
4. ball temporal coverage improves materially without unacceptable false recovery;
5. remaining failures are documented by match/domain;
6. additional gains would require disproportionate annotation/training time.

Then:

- freeze detector/system configuration;
- complete final TEST annotation if still pending;
- score the chosen detector on TEST exactly once;
- run formal tracking evaluation on a proper continuous benchmark;
- solve pitch calibration before trusting speed/distance.

---

# 17. FIRST EXECUTION REQUEST — DO THIS NOW

Do **not** implement the entire plan in one patch.

## Step 1 — AUDIT ONLY

Return:

1. current EyeCU detector confidence handling by class;
2. current tracker association behavior:
   - class-aware or class-agnostic?
   - can player/GK/referee associate across class changes?
3. current ball handling:
   - candidate threshold
   - multiple-ball selection
   - gap logic
   - interpolation/holding behavior
4. reusable components already in the repo;
5. exact minimal files that would need modification;
6. whether raw source videos are locally available for all four VAL matches.

Do not modify production behavior yet.

## Step 2 — MEASUREMENT SCAFFOLDING ONLY

Implement:

- any-human validation metric
- human-role confusion matrix
- `tools/build_temporal_val.py`
- temporal-val provenance
- split-safety tests

Return the proposed temporal-val windows + frame count before manual annotation.

## Step 3 — WAIT

Do not implement temporal rescue until the continuous validation benchmark exists.

---

# 18. REQUIRED RESPONSE FORMAT

Return:

```text
AUDIT
- ...

FACT-CHECK CONFLICTS WITH CURRENT CODE
- ...

REUSABLE COMPONENTS
- ...

MINIMAL PATCH PLAN
- ...

TEMPORAL-VAL PLAN
- sources:
- windows:
- frame count:
- annotation required:

TESTS
- ...

GO / NO-GO
- ...
```

No speculative model migration.

No TEST access.

No retraining.

The immediate objective is:

> Determine whether a small amount of video-aware engineering can turn A@960 into a substantially better football-analysis SYSTEM before spending more time on detector training.
