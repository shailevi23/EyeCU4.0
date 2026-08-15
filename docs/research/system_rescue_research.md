# System-rescue research notes

_Distilled 2026-08-09 from the original fact-checked rescue plan. The full
original, including the parts that have since been executed or superseded, is
preserved at [../archive/system_rescue_plan_original.md](../archive/system_rescue_plan_original.md)._

This file keeps only the material that is still **reference-worthy**: verified
facts about the reference implementation, football-CV literature that informs
design choices, and architectural rationale. Everything that was a task list has
moved to [../coursework/COURSEWORK_PLAN.md](../coursework/COURSEWORK_PLAN.md);
everything that was measured has moved to
[../results/RESULTS.md](../results/RESULTS.md).

---

## 1. The reference repository — verified findings

Reference: <https://github.com/abdullahtarek/football_analysis>

These were checked against the actual source, not assumed.

**Training recipe (confirmed).** `training/football_training_yolo_v5.ipynb`
trains `yolov5x.pt` for 100 epochs at `imgsz=640` on Roboflow project
`football-players-detection-3zvbc` v1.

**Dataset size — precise wording matters.** The project exposes **372 source
images**; dataset v1 contains **663 generated images** (612 train / 38 val /
13 test) after horizontal-flip, saturation and brightness augmentation.
Provenance does *not* prove all 372 came from one video, so the defensible
description is "a small, visually narrow football dataset", not "trained on one
video".

**Destructive goalkeeper merge (confirmed).** `trackers/tracker.py` rewrites
goalkeeper detections to `player` before tracking. **EyeCU deliberately does
not do this** — see the four-class rule below.

**The reference does not track the ball (confirmed).** Ball boxes bypass
ByteTrack, are stored under a fixed id, then passed through
`interpolate_ball_positions()` with `bfill()`. There is no maximum-gap limit,
so a position can be fabricated before the first observation and held
indefinitely after the last. A smooth-looking demo video is therefore not
evidence of detector quality.

**Source-specific hard-coding (confirmed).** Fixed perspective vertices and a
literal `if player_id == 91` in team assignment. The demo is not a clean
generalisation benchmark.

> EyeCU inherited a variant of the interpolation bug and it was worse: gaps were
> filled with `[0,0,0,0]`, so the DataFrame contained no NaN, `interpolate()`
> was a no-op, and a ball was emitted at the frame origin on every missed frame.
> Patch 0 removed the fabrication from the **tracker**, but the pipeline kept
> calling `interpolate_ball_positions()` afterwards, so the bug survived one
> step further down than the note originally implied. The method has since been
> deleted outright and `BallTemporalSelector` is the production ball path; see
> "Ball temporal recovery (current)" in
> [../results/RESULTS.md](../results/RESULTS.md).

---

## 2. Literature relevant to EyeCU's measured failures

**ByteTrack** (<https://arxiv.org/abs/2110.06864>) — retaining low-score
detections and associating them is the core idea; Ultralytics defaults are
`track_high_thresh 0.25`, `track_low_thresh 0.10`, `new_track_thresh 0.25`.
This supports a low-confidence *candidate pool*, which EyeCU adopted at 0.10.
It does **not** establish that ByteTrack is the right tracker for a tiny, fast,
erratically-moving ball.

**Temporal reasoning for the ball** (<https://arxiv.org/abs/1909.02406>) —
frame history helps through brief disappearance and partial occlusion. It does
not justify unlimited interpolation. EyeCU's `BallTemporalSelector` is bounded
and provenance-tagged for this reason.

**SAHI** (<https://arxiv.org/abs/2202.06934>) — sliced inference improves
small-object AP on VisDrone/xView without fine-tuning, at the cost of multiple
passes. Evidence is not football-specific. EyeCU's own measurements weaken the
case further: all four validation videos are 640×360, so nothing is being
downsampled at `imgsz=960` for slicing to recover.

**FootAndBall** (<https://arxiv.org/abs/1912.05445>) and **DeepBall**
(<https://arxiv.org/abs/1902.07304>) — football-specific ball detection using
high-spatial-resolution features and multi-level semantic context, plus
hard-negative mining because easy background overwhelms difficult ball-like
negatives. Directly relevant: EyeCU's detector produces false balls on a
goalkeeper's shirt crest, and its finest detection stride is 8, at which the
median training ball (6.7 px at 960 geometry) is sub-cell.

**SoccerNet Game State Reconstruction**
(<https://github.com/SoccerNet/sn-gamestate>, <https://arxiv.org/abs/2404.11335>)
— decomposes the problem into detection, re-identification/tracking, role
classification, team affiliation, jersey recognition and pitch localisation.
This supports EyeCU keeping *association* and *semantic role* as related but
separate operations.

**YOLO26 small-dataset guidance** — Ultralytics suggests `mosaic=0.5`,
`mixup=0.0`, `copy_paste=0.0`, `lr0=0.001`, `epochs=50`, `patience=20` below
~1,000 images. **Caveat that has bitten this project:** with
`optimizer="auto"`, manually supplied LR and momentum are ignored — every EyeCU
run so far logged `optimizer='auto' found, ignoring 'lr0' and 'momentum'` and
resolved to AdamW at lr 0.00125. Do not claim a manual-LR experiment without
setting the optimizer explicitly.

---

## 3. Architectural rationale that still stands

**Four semantic classes, end to end.** `player`, `goalkeeper`, `referee`,
`ball` are preserved through detection, association and reporting. A
class-agnostic *human* view is permitted for association only; `raw_class` is
never overwritten and goalkeeper never collapses into player. Measurement
supports this: goalkeeper errors are ~59% role confusion rather than missed
localisation, so the information is present and worth keeping.

**Ball state must be explicit.** Every ball output carries one of
`observed`, `recovered_low_conf`, `interpolated_short_gap`, `unknown`.
"I don't know" is a first-class answer, and a recovered or interpolated ball is
never counted as a raw detector hit.

**Sparse validation cannot evaluate temporal logic.** The 208-image validation
set was interval-sampled, so consecutive frames are seconds apart. Any
gap-recovery or velocity rule scored on it would be measuring nothing. This is
why the 104-frame continuous benchmark exists.

**Duplicate suppression is ball-only.** YOLO26 is end-to-end and runs no NMS —
the `iou` argument to `predict()` is inert. Humans legitimately overlap when
contesting a header and the head is meant to emit both, so suppression is
restricted to the ball class.

---

## 4. Recommendations from the original plan — current status

| original recommendation | status |
|---|---|
| Remove unbounded ball interpolation | **done** (Patch 0) |
| Bounded, provenance-aware temporal recovery | **done** (`trackers/ball_temporal.py`) |
| Continuous temporal validation benchmark | **done** (104 frames, 8 windows) |
| Ball low-confidence candidate study | **done** — threshold frozen at 0.10 |
| Ball-only duplicate suppression | **done** (Patch 0b, IoU 0.70) |
| Any-human recall + role-confusion diagnostics | **done** (`tools/diagnose_detector.py`) |
| Class-agnostic human association | **not needed** — supervision's ByteTrack is already class-agnostic; the gap is role smoothing, worth at most +4.3 points |
| ROI / SAHI ball rescue | **not pursued** — 640×360 sources leave nothing for slicing to recover |
| Hard-100 active-learning batch | **not pursued as scoped** — "ball near player" is already 77.7% of TRAIN |
| Dedicated high-resolution ball pathway | **open** — the current architecture audit's recommended direction |
