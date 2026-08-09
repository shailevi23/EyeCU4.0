# EyeCU 4.0 — Results

_Last updated: 2026-08-09._

Measured results only. Plans live in [../coursework/COURSEWORK_PLAN.md](../coursework/COURSEWORK_PLAN.md).

---

## Headline

A football-specific detector trained on our own footage, replacing a generic
COCO model behind a cloud API.

| | before | after |
|---|---|---|
| Pipeline throughput | **0.6 FPS** (Roboflow cloud) | **57.7 FPS** local |
| Ball detection | 1 frame in 60 | recall **0.49**, precision 0.81 |
| Goalkeeper | **not detectable** (no such COCO class) | recall 0.52, precision 0.85 |
| Referee | **not detectable** | recall 0.80, precision 0.65 |

Current model: `A_yolo26s_960_realval`, mAP50 **0.739**, mAP50-95 **0.474** on
208 held-out images from 4 matches never trained on.

The first two rows are the project's original problem statement. A COCO model
has no `goalkeeper` or `referee` class at all, so the last two rows are a
capability that did not previously exist.

---

## Detector — retired run (`eyecu_football_v1.pt`)

YOLO26s @ 960 px, 80 epochs requested, **early-stopped at 65** with the best
checkpoint at epoch 45. 30.4 minutes on a Colab T4. Trained on 366 EyeCU
images, validated on 85 held out by match.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| player | 0.927 | 0.910 | 0.949 | 0.735 |
| referee | 0.617 | 0.703 | 0.709 | 0.518 |
| ball | 0.908 | 0.591 | 0.694 | 0.448 |
| goalkeeper | 0.632 | 0.576 | 0.590 | 0.486 |
| **all** | 0.771 | 0.695 | **0.735** | **0.547** |

Inference 13.6 ms/image → **41 FPS**.

### Progression from the 10-epoch pilot

| Class | pilot recall | Experiment A |
|---|---|---|
| goalkeeper | 0.121 | **0.576** |
| referee | 0.505 | **0.703** |
| ball | 0.570 | 0.591 |
| player | 0.909 | 0.910 |

mAP50-95 0.450 → 0.547. Goalkeeper was the class predicted to fail on 110
training instances; most of the gap closed with more epochs.

### ⚠️ These numbers are superseded

They were measured against a **temporary** 85-image validation set carved from
training sources, because the real validation matches had no labels yet. That
set had 33 goalkeeper instances — a 95% confidence interval 0.32 wide (±0.16)
on goalkeeper recall, far too wide to compare models on.

The real validation set (208 images, 4 frozen matches, 115 goalkeepers) now
exists. **Experiment A must be re-run against it, and the result will not be
comparable to the table above.**

### Pilot on the real validation set — 2026-08-09

10 epochs, 823 train / 208 val, to confirm the rebuilt dataset trains and to
time a full run. **This is the first measurement against the real held-out
validation set** and supersedes everything above.

| Class | n | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| player | 2,490 | 0.876 | 0.926 | 0.953 | 0.667 |
| referee | 257 | **0.611** | 0.747 | 0.733 | 0.466 |
| goalkeeper | 115 | 0.773 | 0.574 | 0.653 | 0.435 |
| ball | 111 | 0.779 | **0.486** | 0.548 | 0.267 |
| **all** | 2,973 | 0.757 | 0.683 | **0.722** | 0.459 |

49.7 FPS. 9.7 minutes for 10 epochs.

Three things this measurement establishes:

**The merged dataset is doing real work.** 10 epochs here reaches mAP50 0.722;
the previous model needed 65 epochs to reach 0.735 on an *easier* val set.
Training images went 366 → 823.

**Goalkeeper precision rose 0.632 → 0.773** while recall held steady. Training
instances went 110 → 431, mostly from the external merge.

**Referee is now the weakest class by precision (0.611), not recall (0.747).**
The model over-predicts referees — it finds them, and it also finds things that
are not them. This is the kit-confusion failure below, appearing in a headline
metric for the first time, because the new val set contains
`austin_fc_vs__club_tijuana`. It was always happening; the old val set could
not see it.

Ball is the weakest class overall (mAP50-95 0.267) and the main open question
for the full run.

### Experiment A on the real validation set — 2026-08-09

`A_yolo26s_960_realval`. YOLO26s @ 960, 80 epochs requested. **Early-stopped at
epoch 32, best checkpoint at epoch 12.** 34.7 minutes, 823 train / 208 val.

| Class | n | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| player | 2,490 | 0.864 | 0.918 | 0.939 | 0.663 |
| referee | 257 | **0.646** | 0.798 | 0.770 | 0.515 |
| goalkeeper | 115 | 0.845 | **0.515** | 0.701 | 0.453 |
| ball | 111 | 0.812 | **0.486** | 0.547 | 0.262 |
| **all** | 2,973 | 0.789 | 0.679 | **0.739** | **0.474** |

**57.7 FPS** (13.4 ms inference).

#### The finding is the plateau, not the score

Best checkpoint at **epoch 12 of 80**. Against the 10-epoch pilot on the same
val set:

| metric | pilot (10 ep) | A (best ep 12) | Δ | |
|---|---|---|---|---|
| mAP50 | 0.722 | 0.739 | +0.017 | |
| mAP50-95 | 0.459 | 0.474 | +0.015 | |
| player recall | 0.926 | 0.918 | −0.008 | within noise |
| goalkeeper recall | 0.574 | 0.515 | −0.059 | within noise |
| referee recall | 0.747 | 0.798 | +0.051 | within noise |
| **ball recall** | **0.486** | **0.486** | **0.000** | identical |
| ball mAP50-95 | 0.267 | 0.262 | −0.005 | |

Ball recall is identical to three significant figures — 54 of 111 in both runs.
Note this does **not** establish that the same 54 instances were found; two
models can share a recall while succeeding and failing on different frames.
Instance-level agreement has not been measured (see `tools/compare_models.py`).

**Training past epoch 12 produced no further improvement**, and the run was
stopped at 32. Compared with the 10-epoch pilot, the selected checkpoint
represents about two extra epochs of training. So the accurate claim is
narrower than "80 epochs bought nothing": *the 20 epochs after the best
checkpoint bought nothing*, and the pilot had already reached essentially the
same place.

What this does **not** establish is the cause. A plateau at 960 px says this
*configuration* stopped improving. It does not isolate which of dataset size,
dataset diversity, object scale, input resolution or augmentation is binding —
those are confounded in a single run. Experiment B (1280 px, everything else
held fixed) is the controlled ablation that separates resolution from the
rest.

Ball and goalkeeper remain the weak classes. Referee precision (0.646) is still
the weakest precision, unchanged in character from the pilot — the kit
confusion is not a training-length problem either.

### Confidence intervals — why the val rebuild mattered

95% Wilson intervals on recall:

| Class | n | recall | 95% CI | full width | ± | old width |
|---|---|---|---|---|---|---|
| player | 2,490 | 0.926 | [0.915, 0.936] | 0.021 | ±0.011 | 0.034 |
| referee | 257 | 0.747 | [0.690, 0.796] | 0.106 | ±0.053 | 0.174 |
| goalkeeper | 115 | 0.574 | [0.483, 0.661] | 0.178 | **±0.089** | 0.320 |
| ball | 111 | 0.486 | [0.395, 0.578] | 0.183 | **±0.092** | 0.231 |

Goalkeeper and ball intervals nearly halved.

Two caveats on how to use these. The interval **width** is ~0.18 for ball, so
the margin either side is ~±0.09, not ±0.18. And these are *independent-sample*
intervals: when two models are scored on the **same** validation instances, a
paired comparison is far more sensitive, because the shared instances cancel.
Comparing models by checking whether one point estimate clears the other's
independent CI is too conservative and will hide real differences. Use
`tools/compare_models.py`, which pairs per instance and applies McNemar.

---

### Experiment B @1280, and why A was kept — 2026-08-09

`B_yolo26s_1280_realval`. YOLO26s @ 1280, full 80 epochs, 139 minutes.
mAP50 **0.758**, mAP50-95 **0.488**, **32.6 FPS**.

Two methodological points had to be settled before the comparison meant
anything, and both changed the answer.

**Ultralytics does not report P/R at a fixed threshold.**
`ultralytics/utils/metrics.py:883` selects a *single shared* operating point
per model — `i = smooth(f1_curve.mean(0), 0.1).argmax()` — the confidence that
maximises the class-*mean* F1, then applies that one index to every class. It
is not per-class, and it is not constant across models:

| model | index | confidence |
|---|---|---|
| A @960 | 248 | **0.2482** |
| B @1280 | 193 | **0.1932** |

So the headline A-vs-B tables compare A at conf 0.248 with B at conf 0.193.
Every per-class number quoted for B is measured at a lower threshold than A's.

**Re-measured at an identical threshold**, B is better on ball at every
operating point, and *worse* on goalkeeper and referee:

| conf | A ball R | B ball R | A gk R | B gk R | A ref R | B ref R |
|---|---|---|---|---|---|---|
| 0.10 | 0.550 | **0.595** | — | — | — | — |
| 0.25 | 0.459 | **0.514** | **0.513** | 0.409 | **0.809** | 0.747 |

**A was kept on cost, not on accuracy.** B costs 1.73× per image
(measured twice: 0.565 FPS ratio on T4, 0.579 on CPU) for +0.055 ball recall
at matched threshold, while losing goalkeeper and referee. The defensible
conclusion is narrow: *raising input resolution from 960 to 1280 did not
produce an improvement large or robust enough to justify its cost under this
training setup* — not that resolution is irrelevant.

Context that explains the small gain: **all four validation videos are
640×360**, and the median validation ball is **6.0 × 6.0 px**. A@960 already
upsamples 1.5×; B@1280 upsamples 2×. Neither adds information.

### Patch 0b — ball duplicate suppression — 2026-08-09

YOLO26 is end-to-end and runs no NMS. The `iou` argument to `predict()` is
inert — verified on this checkpoint, which returns 18 boxes at iou 0.1, 0.5 and
0.9 alike. Nothing suppressed a second box on the same ball.

At the candidate threshold 0.10, 14 of A's 53 ball false positives overlapped a
real ball. Thirteen of those overlapped a ball *already claimed* by a better
prediction; twelve of those thirteen had **prediction-to-prediction** IoU
≥ 0.70 (median 0.84). Prediction-vs-GT overlap is not what NMS thresholds on,
so the pairwise distribution was measured separately before choosing.

Suppression threshold chosen from a measured plateau, not tuned: 0.50/0.60/0.70
all remove the same 12 pairs with zero true detections lost; 0.80 removes 10.
**0.70 is the top of the plateau** — maximum removal, widest margin against
suppressing genuinely distinct balls.

| | TP | FP | recall | precision | mean IoU | median IoU |
|---|---|---|---|---|---|---|
| accepted ≥0.25, before | 51 | 14 | 0.4595 | 0.7846 | 0.7553 | 0.7784 |
| accepted ≥0.25, **after** | **51** | **10** | 0.4595 | **0.8361** | 0.7518 | 0.7652 |
| candidate ≥0.10, after | 61 | 41 | 0.5495 | 0.5980 | 0.7444 | 0.7503 |

TP is unchanged globally **and in every one of the four validation matches**.
Because suppression keeps the most confident box and in 7 of 13 pairs that was
not the best-fitting one, localisation drifts slightly: mean matched IoU
−0.0035, median −0.0132. On a 6 px ball that is sub-pixel, and it is reported
rather than rounded away.

Two limits stated deliberately:

- **This validation set cannot detect harm to a spare ball.** No frame in it
  carries more than one ball box (97 frames with none, 111 with exactly one),
  and `LABELING.md` specifies only *visibility*, never *which* ball. A correctly
  detected second football is already scored as a false positive here, so
  suppressing one would register as an improvement. The 0.70 threshold bounds
  the risk — a spare ball elsewhere on the pitch has near-zero IoU with the
  match ball — but the measurement is silent on it.
- Suppression is **ball-only**. Humans are excluded by design: players
  legitimately overlap when contesting a header, and the end-to-end head is
  meant to emit both. Verified bit-identical human boxes with the flag on.

The flag is **off by default**; with it off the detector emits exactly what it
emitted before this patch, including no `state` key.

### Continuous temporal benchmark — 2026-08-09

The 208-image validation set is interval-sampled, so consecutive frames are
seconds apart and no temporal rule can be evaluated on it. A separate benchmark
was built: **104 frames**, two 2.5 s windows per validation match at fixed
fractions of duration (40% and 70%), sampled at 5 FPS, hand-annotated for the
ball only. 77 frames contain a ball, 27 do not; 13 are tagged
`closeup_or_non_gameplay`.

Window selection is content-neutral — chosen by a fixed rule, never by
inspecting detector behaviour.

| | A@960 raw | with frozen selector |
|---|---|---|
| ball recall / coverage | **0.5714** (44/77) | **0.6234** (48/77) |
| precision | 0.9167 | — |
| FP per frame | 0.0385 | — |
| hallucinated frames | — | 1 of 27 |

The selector recovered 3 balls from the low-confidence pool and interpolated 5,
with **zero false recoveries attributable to the temporal layer** — the single
hallucinated frame is a raw detector false positive present in the baseline
too. On the 13 close-up frames the detector produced 2 ball candidates (both
the goalkeeper's circular shirt crest) and the selector correctly refused both,
having no history anchor.

Two failure regimes dominate, and they are different problems:

| window | GT | A recall | character |
|---|---|---|---|
| youth w1 | 12 | **0.000** | large balls (18.9–24.2 px @960) at players' feet in close-contact duels |
| women_1 w1 | 6 | 0.167 | small balls (7–11 px) against stone-wall and stand backgrounds |

Of 17 misses across those two windows, **12 had no usable proposal even at
confidence 0.01** — detector blindness, not thresholding.

### Cross-resolution 2×2 — 2026-08-09

A and B weights each evaluated at 960 and 1280 on the continuous benchmark,
identical matching:

| config | TP | FP | recall |
|---|---|---|---|
| A@960 | 44 | 4 | 0.5714 |
| A@1280 | 41 | 3 | 0.5325 |
| B@960 | 44 | 4 | 0.5714 |
| B@1280 | 50 | 7 | **0.6494** |

At 960, A and B have equal aggregate recall but materially different per-window
behaviour — B gains in some windows and loses in others. B's aggregate
advantage appears only at 1280, indicating a strong interaction between
training configuration, learned weights and inference resolution. **For these
specific checkpoints on this benchmark, A performs worse at 1280 than at 960
while B performs better at 1280 than at 960.** That is a checkpoint-specific
interaction, not a universal rule.

24 of A@960's 33 misses were recovered by **none** of the four configurations.

### Experiment C — rejected — 2026-08-09

`C_yolo26s_960_scale_context_hardneg`. YOLO26s @960, best epoch 13, early
stopped at 33, 37 min. Training data: the 823 original frames **plus 100
derived views** — 70 contextual crop/zoom positives placing balls in the
12–25 px band (median 18.4) and 30 mined hard negatives. Recipe otherwise
identical to A.

This was a **targeted contextual crop/zoom + hard-negative data ablation**, not
pure ball-scale augmentation: the crops also raised median human box height
from 32.8 px to 69.7 px (2.12×).

| | A | C |
|---|---|---|
| 208 VAL ball recall | **0.4595** | 0.4144 |
| 208 VAL referee recall | **0.809** | 0.650 |
| 208 VAL goalkeeper recall | 0.513 | **0.539** |
| 208 VAL player recall | 0.916 | 0.915 |
| 104 temporal raw recall | **0.5714** (44 TP) | 0.4416 (34 TP) |
| 104 temporal precision | 0.9167 | **0.9714** |
| temporal coverage w/ selector | **0.6234** | 0.5974 |
| speed | 154.7 ms/frame | 164.4 ms/frame |

Ball paired on 208 VAL: both 42, A-only 9, C-only 4, neither 56 — McNemar
p = 0.2668.

**The intervention hit its target and was still net-negative.** All 3 rescued
frames were in youth w1, all large balls in player contact — 3 of 3 in the
predeclared regime, from a baseline of 0/12. But C lost 13 detections
elsewhere, 9 of them in youth w0. Rescue mean ball width was ~21.8 px against
~11.8 px for the regressions; the pattern suggests the intervention biased
performance toward the larger apparent-ball regime while sacrificing detections
at more common scales. No precise scale boundary is claimed.

False positives fell on every measure — temporal FP 4→1, ball FP 10→8, native
duplicates 4→2. This is consistent with the intended effect of hard-negative
mining, **but the contribution of hard negatives cannot be isolated from the
crop/zoom intervention**, since the two were introduced together.

**Verdict: rejected as production candidate.** A@960 remains production;
B@1280 remains the accuracy reference at 0.7143 coverage for 1.70× compute.

What this does **not** establish: that large-ball coverage does not help. The
valid conclusion is that *this specific* crop/zoom + hard-negative intervention
did not improve the required validation performance. The ball-scale effect and
the human-scale shift cannot be separated without a further run, which was not
made.

## Measured failure modes

### Generalisation: kit-based referee confusion

`eyecu_football_v1` run on `austin_fc_vs__club_tijuana`, a match in neither
train nor val. Austin play in green. On one frame:

| | our model | Roboflow (generic) |
|---|---|---|
| player | 8 | **14** ✓ |
| referee | **6** ✗ | 0 ✓ |
| goalkeeper | 0 | **1** ✓ |

Across 20 frames: **81 referees** (4/frame, where a pitch has at most 3
officials) and only 8.1 players/frame against Roboflow's 13.2. The model
labels green-shirted **players** as referees.

Its validation referee recall of 0.703 was measured on the retired val set,
whose four matches had kits resembling the training set. It did not hold on a
genuinely new one.

**The rebuilt validation set now catches this.** `austin_fc_vs__club_tijuana`
is one of the four frozen val matches, and referee precision on the 2026-08-09
pilot is 0.611 — the weakest figure of any class on any metric. The failure was
always present; it is now measured rather than anecdotal.

### Close-up blindness

**42 of 451 frames (9.3%)** returned zero detections from the hosted drafter,
concentrated in `betis_3_vs_5_fc_barcelona` (14/41) and `youth_3` (14/51).
Two were inspected visually: one a player-and-referee close-up, one two clearly
visible players. Wide-shot detectors collapse outside broadcast wide framing.

### Tracking: ID churn

**145 unique track IDs across 300 frames**, where a match has ~22 players on
pitch. `player_statistics.json` therefore counts *tracks*, not people, and is
reported as `unique_track_ids` for that reason. Not yet measured properly —
see COURSEWORK_PLAN §6.

### Speed and distance are UNCALIBRATED

`pixels_per_meter = 12.0` is a guess applied uniformly despite perspective and
zoom. Output fields are named `*_UNCALIBRATED` and `final_report.json` carries
`speed_distance_calibrated: false`. Figures are relative only.

An earlier run reported player speeds of 124–205 km/h. Half of that was a
separate bug: the estimator was fed the source FPS instead of
`source_fps / skip_frames`, inflating every speed by exactly the skip factor
(2× at the default). Fixed and tested. The calibration error remains.

---

## Dataset

| Split | images | matches | player | goalkeeper | referee | ball |
|---|---|---|---|---|---|---|
| train | 823 | 29 | 12,975 | 431 | 1,453 | 673 |
| val | 208 | 4 | 2,490 | 115 | 257 | 111 |
| test | 177 | 3 | **unlabelled — deliberately untouched** | | | |

Match-disjoint and domain-stratified (pro / women / youth / amateur). Leakage
is checked automatically and the build fails if any match appears twice.

**Sources.** 1,483 frames extracted from 23 of our own videos, plus 372
external frames from
[roboflow-jvuqo/football-players-detection-3zvbc v20](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
(CC BY 4.0), pinned to train only. Provenance in
`data/external_provenance.json`.

**Annotation.** 659 frames corrected by hand across two batches (451 train,
208 val), each drafted by a hosted detector first. Roughly 40 hours.

The correction step is where the value is. Batch 02, validation:

| | drafted | corrected | change |
|---|---|---|---|
| goalkeeper | 37 | **115** | +78 (3.1×) |
| referee | 135 | **257** | +122 |
| ball | 84 | 111 | +27 |
| player | 2,544 | 2,490 | −54 |

---

## Bugs found and fixed

Recorded because several would have silently corrupted results rather than
failing loudly.

| Bug | Consequence if missed |
|---|---|
| Roboflow alphabetises classes on export | Every player relabelled as referee |
| External images squashed 16:9 → 1:1 | Model taught people are 1.8× too narrow |
| `cv2.imwrite` silently fails on non-ASCII paths | 178 frames reported saved, none written |
| Speed used source FPS, not effective FPS | Every speed inflated by `skip_frames` (2×) |
| Cache key ignored video/model/settings | A run silently reuses another run's detections |
| Unlabelled frames included as background | Model taught real players are not there |
| Split written only for non-zero ratios | A pinned-only val split never reached disk |
| 80 annotations returned as polygons | Ultralytics rejects those lines outright |
| 18 duplicate/near-duplicate boxes | Teaches the detector that double-boxing is correct |
| API key printed in error text | Key leaked to console on every failed request |

The aspect-ratio and class-order catches both came from measuring the external
data against ours rather than trusting the export.

---

## Reproducibility

- **55 automated tests** (46 fast, 9 slow). `pytest -m "not slow"` runs in ~2 s.
- Splits are seeded and pinned; frozen val/test cannot drift as the pool grows.
- Every tool is CLI-driven, with `--dry-run` wherever it writes.
- Detection caches are keyed on video, model, detector and tracker settings,
  `skip_frames` and `max_frames`.

The pipeline runs fully offline — no Roboflow, no network, no API key:

```bash
python run_pipeline.py --input input-videos/short.mp4 \
    --yolo-model eyecu_football_v1.pt --imgsz 960 --max-frames 300
```
