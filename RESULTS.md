# EyeCU 4.0 — Results

_Last updated: 2026-08-09._

Measured results only. Plans live in [COURSEWORK_PLAN.md](COURSEWORK_PLAN.md).

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
set had 33 goalkeeper instances — a 95% confidence interval of ±0.16 on
goalkeeper recall, far too wide to compare models on.

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

Ball recall is identical to seven decimal places — 54 of 111 instances, the
same 54 — across two independently trained models. Every per-class change sits
inside its confidence interval.

**Seventy extra epochs bought nothing measurable.** The model converges in ~12
epochs and then stops improving. That is a statement about the dataset, not the
schedule: at 823 training images with 673 ball instances, capacity and training
time are not the binding constraint. More epochs, and probably more resolution,
cannot manufacture examples that are not there.

This is the single most useful negative result in the project so far. It says
where the remaining effort should go: **more and more varied labelled data**,
not longer training.

Ball and goalkeeper remain the weak classes. Referee precision (0.646) is still
the weakest precision, unchanged in character from the pilot — the kit
confusion is not a training-length problem either.

### Confidence intervals — why the val rebuild mattered

95% Wilson intervals on recall:

| Class | n | recall | 95% CI | width | old width (85-img val) |
|---|---|---|---|---|---|
| player | 2,490 | 0.926 | [0.915, 0.936] | 0.021 | 0.034 |
| referee | 257 | 0.747 | [0.690, 0.796] | 0.106 | 0.174 |
| goalkeeper | 115 | 0.574 | [0.483, 0.661] | **0.178** | 0.320 |
| ball | 111 | 0.486 | [0.395, 0.578] | **0.183** | 0.231 |

Goalkeeper and ball intervals nearly halved. Still wide enough that a
difference under ~9 points on those two classes is not a difference.

---

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

- **41 automated tests** (32 fast, 9 slow). `pytest -m "not slow"` runs in ~2 s.
- Splits are seeded and pinned; frozen val/test cannot drift as the pool grows.
- Every tool is CLI-driven, with `--dry-run` wherever it writes.
- Detection caches are keyed on video, model, detector and tracker settings,
  `skip_frames` and `max_frames`.

The pipeline runs fully offline — no Roboflow, no network, no API key:

```bash
python run_pipeline.py --input input-videos/short.mp4 \
    --yolo-model eyecu_football_v1.pt --imgsz 960 --max-frames 300
```
