# EyeCU 4.0 — what remains to finish

_Updated 2026-08-09. This is the only active roadmap. Measured evidence lives in
[../results/RESULTS.md](../results/RESULTS.md); superseded plans are in
[../archive/](../archive/)._

This document answers one question: **what is left to do?** Anything already
measured has been removed from it and recorded in RESULTS.

---

## Where the project stands

| | |
|---|---|
| Production detector | **`best_A_960.pt`** — YOLO26s @960, 823 train frames |
| Sparse val (208 imgs) | mAP50 **0.739**, player R 0.916, referee R 0.809, GK R 0.515, ball R 0.486 |
| Continuous val (104 frames) | ball recall **0.571** raw, **0.623** coverage with the temporal selector |
| Speed | ~58 FPS on T4; 154.7 ms/frame on the dev CPU |
| Alternatives | `best_B_1280.pt` — better ball, 1.70× cost, kept as accuracy reference. `best_C_960.pt` — **rejected** |
| Tests | 111 fast, 9 slow |
| **Test set** | **still unlabelled — 177 frames. No held-out number exists yet.** |

---

## Done — do not redo

Recorded in [../results/RESULTS.md](../results/RESULTS.md) with numbers.

- Real 208-image validation set built, corrected and frozen (4 match-disjoint matches)
- **Experiment A** — YOLO26s @960, the production candidate
- **Experiment B** — @1280; better ball, worse GK/referee at a matched threshold, 1.70× cost
- **Experiment C** — contextual crop/zoom + hard negatives; **rejected**, net −10 ball on continuous val
- 104-frame continuous temporal benchmark built, hand-annotated and evaluated
- `BallTemporalSelector` — bounded, provenance-tagged, frozen at v1 settings,
  and **wired into the production pipeline** (2026-08-15); the legacy
  `interpolate_ball_positions()` was deleted
- **Patch 0** — removed the origin-ball fabrication from the tracker. The
  pipeline kept calling the legacy interpolator until the selector replaced it
- **Patch 0b** — ball-only duplicate suppression at IoU 0.70; FP 14→10 with TP unchanged
- Ball candidate threshold study — frozen at 0.10, run once
- Detector diagnostics — any-human recall, role confusion, per-match ball metrics
- 2×2 cross-resolution ablation — A/B weights × 960/1280 inference

---

## Critical path

### 1. Label the test set — **the blocking task**

177 frames across `como_2-0_sassuolo`, `manchester_city_v_liverpool`, `youth_2`.

- [ ] Draft with `--backend roboflow` (**not** the local model — it hallucinates
      referees on unseen kits; see RESULTS)
- [ ] Correct by hand — budget 8–15 h
- [ ] Validate and dedupe

Procedure: [../guides/LABELING.md](../guides/LABELING.md).

### 2. Score the test set — **once**

- [ ] Rebuild the split so `test` exists
- [ ] Evaluate the already-chosen detector on `split='test'`
- [ ] Record as the headline result

Choose the model on validation. Touch test exactly once. If you evaluate and
then change something, the number is gone.

### 3. Tracking evaluation

EyeCU is a tracking system and only the detector has been measured.

- [ ] Freeze detector output to disk
- [ ] Run ByteTrack over the frozen detections
- [ ] Report unique track ids, track-length distribution, fragmentation proxy,
      role flips per track
- [ ] Report the ~145-ids-over-300-frames churn figure and what it means

Do not call these IDF1 or HOTA — there is no identity ground truth.

### 4. Speed and distance — calibrate or cut

`pixels_per_meter = 12.0` is a guess applied uniformly despite perspective and
zoom, so every metre and km/h figure is meaningless.

- [ ] Either calibrate via pitch homography from known line markings,
- [ ] or **remove speed/distance from the output** and say why in the write-up

Both are defensible. Shipping numbers labelled `_UNCALIBRATED` is the weakest
option.

---

## Optional, in priority order

### 5. One architecture ablation — P2 / stride-4 head

The audit found the finest detection stride is 8, at which the median training
ball (6.7 px at 960 geometry) is **sub-cell**; 63% of training balls are under
8 px. A width-matched P2 head costs 88.2 GFLOPs against A's 51.9 (1.70×) and
lands within 6% of B@1280, enabling a matched-cost comparison of finer feature
stride versus input upsampling.

Designed but **not approved or run**. Requires Colab. Main confound: only 58.2%
of pretrained weights transfer to the P2 topology, against 100% for A.

### 6. Quantify the generalisation failure

The kit-based referee confusion is currently one anecdote plus per-match
numbers. Turn it into a per-match referee-precision table across held-out
matches.

### 7. External data

[../research/EXTERNAL_DATASETS.md](../research/EXTERNAL_DATASETS.md).
SoccerTrack v2 first (CC BY 4.0, documented schema); SoccerNet GSR second,
gated on the NDA. Only after the critical path.

---

## Explicitly out of scope

Say so in the write-up rather than leaving them looking forgotten: jersey OCR,
face recognition, 3D mesh, persistent player database, cross-match
re-identification, UI, highlight generation, event detection (preserved
unintegrated in `experimental/`), BoT-SORT and Deep OC-SORT, SAHI, ROI rescue,
and a further Hard-100 annotation batch.

---

## The write-up

- [ ] **Problem** — duplicate boxes, ID churn, referee/goalkeeper confusion, cloud dependency
- [ ] **Method** — dataset construction, match-disjoint splitting, the pseudo-label → correct → train loop
- [ ] **Results** — per-class test metrics, FPS, duplicate rate before/after, tracking baseline
- [ ] **Failure analysis** — Austin referee confusion, close-up blindness, the two ball failure regimes
- [ ] **Limitations** — small val, uncalibrated speed, one annotator, single-domain external data
- [ ] **Reproducibility** — tooling, tests, frozen splits, per-experiment hashes

Four things worth foregrounding, because they are unusual in student work:

1. **Declining to compare models on 33 goalkeeper instances**, and saying why.
2. **The aspect-ratio catch** — external data was squashed 1.83×; merging it
   blind would have taught the model that people are ~1.8× too narrow.
3. **The class-order catch** — Roboflow alphabetises classes on export;
   importing on id would have relabelled every player as a referee.
4. **A negative result reported as a negative result** — Experiment C hit its
   target regime (3 of 3 rescues in-regime) and was still rejected, because it
   lost more than it gained.

---

## Sequencing

| stage | effort | blocks |
|---|---|---|
| 1. Label test (177) | 8–15 h | the final number |
| 2. Test evaluation | ~1 h | — |
| 3. Tracking baseline | 4–6 h | — |
| 4. Calibrate or cut speed | 2–4 h | — |
| 5–7. Optional | 10–20 h | — |
| Write-up | 8–12 h | — |

**Annotation is the bulk of what is left.** If time is short, cut in this
order: external data → architecture ablation → generalisation study. Never cut
the test set — a project without a held-out result has no result.

## The one thing not to do

Do not retrain chasing a better validation number. The val set carries roughly
±0.09 on goalkeeper recall; much of the movement between runs is noise, and
tuning against noise is how a project ends up with an impressive figure that
means nothing. Where two models must be compared, pair them on the same
instances — `tools/compare_models.py`.
