# EyeCU 4.0 — Plan to Finish

_Written 2026-08-08. Supersedes the forward-looking half of TODO.md; the
cleanup sections there are done and stay as a record._

## Where the project actually stands

| | |
|---|---|
| Detector | `eyecu_football_v1.pt` — YOLO26s @ 960, trained on 366 frames |
| Val results | mAP50 **0.735**, player R 0.910, referee R 0.703, goalkeeper R 0.576, ball R 0.591 |
| Speed | **41 FPS** local, against ~0.6 FPS for the old Roboflow-cloud pipeline |
| Dataset | **823 train / 208 val** images, match-disjoint, leakage-checked, splits pinned |
| Labels | 1,031 of 1,855 frames — 659 hand-corrected, 372 external |
| Tests | 41 automated (32 fast, 9 slow), all passing |
| **Test set** | **never evaluated — 177 frames still unlabelled** |

The engineering is in good shape. The **measurement** is not finished, and that
is what a project like this is graded on.

---

## The honest assessment

What is already strong, and worth writing up as such:

- **Match-disjoint splitting with automated leakage checks.** Most student
  projects split frames randomly and report inflated numbers. Ours cannot.
- **A documented, reproducible pipeline** from raw video to trained model.
- **Real measured improvement** over the starting point: 0.6 → 41 FPS, and ball
  recall from 1-in-60 frames to 0.59.
- **Failure modes found and evidenced**, not hidden.

What is missing, in order of how much it costs the project:

1. **No test-set number.** Every figure quoted is validation. A project that
   never touches its held-out set has no final result.
2. ~~The validation set is too small to conclude anything.~~ **Fixed** —
   goalkeepers went 33 → 115. Still not large: expect roughly ±0.09 on
   goalkeeper recall, enough to rank models but not to call a 2-point
   difference meaningful.
3. **No tracking evaluation.** EyeCU is a tracking system; only the detector
   has been measured.
4. **Known generalisation failure, unquantified.** The model calls
   green-shirted players "referee" on an unseen match. Measured on one match,
   never on a set.
5. **Speed/distance output is uncalibrated** and currently meaningless.

---

## Critical path — do these in order

Everything here is required for a defensible final result.

### 1. ✅ Finish the validation set — **DONE 2026-08-09**

208 frames across the 4 frozen val matches, corrected and imported. Validator
passes with 0 errors after 5 duplicate boxes were removed.

| | drafted | corrected | change |
|---|---|---|---|
| goalkeeper | 37 | **115** | +78 (3.1×) |
| referee | 135 | **257** | +122 |
| ball | 84 | 111 | +27 |
| player | 2,544 | 2,490 | −54 |

Validation is now genuinely held out: 4 EyeCU matches, no external frames,
never trained on. **Goalkeeper val instances went 33 → 115**, which is what
makes model comparison meaningful. The estimate above was 80–150; the outcome
landed mid-range.

Note the previous 85-image val was a stand-in carved from train sources and is
retired. Any number measured against it is not comparable to what follows.

### 2. Retrain on the real split — **NEXT**

The split rebuild is done. `build_dataset.py` gained `--force-val` /
`--force-test`, and the frozen matches are now pinned rather than re-derived,
so the split cannot drift as the training pool grows.

| Split | images | matches | player | goalkeeper | referee | ball |
|---|---|---|---|---|---|---|
| train | 823 | 29 | 12,975 | 431 | 1,453 | 673 |
| val | 208 | 4 | 2,490 | 115 | 257 | 111 |

- [ ] Upload the rebuilt `data/football_dataset.zip` (**102 MB** — the previous
      90 MB file was built against the retired val and must be replaced)
- [ ] In Colab, re-run cells 2–4 and confirm the check prints
      `val 208 images, 4 matches`
- [ ] Run **Experiment A only**:
      `train('A_yolo26s_960', 'yolo26s.pt', imgsz=960, epochs=80)`
- [ ] Record per-class precision/recall and FPS

Expect goalkeeper to improve — training instances went 110 → 431.

**The new numbers will not be comparable to mAP50 0.735.** That was measured on
the retired stand-in val. The real val is harder and more honest: it includes
`austin_fc_vs__club_tijuana`, the match where the current model calls
green-shirted players "referee". Referee precision may well look *worse*, and
that would be the measurement finally catching a failure that was always there.

**Do not** run Experiments B or C. Even at 115 goalkeepers the confidence
interval is wide; comparing three architectures on it would be ranking noise.

### 3. Label the test set

- [ ] Draft the 177 test frames with `--backend roboflow`
      (`como_2-0_sassuolo`, `manchester_city_v_liverpool`, `youth_2`)
- [ ] Correct them — **budget 8–15 h**
- [ ] Validate and dedupe

**Use Roboflow, not the local model.** On unseen matches `eyecu_football_v1`
hallucinates referees; feeding those drafts into the test set would corrupt the
one measurement that has to be trustworthy.

### 4. Score the test set — **once**

- [ ] Rebuild with `--ratios 0.70,0.15,0.15` so test exists
- [ ] Evaluate the already-chosen model on `split='test'`
- [ ] Record it as the headline result

Choose the model on val. Touch test exactly once. If you evaluate on test and
then change something, the number is gone.

### 5. Duplicate-box suppression (TODO §Phase 4)

- [ ] Measure `duplicate_prediction_rate` on the current detector — the
      baseline number
- [ ] Add class-aware NMS plus conservative same-class suppression at
      IoU 0.85–0.90 in the detector layer
- [ ] Re-measure and report before/after

This is the original problem statement — "two boxes around the same player".
Going from problem statement to a measured before/after is exactly the arc a
project should show. The tooling already exists: `tools/dedupe_labels.py` uses
the same IoU logic on labels.

### 6. Tracking evaluation (TODO §Phase 5)

- [ ] Freeze detector output to disk
- [ ] Run ByteTrack over the frozen detections; record ID switches, track
      fragmentation, and unique track count against ~22 real players
- [ ] Report the **145 track IDs over 300 frames** figure and what it means

Even without BoT-SORT, a measured ByteTrack baseline plus an honest statement
of the ID-churn problem is a complete section. Comparing trackers is optional;
measuring one is not.

---

## High value, if time allows

### 7. Quantify the generalisation failure

Currently one anecdote (Austin FC green kit → "referee"). Turn it into a
measurement: run the detector across all held-out matches and report
per-match referee precision. A project that finds, evidences and explains its
own failure is stronger than one that reports only the good number.

### 8. Calibrate speed, or remove it

`pixels_per_meter = 12.0` is a guess applied uniformly despite perspective and
zoom. Either:

- calibrate via pitch homography from known line markings (the real fix), or
- **remove speed/distance from the output entirely** and say why.

Both are defensible. Shipping numbers marked `_UNCALIBRATED` is the weakest
option — either fix it or cut it.

### 9. External data (see `EXTERNAL_DATASETS.md`)

Only after the critical path. SoccerTrack v2 first (CC BY 4.0, no NDA,
documented schema); SoccerNet GSR second, gated on reading the NDA. Estimated
~4× more goalkeepers with no annotation labour.

---

## Explicitly out of scope

Say so in the write-up rather than leaving them looking forgotten:

- Jersey OCR, face recognition, 3D mesh, persistent player database
- Cross-match re-identification
- Streamlit / UI
- Highlight generation
- Event detection (preserved unintegrated in `experimental/`)
- BoT-SORT and Deep OC-SORT (measure ByteTrack first)

---

## The write-up

Worth planning as a deliverable, not an afterthought. The material already
exists in the repo's history.

- [ ] **Problem** — duplicate boxes, ID churn, referee/goalkeeper confusion,
      cloud dependency. Evidenced with the original failures.
- [ ] **Method** — dataset construction, match-disjoint splitting, the
      pseudo-label → correct → train → re-draft loop.
- [ ] **Results** — per-class test metrics, FPS, duplicate rate before/after,
      tracking baseline.
- [ ] **Failure analysis** — the Austin referee confusion with the side-by-side
      image; the 9% of frames returning nothing; goalkeeper scarcity.
- [ ] **Limitations** — 33-goalkeeper val CI, uncalibrated speed, single-domain
      external data, one annotator.
- [ ] **Reproducibility** — the tool chain, the 41 tests, the frozen splits.

Three things worth foregrounding, because they are unusual in student work and
are genuinely defensible:

1. **The confidence-interval argument.** Declining to compare models on 33
   goalkeeper instances, and saying why, is better science than reporting a
   winner.
2. **The aspect-ratio catch.** External data was squashed 1.83×; merging it
   blind would have taught the model that people are ~1.8× too narrow. Caught
   by measuring box aspect ratios against our own.
3. **The class-order catch.** Roboflow alphabetises classes on export; importing
   on id would have relabelled every player as a referee.

---

## Realistic sequencing

| Stage | Effort | Blocks |
|---|---|---|
| 1. Correct val (208) | 10–20 h | everything |
| 2. Retrain | ~1 h | — |
| 3. Draft + correct test (177) | 8–15 h | the final number |
| 4. Test evaluation | ~1 h | — |
| 5. Duplicate suppression | 3–5 h | — |
| 6. Tracking baseline | 4–6 h | — |
| 7–9. High value | 10–20 h | optional |
| Write-up | 8–12 h | — |

**Annotation is ~60% of the remaining effort.** If time is short, cut scope in
this order: external data → tracker comparison → generalisation study. Never cut
the test set — a project without a held-out result has no result.

## The one thing not to do

Do not retrain repeatedly chasing a better validation number. The val set has a
±0.09 margin on goalkeeper recall; much of the movement between runs is noise,
and tuning against noise is how a project ends up with an impressive figure
that means nothing. Where two models must be compared, pair them on the same
validation instances rather than reading independent intervals — see
`tools/compare_models.py`.
