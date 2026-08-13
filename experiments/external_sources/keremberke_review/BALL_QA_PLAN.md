# Ball annotation QA — plan

Plan only. Nothing was implemented, no annotation was modified, no model was run,
no training, no TEST access. Every number below is derived from the promoted
export and the decision log on 2026-08-13.

---

## 1. Promoted export verification

Read from the files on disk, not from the earlier in-memory build.

| Check | Result |
| --- | --- |
| `repaired_export/` exists, 4 files | yes |
| `EXPORT_MANIFEST.json` parses, contract_version | **2** |
| created / git commit | 2026-08-13T09:57:12Z / `1b1dd68340ff` |
| decisions fingerprint in manifest vs live | **MATCH** `8bc9eaa2…` · 26,436 lines |
| total annotations (derived) | **22,891** |
| train / valid / test | 15,966 / 4,518 / 2,407 |
| images | 859 / 243 / 130 = **1,232** |
| by class (derived) | football **1,267** · player 18,475 · referee 1,982 · goalkeeper 1,167 |
| manifest agrees with files | yes |
| additions / removals / geometry repairs / class changes / ball changes | 46 / 33 / 7 / 3,123 / 4 |
| live source hashes | train, valid, test **all MATCH** |
| original 1,263 ball GT set-identical | **yes** |
| human-approved additional balls | **4**, all `RECLASSIFY_TO_BALL` |

**PROMOTED == CHECKED STATE: true.** File hashes recorded:
`train 9caf8edd…` · `valid b00432fa…` · `test ccfd491a…` · `manifest 2a604f3d…`

The four human-approved balls, for the record:
`train:1858 [643.0,382.0,4.89,6.07]` · `train:9614 [561.0,52.0,8.6,7.21]` ·
`train:15737 [1129.0,248.0,8.25,7.5]` · `valid:2721 [529.0,456.0,17.2,13.64]`

---

## 2. Ball dataset profile

All 1,232 images are 1280×720. No augmentation families exist (0 stems shared
across rf-hashes), so there is no cross-augmentation consistency signal.

### Balls per image

| | images | 0 ball | 1 ball | >1 ball |
| --- | --- | --- | --- | --- |
| train | 859 | **180 (21.0%)** | 562 | 117 |
| valid | 243 | **43 (17.7%)** | 160 | 40 |
| test | 130 | **33 (25.4%)** | 76 | 21 |
| **all** | **1,232** | **256 (20.8%)** | 798 | **178** |

Distribution of counts: `{0:256, 1:798, 2:104, 3:42, 4:26, 5:5, 6:1}`

### By run and family

| Group | images | 0-ball |
| --- | --- | --- |
| plain_A | 187 | 38 (20.3%) |
| plain_B | 555 | 112 (20.2%) |
| pp_A | 49 | 15 (30.6%) |
| pp_B | 436 | 90 (20.6%) |
| pp (family) | 486 | 106 (21.8%) |
| plain (family) | 746 | 150 (20.1%) |

The 0-ball rate is strikingly flat across runs. That is *weak* evidence against a
run-specific annotation failure, and no evidence at all about the overall rate.

### Ball box size (stored px, 1280×720)

median width **9.14** · median height 8.97 · median area 82.1 · min width 3.17 ·
max width **664.66**

| threshold | boxes | share |
| --- | --- | --- |
| ≤5 px | 91 | 7.2% |
| ≤8 px | 475 | 37.5% |
| ≤12 px | 972 | **76.7%** |
| ≤20 px | 1,169 | 92.3% |

Percentiles: p50 9.1 · p90 16.0 · p95 48.4 · p99 86.0. The jump between p90 and
p95 is the signature of a second population, not a tail.

### Two structural findings the profile surfaced

**72 boxes (5.7%) are wider than 40 px**, up to 664×252. A football at broadcast
scale on a 1280×720 frame is 3–20 px. A 664 px "ball" is not a ball; the ones
around 85–140 px are likely close-up or replay frames, or annotations on
something else. **This is a category C (false ball) population that nobody has
looked at**, and it is large enough to matter: 5.7% of the ball GT.

**178 images carry more than one ball**, but only 3 pairs overlap at IoU > 0.1 and
only 4 pairs sit within 2× box-width. So multi-ball images are overwhelmingly
*genuine* separate boxes, not duplicates. Any candidate generator that assumes
"one ball per image" would be wrong on 14% of images, and the duplicate problem
(category D) appears to be ~3 pairs, i.e. negligible.

Aspect ratio is reassuring: median 0.99, only 11 boxes (0.9%) outside 0.5–2.0.

---

## 3. What is currently known about ball label quality

- **Every original ball annotation survived the repair set-identical.** Verified
  by hash, not by count.
- **Four balls were added**, each from an explicit human `RECLASSIFY_TO_BALL`
  decision on a box that was mis-annotated as a person. All four were found
  incidentally, by the `BALL_WRONG_HUMAN_BOX` queue — not by any ball search.
- Ball geometry was **never modified** by the repair. No geometry-repair event
  targets a ball.
- The size distribution matches the audit's original finding: this dataset's
  value is small balls, 76.7% under 12 px.
- Aspect ratios are round, so the boxes that *do* exist are shaped like balls.

---

## 4. What is NOT known — the honest list

1. **How many visible footballs have no annotation at all.** Never measured. Not
   once, by anyone, at any point. This is the entire reason for this plan.
2. **Whether the 256 zero-ball images are correct.** A frame with the ball out of
   play, off-frame, or hidden is legitimately 0-ball. A frame where the ball is
   visible and unlabelled is a training-set defect. The ratio is unknown.
3. **Whether the 72 oversized boxes are balls.** Untouched by every pass so far.
4. **Whether existing ball boxes are well-placed.** No ball box has ever been
   inspected by a human for geometry, only for existence.
5. **Whether small balls are systematically missed.** The most damaging possible
   bias, and the one the size distribution cannot reveal: a missing 4 px ball
   leaves no trace in a histogram of the balls that *were* labelled.
6. **The residual rate after any repair.** No post-repair QA exists for balls,
   exactly as it did not exist for roles until the 6.40% measurement forced it.

---

## 5. Candidate-generation design

### Available signals, each with its blind spot and circularity risk

| Signal | Usefulness | Likely blind spot | Circularity risk |
| --- | --- | --- | --- |
| **Frozen EyeCU detector** (`best_A_960.pt`, the one `kb_role_triage` used) at a very low ball confidence floor | **High.** The only signal that can point at a specific pixel location. | Small, blurred, occluded and edge balls — precisely EyeCU's known weakness and the reason this data was wanted. | **Severe.** This detector's ball weakness is *why* keremberke was acquired. Using it to audit ball labels asks the model to find its own failures. |
| **Low-confidence predictions** (0.03–0.15) | High for recall; this is where a missed tiny ball lives if it is detected at all. | Enormous false-positive volume; socks, line markings, distant heads. | Same detector, same blind spot — a lower threshold widens the net without changing its shape. |
| **Temporal / adjacent frames** | Would be excellent. | **Not available.** Measured earlier at ~0.52 median NCC between consecutive filename ids; the images are sparsely sampled, not consecutive frames. | n/a — cannot be used. |
| **Box-size priors** (3–20 px, aspect ≈1) | Good as a *filter* on candidates, and it flags the 72 oversized boxes for category C. | Cannot generate a candidate; only reject one. | Low. Derived from the labelled balls, so it inherits their bias toward labelled sizes. |
| **Image regions** (ball rarely in the top rows / crowd) | Modest ranking prior. | Lobbed and headed balls are exactly the interesting cases and sit high. | Low, but it would suppress the hardest positives. |
| **GT ↔ detector disagreement** | The core mechanic: unmatched plausible predictions are candidates. | Only finds what the detector proposed. | **Severe** — this is the circularity, stated plainly. |
| **Augmentation / duplicate relationships** | Would let one image's label vote on its twin. | **Not available**: 0 augmentation families found. | n/a |
| **Run / source grouping** | Useful for *stratification*, not generation. | Flat 0-ball rate across runs means it discriminates little. | None. |
| **Existing 1,267 balls as positive priors** | Good for calibrating size, aspect and appearance. | Only describes balls that were labelled. | Moderate — a systematic labelling bias would be learned as "normal". |

### Proposed pipeline

```
frozen detector @ conf ≥ 0.03, ball class, 1280 imgsz
        +  existing 1,267 ball GT
                 ↓
   match predictions to GT (IoU ≥ 0.3, or centre within 1.5× GT width —
   tiny boxes make IoU brittle, so centre distance is the primary matcher)
                 ↓
   unmatched predictions → filter: 2 ≤ w ≤ 40 px, 0.4 ≤ aspect ≤ 2.5
                 ↓
   rank by (confidence × size-prior × region-prior)
                 ↓
        human review queue, image-centric
```

Every prediction is a **proposal**. Nothing becomes GT without a human drawing or
confirming, exactly as in the role repair.

**Also queue unconditionally, independent of the detector:**
- the **72 boxes wider than 40 px** (category C/A) — a fixed, complete population
- the **3 overlapping pairs** in multi-ball images (category D)

Those two need no detector at all and are cheap and certain.

---

## 6. Circularity and blind-spot risks — and why candidate review is not enough

The role cleanup already ran this experiment. The triage flagged 4,153 candidates;
a human answered every one; and a stratified QA sample of what the triage
*rejected* then found a **6.40% missed-role rate**, implying ~1,118 missed
officials. Completing the queue proved nothing about recall.

Balls are worse, for a specific reason: the assisting detector is weakest on
exactly the object this dataset was acquired to supply. If it misses a 4 px ball
— the class of ball that is 7.2% of the labelled data and probably a larger share
of the *unlabelled* data — that ball never enters any queue, and no amount of
candidate review will surface it.

**Therefore the candidate pass cannot be the evidence.** It is the cheap
correction mechanism. The evidence has to come from a population the detector
was not consulted about.

---

## 7. Independent QA design

Sample from what the candidate generator **rejected**, mirroring `qa_player`.

**Frame:** images where the pipeline produced **no candidate at all** — no
unmatched plausible prediction. This is the population where the system is
implicitly asserting "nothing is missing here", and it is the only place its
recall can be measured.

That frame splits into two strata that fail differently and must be sampled
separately:

- **S1 — zero ball GT and zero candidate.** 256 images are 0-ball; the subset with
  no candidate is the highest-risk stratum. A visible ball here is both
  unlabelled *and* invisible to the assistant.
- **S2 — ball GT present, every prediction matched, no unmatched candidate.** Tests
  the *second* ball in a multi-ball frame, which the 178 multi-ball images show is
  common.

**Stratification within each**, to prevent a bias toward easy frames:
run (plain_A/B, pp_A/B) × view (wide vs tight, proxied by median player box
height in the image) × field region. Reviewers see the whole image, so
ball-size regime cannot be a sampling axis — it is only knowable after the
answer. Instead, **report the size of every ball found**, and treat a found ball
under 8 px as a distinct and more serious signal than one over 20 px.

**The reviewer's task is a full-image sweep**, not a box judgement: *"is there a
football visible in this image that is not annotated?"* That is what makes it
independent of the detector.

---

## 8. Tiny-ball coverage — avoiding a bias toward easy balls

The failure mode to design against: a QA that finds only the balls that were easy
to see, concludes the dataset is fine, and hides the small-ball problem that is
the dataset's entire value.

Countermeasures:

1. **Sample images, not boxes.** Sampling boxes can only find problems where a box
   already exists. Whole-image review is the only way a 4 px unlabelled ball can
   be found.
2. **Mandatory zoom in the UI**, with the reviewer able to pan at 4–8×. A 4 px ball
   is invisible at fit-to-window on a 1280×720 image.
3. **Do not rank the QA sample by anything.** Ranking is for the candidate queue;
   the QA sample must be a probability sample or it estimates nothing.
4. **Report found balls by size bucket** (≤5, ≤8, ≤12, >12) and treat the buckets
   separately in the escalation rule. Three missed 4 px balls mean something very
   different from three missed 30 px balls.
5. **Force coverage of hard conditions** by including, as *named strata* in the
   candidate queue rather than the QA sample: edge-of-frame regions, balls within
   a player box (occlusion/adjacency), and high/wide broadcast shots (identified
   by small median player height).
6. **Motion blur and partial occlusion** cannot be detected reliably from metadata
   here. They are handled by the reviewer's `UNSURE` answer, and an image marked
   `UNSURE` is counted separately — never folded into "no ball visible".

---

## 9. Workload options

Estimates assume ~15 s per candidate image and ~30 s per QA image (zoom and sweep
take longer than judging a proposal).

### MINIMAL
- Candidate queue: **72 oversized boxes + 3 overlap pairs only.** No detector run.
- Independent QA: **120 images** (S1 80 / S2 40).
- Effort: ~20 min + ~1 h ≈ **1.5 h**
- Defensible: "the obviously wrong ball boxes are fixed"; "no *gross* missing-ball
  problem, detectable only above roughly 3–4%".
- **Not** defensible: any claim about small balls, any residual rate below ~3%, any
  statement that the dataset is clean.

### BALANCED — recommended
- Candidate queue: detector @ conf ≥ 0.03, filtered and ranked. Expect roughly
  **300–500 candidate images** (extrapolating from role triage, which produced
  4,153 candidates over 1,170 images at a comparable recall setting; balls are one
  object per frame rather than 22, so the yield should be far lower). **Plus the 72
  + 3 fixed populations.**
- Independent QA: **300 images**, stratified (S1 200 / S2 100).
- Effort: ~2 h candidates + ~2.5 h QA ≈ **4.5 h**
- Defensible: a corrected dataset, **plus** a measured residual missing-ball rate
  with a 95% CI. If 0 found in 300 → CI [0, 1.2%]. Size-bucketed reporting.
- Not defensible: a residual claim below ~1%, or per-run claims (strata too thin).

### HIGH-CONFIDENCE
- Candidate queue: as BALANCED at conf ≥ 0.01, plus a **second, independent
  proposal source** — `yolov8n.pt` COCO `sports ball`, which has a different
  training distribution and therefore different blind spots. Union the two.
- Independent QA: **600 images**, and a **100-image double-review** by the same
  person on a different day to estimate reviewer miss rate — because a human
  sweeping for a 4 px ball also misses some, and an unmeasured reviewer miss rate
  silently floors the whole estimate.
- Effort: ~4 h + ~5 h + ~1 h ≈ **10 h**
- Defensible: residual rate below ~0.6%, per-size-bucket rates, and a stated
  reviewer miss rate that bounds the estimate honestly.

**Recommendation: BALANCED.** MINIMAL cannot say anything about the small balls
that are the dataset's whole point. HIGH-CONFIDENCE is the right shape but its
extra 5.5 hours buys precision that only matters if BALANCED finds a problem — and
its escalation path leads there anyway.

---

## 10. Stopping and escalation rule — fixed before any review

Set now so the result cannot be rationalised afterwards. `n = 300`.

| QA finding | Action |
| --- | --- |
| **0 missing balls** | Stop. Report 95% CI [0, 1.2%]. Ball labels defensible. |
| **1–3 missing (≤1%)**, all >12 px | Correct them. Stop. Report the rate with CI. Large balls missed occasionally is a known annotation-fatigue pattern and does not threaten small-ball value. |
| **1–3 missing, any ≤8 px** | Correct, and **extend QA to 600**. A small ball missed even once at n=300 implies a rate that matters for the exact use case. |
| **4–9 missing (1.3–3%)** | Correct, extend to 600, and characterise: which run, which size, which region. |
| **≥10 missing (≥3.3%)** | **Stop reviewing and investigate.** This is a systematic pocket, not noise. Characterise it and build a *targeted* second-pass queue from the pattern — exactly what the 6.40% role finding produced in the 6,684-box sweep. |

Justification for the thresholds: 3% is roughly half the 6.40% role rate that was
judged serious enough to force a full retrospective sweep, so it is a defensible
line for "systematic". 1% is the level below which a residual defect is smaller
than the reviewer miss rate we cannot measure at BALANCED, so claiming better
would be claiming precision we do not have.

**Errors found during QA stay in the measured rate even after correction.**
Correcting them does not un-find them; removing them from the denominator would
be measuring the dataset after fixing exactly the parts we looked at.

---

## 11. Tools that would need to be built

| Tool | Purpose | Reuses |
| --- | --- | --- |
| `kb_ball_candidates.py` | Run the frozen detector at a low floor, match to GT, emit the ranked queue + the 72 oversized + 3 overlaps | `kb_role_triage` matching logic |
| `kb_ball_qa_sample.py` | Draw the stratified independent QA sample, seeded and reproducible | `kb_build_review_package` sampling |
| `kb_ball_review_server.py` | The review UI | `kb_geometry_repair_server` almost entirely — drawing, zoom, image-coordinate conversion, `kb_images` resolver, preflight |
| `kb_ball_gate.py` | Ball-specific gate conditions | `kb_second_pass_gate` structure |
| extend `kb_export_v2.py` | Ball additions/removals/repairs as new change kinds under contract v2 | the existing clause structure |

**UI answers** (section 7 of the brief): `EXISTING BALL OK` · `MISSING BALL → draw`
· `FALSE BALL → remove` · `BAD BALL BOX → redraw` · `NO BALL VISIBLE` · `UNSURE`.
Whole image visible, zoom to 8×, every ball in a multi-ball image independently
actionable, all existing annotations shown as context. Same rules as the last
three tools: append-only events, human geometry only, no model proposal ever
becomes GT, provenance on every change.

## 12. Dataset safety

The promoted `repaired_export/` is the **immutable input**. Ball QA produces a
*new* derived export, not a mutation of the only clean copy, and passes its own
gate before promotion. Decisions append to the same log in new modes
(`ball_qa_*`), so `kb_decisions.resolve()` stays the single precedence rule. **No
already-completed human role decision is reopened.**

---

## 13. Should ball QA happen before Experiment D?

**One reason it should:** Experiment D exists to test whether keremberke's tiny
balls improve EyeCU's ball detection. If ~20% of frames have an unlabelled
visible ball, those frames teach the detector that a ball is background — the
precise opposite of the intent. A null or negative Experiment D result would then
be uninterpretable: bad data, or a bad idea? Ball QA is what makes the experiment
falsifiable.

**Two honest reasons it might wait:**

1. The 0-ball rate is **20.8% and flat across all four runs**. A systematic
   annotation failure would more likely cluster. Flatness is consistent with
   "the ball genuinely is out of frame or occluded in about a fifth of sampled
   broadcast frames", which is plausible for sparsely sampled footage.
2. Experiment D's design could **mask the question entirely** by training only on
   frames that have a ball. That does not fix the dataset, but it makes the
   experiment valid without the QA, at the cost of discarding 256 images.

**Recommendation: run BALANCED ball QA first, but only the 300-image independent
sample** — about 2.5 hours — before Experiment D. If it returns 0–1 missing balls,
proceed to Experiment D immediately and treat the candidate correction pass as
optional cleanup. If it returns more, the candidate pass has become necessary and
you will know that before spending compute on an experiment whose result you
could not have interpreted.

That inverts the usual order deliberately: **measure first, correct second.** The
role cleanup did it the other way round and the QA is what revealed that the
correction had been incomplete.
