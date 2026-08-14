# Ball annotation QA — plan

Plan only. Nothing was implemented, no annotation was modified, no model was run,
no training, no TEST access. Every number below is derived from the promoted
export and the decision log on 2026-08-13.

**Revision 2 — 2026-08-14: the first measurement is model-independent.** The
first version made the independent QA a sample of images the *candidate
generator* rejected. That estimates the generator's recall, not the dataset's
missing-ball prevalence: the population is defined by a model, so every inclusion
probability is conditional on detector behaviour and the denominator is not the
dataset. **BALL QA ROUND 0** (§5) now draws directly from all 1,232 images with
no detector involved, and candidate generation moves to a conditional **Round 1**
(§6). The generator-recall question survives, correctly scoped, in §7.

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
   once, by anyone, at any point. This is the entire reason for this plan, and
   **Round 0 (§5) is the instrument that answers it** — items 2 and 5 below are
   answered by the same 300 images, which is why Round 0 comes first.
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

## 5. BALL QA ROUND 0 — the model-independent measurement

**Revision, 2026-08-13.** The earlier version of this plan framed the independent
QA as a sample of images the candidate generator *rejected*. That frame estimates
the **candidate generator's recall**. It does not estimate the dataset's
missing-ball prevalence, because the population it draws from is defined by a
model — every inclusion probability is conditional on detector behaviour, and the
denominator is not the dataset. It was the right instrument aimed at the wrong
question.

**Round 0 does not consult the detector at all.**

> **Purpose.** Estimate the image-level prevalence of visible but unannotated
> footballs in the promoted `repaired_export`, independently of any assisting
> model.

### 5.1 Primary endpoint

> **Image-level missing-ball defect rate.** An image is **positive** if it
> contains ≥ 1 visible football that lacks an annotation.

The endpoint is a property of the **image**, not of the ball. An image holding
three missing balls contributes:

| | |
| --- | --- |
| **primary endpoint** | **one positive image** |
| secondary count | three missing ball objects |

Both are reported; they are never added, averaged, or substituted for each other.
The primary endpoint is what the confidence interval is computed on, because the
sampling unit is the image and only the image has a defined inclusion
probability. The object count has no denominator — there is no enumerable
population of unannotated balls to sample from — so it is reported as a **count
and a size distribution, never as a rate**.

`UNSURE` is a third outcome and is never folded into `NO MISSING BALL`. It is
reported separately and, in the escalation rule, treated as its own quantity.

### 5.2 Sampling design — simple random sampling without replacement

**Revision 3, 2026-08-14.** An earlier draft of this section used proportional
stratified allocation by run × GT state. It was *approximately* self-weighting —
π_h ranged 0.2000 to 0.2667 against f = 0.2435 — and that approximation is the
problem. "Self-weighting to within rounding" is a claim a reader has to take on
trust, and it invites exactly the ambiguity the primary estimator must not have:
if π_h differs by stratum at all, then p̂ = positives/300 is only *nearly* the
right estimator, and the honest version needs weights.

> **SRSWOR. n = 300 from N = 1,232. Fixed reproducible seed.
> Every image has inclusion probability exactly 300/1232 = 0.24350649…**

Not approximately. Exactly, by construction, for every image in the population.
The primary estimator is therefore exactly

    p̂ = positive_images / 300

with **no weighting ambiguity and no stratum bookkeeping to get wrong.**

**No fixed counts are forced by run or GT state.** The realised sample will land
near the population proportions because that is what random sampling does, and
where it does not, that is ordinary sampling variation — not a design flaw to
correct. Correcting it *would* be the flaw.

Instead, for every sampled image the manifest **records** descriptive metadata:
source split · run · current ball GT count · 0-ball vs ≥1-ball · view proxy
where available · image dimensions. These variables exist for **post-hoc
descriptive analysis, not for inclusion.** They are read at reporting time and
never at sampling time.

The sample manifest records, and the analysis reads back: **the population
fingerprint (sha256 of the three promoted annotation files), N, n, seed, the
inclusion probability, and the exact ordered list of sampled image IDs** — so
the measurement is bound to the snapshot it measured, and a changed source
fingerprint invalidates the sample rather than silently re-pointing it.

### 5.3 Human task

The reviewer sees the **entire image with the existing ball GT drawn on it**, and
mandatory zoom/pan (to 8×; a 4 px ball is invisible at fit-to-window).

Answers, one per image:

| | |
| --- | --- |
| `NO MISSING BALL` | sweep complete, nothing unannotated found |
| `MISSING BALL` | ≥1 visible unannotated football |
| `UNSURE` | cannot decide — reported separately, never counted as clean |

On `MISSING BALL`, the reviewer draws **every** missing ball visible in that
image. Each object is recorded separately with its own geometry; **the image is
counted positive exactly once regardless of object count.** Findings append to
the decision log under new `ball_qa_r0_*` modes, so `kb_decisions.resolve()`
remains the single precedence rule.

**Freeze the measurement before any correction is promoted.** The Round-0 result
is computed from the log at a recorded fingerprint and written to an immutable
report. A discovered error stays in the measured rate after it is corrected —
removing it from the numerator would be measuring the dataset after fixing
exactly the parts we looked at.

### 5.4 Statistical analysis

Under SRSWOR every image carries the same inclusion probability, so the
estimator is the unweighted proportion p̂ = (positive images)/300 — exactly, not
approximately.

**Revision 4, 2026-08-14: the interval now matches the design.** Sampling is
without replacement from a finite N = 1,232, so the positive count is
**hypergeometric, not binomial**. The primary 95% interval is obtained by
**inverting the hypergeometric sampling distribution** — the set of population
positive-counts M under which the observed x falls in neither 2.5% tail:

    M_lo = min{ M : P(X ≥ x | N, M, n) ≥ 0.025 }
    M_hi = max{ M : P(X ≤ x | N, M, n) ≥ 0.025 }

Both probabilities are monotone in M and there are only 1,233 candidate values,
so the endpoints are found by exact search — no root-finding, no continuity
correction, no normal approximation.

| positives | p̂ | **exact finite-population 95% CI** | as image counts /1,232 | binomial reference |
| --- | --- | --- | --- | --- |
| **0** | 0.00% | **[0.00%, 1.06%]** | **[0, 13]** | [0.00%, 1.22%] |
| 1 | 0.33% | [0.08%, 1.62%] | [1, 20] | [0.01%, 1.84%] |
| 2 | 0.67% | [0.16%, 2.11%] | [2, 26] | [0.08%, 2.39%] |
| 3 | 1.00% | [0.32%, 2.60%] | [4, 32] | [0.21%, 2.89%] |
| 4 | 1.33% | [0.49%, 3.08%] | [6, 38] | [0.36%, 3.38%] |
| 5 | 1.67% | [0.73%, 3.49%] | [9, 43] | [0.54%, 3.85%] |
| 9 | 3.00% | [1.62%, 5.28%] | [20, 65] | [1.38%, 5.62%] |
| 10 | 3.33% | [1.87%, 5.68%] | [23, 70] | [1.61%, 6.04%] |

**The endpoints are integer counts of images**, which is what the estimand
actually is; the percentages are those counts divided by 1,232.

Clopper–Pearson remains in the report but is labelled a **conservative binomial
reference**. It assumes sampling with replacement from an infinite population,
discarding the information that 300 of the 1,232 images were genuinely
inspected, so it is uniformly wider. It is **not** the exact interval for this
design and must not be described as one.

One consequence worth understanding rather than patching: **M_lo can exceed x.**
At x = 3 the lower bound is 4, because if the population held exactly 3
positives, drawing all 3 of them in a 300-image sample has probability 0.0143 —
itself a 2.5%-tail event. Observing 3 is mild evidence that more than 3 exist.

**No interval before the round is complete.** `--interim` reports counts only —
answered, outstanding, positive images, missing objects drawn, UNSURE — plus the
*logical* bounds on the final positive count (minimum = current positives;
maximum = current + outstanding + UNSURE). Those bounds carry no sampling
probability and are not a confidence interval. A partial sample has no
denominator, since the unreviewed images are not negatives, and a number printed
beside the word "CI" gets quoted regardless of its caption. **The frozen report
is the first place a prevalence interval may appear.**

**If equal inclusion probability is ever broken** — an image found unreadable
and dropped, the population changed, any image reviewed outside the drawn
sample — the unweighted interval stops being valid and the report must say so
rather than quote it. A "0/300 → [0, 1.06%]" that does not come from an equal-
probability sample of the stated population is a false precision claim — and the
hypergeometric inversion makes that worse, not better, since it assumes the 300
were drawn from exactly these 1,232. The report checks this and refuses rather
than adjusting silently.

Alongside the interval, report:

- **positive images / 300**, with CI (primary endpoint)
- **total missing ball objects** found, and objects per positive image
  (secondary counts, no rate)
- **size distribution** of found objects in the ≤5 / ≤8 / ≤12 / >12 px buckets
- **split, run, GT state (0-ball vs ≥1-ball) and view**, descriptively. These
  are post-hoc cuts of a sample that was not allocated by them, so subgroup
  counts will be small and unequal; they may **not** be quoted as per-run rates
  or used for between-run comparison
- **`UNSURE` count**, separately, with its own images listed

**Round 0 says nothing about candidate-generator recall.** No candidate
generator was used. Any recall claim requires the rejected-population QA
described in §7, which is a *different* measurement of a *different* quantity and
must not be conflated with this one.

---

## 6. Candidate generation — ROUND 1, conditional on Round 0

Round 1 runs **only after Round 0 is measured, locked and reported.**

| Round 0 result | Round 1 |
| --- | --- |
| missing-ball evidence low | **Experiment D may proceed without a full correction campaign.** Round 1 is optional cleanup. |
| meaningful problem | Run the frozen detector and any other proposal sources to build a **high-recall correction queue**. |

The model-assisted queue is a **CORRECTION mechanism. It is not evidence that the
dataset was clean** — that evidence is Round 0's, and only Round 0's. The
generator's own recall is a separate question, measurable afterwards by the
rejected-population QA in §7.

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

**Also queue, independent of the detector — but AFTER Round 0 is locked:**

**`OVERSIZED_BALL_REVIEW` — the 72 boxes wider than 40 px.** Renamed from the
earlier "false ball population". They are **suspicious and require human review,
but size alone is not authority.** Close-up, replay and cropped views can contain
a legitimately large ball, and a 664 px box on a goalmouth replay is a different
object from a 664 px box on a wide shot. The population is fixed and complete
(5.7% of ball GT); the *verdict* on each box is a human's, per box, with the
image in front of them. No box is removed because of its width.

**`BALL_OVERLAP_REVIEW` — the 3 overlapping pairs** (IoU > 0.1) in multi-ball
images. A deterministic duplicate-review population, also complete.

**Both are reviewed after the Round-0 measurement is locked**, so that correcting
them cannot contaminate the baseline. If an oversized box is removed before
Round 0 finishes, the Round-0 sample would be measuring an export that no longer
exists, and any image containing that box would have been reviewed under a
different GT state than the one the report names.

---

## 6b. Circularity and blind-spot risks — why Round 1 cannot be the evidence

The role cleanup already ran this experiment. The triage flagged 4,153 candidates;
a human answered every one; and a stratified QA sample of what the triage
*rejected* then found a **6.40% missed-role rate**, implying ~1,118 missed
officials. Completing the queue proved nothing about recall.

Balls are worse, for a specific reason: the assisting detector is weakest on
exactly the object this dataset was acquired to supply. If it misses a 4 px ball
— the class of ball that is 7.2% of the labelled data and probably a larger share
of the *unlabelled* data — that ball never enters any queue, and no amount of
candidate review will surface it.

**Therefore Round 1 cannot be the evidence.** It is the cheap correction
mechanism. The evidence has to come from a population the detector was not
consulted about — which, after this revision, is the whole dataset, sampled
directly in Round 0. That is the change: the first version tried to escape the
circularity by sampling the generator's *rejects*, but a reject population is
still drawn by the generator, so its blind spot defines the frame. Round 0
escapes it by never asking the model anything.

---

## 7. Rejected-population QA — a DIFFERENT measurement, only if Round 1 runs

**This section no longer describes the first measurement.** It measures the
**candidate generator's recall**, and it only exists if Round 1 was triggered.
Its denominator is a model-defined population, so it can never be quoted as the
dataset's missing-ball prevalence — that number comes from Round 0 alone.

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

**The reviewer's task is the same full-image sweep as Round 0** — *"is there a
football visible in this image that is not annotated?"* — so the two are directly
comparable. What differs is the frame, and that difference is the whole point: a
positive here is a ball the **generator** missed, and the rate estimates
generator recall on the population it dismissed. It is **not** independent of the
detector; nothing sampled from a detector-defined frame can be. Round 0 is the
independent measurement, and this one is read against it, never in place of it.

---

## 8. Tiny-ball coverage — avoiding a bias toward easy balls

The failure mode to design against: a QA that finds only the balls that were easy
to see, concludes the dataset is fine, and hides the small-ball problem that is
the dataset's entire value.

These countermeasures apply to **Round 0 first**, because Round 0 is the number
that will be quoted.

1. **Sample images, not boxes.** Sampling boxes can only find problems where a box
   already exists. Whole-image review is the only way a 4 px unlabelled ball can
   be found. This is why Round 0's unit — and its endpoint — is the image.
2. **Mandatory zoom in the UI**, with the reviewer able to pan at 4–8×. A 4 px ball
   is invisible at fit-to-window on a 1280×720 image.
3. **Do not rank the Round-0 sample by anything** — not by detector score, not by
   GT count, not by any prior. Ranking is for the Round-1 queue; Round 0 must be
   a probability sample or it estimates nothing. Presentation order is the
   seeded random draw order, so reviewer fatigue cannot align with split, run,
   or any dataset grouping.
4. **Report found balls by size bucket** (≤5, ≤8, ≤12, >12) and treat the buckets
   separately in the escalation rule. Three missed 4 px balls mean something very
   different from three missed 30 px balls.
5. **Force coverage of hard conditions** in the **Round-1** candidate queue, never
   in Round 0: edge-of-frame regions, balls within a player box
   (occlusion/adjacency), and high/wide broadcast shots (identified by small
   median player height). Round 0 gets none of these — deliberately
   over-representing hard frames would destroy equal inclusion probability and
   bias p̂ upward, and p̂ is the whole point of Round 0.
6. **Motion blur and partial occlusion** cannot be detected reliably from metadata
   here. They are handled by the reviewer's `UNSURE` answer, and an image marked
   `UNSURE` is counted separately — never folded into `NO MISSING BALL`. §10 puts
   a ceiling on the UNSURE rate for exactly this reason.
7. **Do not show the reviewer the detector's opinion** during Round 0, even as a
   hint. Round 0's independence is a property of the *review*, not only of the
   sampling: a reviewer primed by a model's proposals inherits its blind spots.

---

## 9. Workload options

Estimates assume ~30 s per Round-0 image (a zoomed whole-image sweep is slower
than judging a proposal) and ~15 s per Round-1 candidate image. 300 × 30 s ≈
2.5 h.

**Round 0 is now the first and mandatory unit of work in every option.** The
options differ in what happens *after* it, and none of them can be started
before it. Round 0 is 300 images regardless — a smaller Round 0 does not save
much time and destroys the interval that is its entire purpose.

### MINIMAL
- **Round 0: 300 images.** ~2.5 h
- Then `OVERSIZED_BALL_REVIEW` (72) + `BALL_OVERLAP_REVIEW` (3). ~20 min
- **No detector run at all**, whatever Round 0 says.
- Effort: ≈ **3 h**
- Defensible: the dataset's image-level missing-ball rate with an exact CI, and
  the suspicious-geometry populations reviewed by a human.
- **Not** defensible: any claim that discovered defects have been *found and
  fixed at scale*. If Round 0 shows a problem, MINIMAL measures it and leaves it
  in place.

### BALANCED — recommended
- **Round 0: 300 images.** ~2.5 h
- Lock and report. Then the 72 + 3. ~20 min
- **Then, only if Round 0 justifies it:** Round 1 candidate queue, detector @
  conf ≥ 0.03, filtered and ranked. Expect roughly **300–500 candidate images**
  (role triage produced 4,153 candidates over 1,170 images at comparable recall,
  but balls are ~1 object per frame rather than 22, so the yield should be far
  lower). ~2 h
- Effort: **~3 h if Round 0 is clean; ~5 h if it triggers Round 1.**
- Defensible: a measured prevalence with CI **and**, when needed, a corrected
  dataset. Size-bucketed reporting throughout.
- Not defensible: a prevalence claim below ~1%, or any per-run claim — the
  sample was not allocated by run, so subgroup counts are incidental and small.

### HIGH-CONFIDENCE
- **Round 0 extended to 600 images** — 0/600 → **[0.00%, 0.41%]**, i.e. at most
  5 of the 1,232 images. (The binomial reference would say 0.61%; sampling half
  the population is where the finite-population correction really bites.)
  ~5 h
- Plus a **100-image double-review** of Round-0 images by the same reviewer on a
  different day, to estimate the **reviewer miss rate**. A human sweeping for a
  4 px ball also misses some, and an unmeasured reviewer miss rate silently
  floors the whole estimate — this is the only measure that bounds it. ~1 h
- Then 72 + 3, then Round 1 at conf ≥ 0.01 with a **second independent proposal
  source** (`yolov8n.pt` COCO `sports ball`, different training distribution,
  different blind spots), unioned. ~4 h
- Then the §7 rejected-population QA to measure the generator's recall. ~2 h
- Effort: ≈ **12 h**
- Defensible: prevalence below ~0.6%, per-size-bucket rates, a stated reviewer
  miss rate, and a separately measured candidate-generator recall.

**Recommendation: BALANCED.** Its first three hours are identical to MINIMAL's,
so the choice between them can be deferred until Round 0 has actually reported —
which is the point of measuring first. HIGH-CONFIDENCE is the right shape, but
its extra nine hours buy precision that only matters if Round 0 finds something,
and BALANCED's escalation path leads there anyway.

---

## 10. Stopping and escalation rule — fixed before any review

Set now, on the **primary endpoint** (positive images / 300), so the result
cannot be rationalised afterwards. `n = 300`, SRSWOR, exact finite-population
interval.

| Round-0 result | Action |
| --- | --- |
| **0 positive images** | **Stop.** Report [0.00%, 1.06%] — at most 13 of the 1,232 images. No Round 1. Experiment D proceeds. Review the 72 + 3, which are geometry questions, not prevalence ones. |
| **1–3 positive (≤1.0%)**, all found balls >12 px | Correct later if desired. **No Round 1.** Report the rate with CI. The defensible statement is: **"no evidence from Round 0 of a tiny-ball-specific missing-label problem."** **Do not claim a cause.** The reason those large balls were missed is unknown — fatigue, ambiguous frames, a systematic rule the original annotators followed, or chance — and Round 0 cannot distinguish them. Establishing a cause requires a separate investigation. |
| **1–3 positive, any found ball ≤8 px** | Correct, and **extend Round 0 to 600**. A small ball missed even once at n=300 bears directly on the one use case this data was acquired for. Round 1 is triggered if the extension confirms. |
| **4–9 positive (1.3–3.0%)** | Correct, extend Round 0 to 600, characterise by run / size / region, and **trigger Round 1** as the correction mechanism. |
| **≥10 positive (≥3.3%)** | **Stop reviewing and investigate.** A systematic pocket, not noise. Characterise it, then build a *targeted* Round 1 queue from the pattern — exactly what the 6.40% role finding produced in the 6,684-box sweep. Experiment D waits. |
| **`UNSURE` > 15 (5%)** | The instrument, not the dataset, is the problem. Stop, review the UNSURE images together, and fix the zoom / guidance / criteria before any number is quoted. A high UNSURE rate makes the positive rate a lower bound and the CI meaningless. |

Threshold justification. **3.3%** is roughly half the **6.40%** role rate that
was judged serious enough to force a full retrospective sweep, so it is a
defensible line for "systematic" in this project's own precedent. **1.0%** is the
level below which a residual defect is smaller than the reviewer miss rate that
BALANCED does not measure, so claiming better would be claiming precision we do
not have. The **size split at 8/12 px** is not arbitrary: 37.5% of labelled balls
are ≤8 px and 76.7% are ≤12 px, so a miss in those buckets is a miss in the
dataset's centre of mass, while a miss above 12 px is in its tail.

**Findings stay in the measured rate after correction.** Round 0 is frozen at a
recorded log fingerprint before any correction is promoted.

**Errors found during QA stay in the measured rate even after correction.**
Correcting them does not un-find them; removing them from the denominator would
be measuring the dataset after fixing exactly the parts we looked at.

---

## 11. Tools that would need to be built

Ordered by when they are needed. **Nothing below Round 0 gets built until Round 0
has reported.**

| Order | Tool | Purpose | Reuses |
| --- | --- | --- | --- |
| 1 | `kb_ball_qa_sample.py` | Draw the seeded SRSWOR Round-0 sample; write the manifest with the population fingerprint, N, n, seed, inclusion probability, the ordered image IDs and their descriptive metadata | `kb_build_review_package` sampling |
| 2 | `kb_ball_qa_server.py` | Round-0 review UI: whole image, GT drawn, zoom to 8×, three answers, multi-object drawing | `kb_geometry_repair_server` almost entirely — drawing, zoom, image-coordinate conversion, `kb_images` resolver, preflight |
| 3 | `kb_ball_round0_report.py` | Fold the log, compute p̂ and the exact finite-population interval, emit the **frozen** report bound to a log fingerprint; `--interim` gives counts with no CI; refuses on a stale or incomplete snapshot | `kb_second_pass_gate` staleness guards |
| 4 | `kb_ball_geometry_review.py` | `OVERSIZED_BALL_REVIEW` (72) + `BALL_OVERLAP_REVIEW` (3) | tool 2, different queue |
| 5 | `kb_ball_candidates.py` | *Round 1 only.* Frozen detector at a low floor, matched to GT, ranked queue | `kb_role_triage` matching logic |
| 6 | `kb_ball_gate.py` | Ball-specific gate conditions before any new export is promoted | `kb_second_pass_gate` structure |
| 7 | extend `kb_export_v2.py` | Ball additions / removals / repairs as new change kinds under contract v2 | the existing clause structure |

**Round-0 UI answers** are exactly the three in §5.3 — `NO MISSING BALL`,
`MISSING BALL` (draw every one), `UNSURE`. The richer per-box vocabulary
(`FALSE BALL`, `BAD BALL BOX`, `EXISTING BALL OK`) belongs to tools 4 and 5, not
to Round 0: Round 0 asks one question with one denominator, and mixing box
verdicts into it would blur the endpoint it exists to estimate.

Same rules as the last three tools throughout: append-only events, human geometry
only, no model proposal ever becomes GT, provenance on every change.

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

**Recommendation: run Round 0 — the 300-image model-independent sample, about
2.5 hours — before Experiment D.** If it returns 0–1 positive images, proceed to
Experiment D immediately and treat Round 1 as optional cleanup. If it returns
more, Round 1 has become necessary and you will know that before spending compute
on an experiment whose result you could not have interpreted.

That inverts the usual order deliberately: **measure first, correct second.** The
role cleanup did it the other way round and the QA is what revealed that the
correction had been incomplete.

---

## 14. Order of operations

Each step's output is the next step's input, and three of the boundaries are
hard: the snapshot must be frozen before sampling, the measurement must be locked
before any correction, and Round 1 must not exist before Round 0 has reported.

| # | Step | Gate |
| --- | --- | --- |
| **1** | **Freeze the current `repaired_export` snapshot.** Record file hashes and the manifest's `decisions_sha256` as the measurement's baseline. | Already true today: verified §1, `PROMOTED == CHECKED STATE`. |
| **2** | **Generate the Round-0 sample** — 300 images, model-independent, seeded SRSWOR at π = 300/1232 for every image, manifest written. | Must bind to the step-1 fingerprint. |
| **3** | **Human review**, three answers, whole-image sweep, mandatory zoom, every missing object drawn. | Append-only; no correction promoted. |
| **4** | **Lock and report the measurement.** p̂, exact CI, object count, size buckets, run/view description, UNSURE count. Frozen at a log fingerprint. | **Nothing is corrected before this line.** |
| **5** | **`OVERSIZED_BALL_REVIEW` (72) + `BALL_OVERLAP_REVIEW` (3).** Human verdict per box; size is suspicion, not authority. | After step 4 only, so corrections cannot contaminate the baseline. |
| **6** | **Round 1 — model-assisted candidate generation**, *only if step 4 justifies it* under the §10 rule. | Correction mechanism, never evidence. |
| **7** | **Correct all findings into a NEW derived export.** Contract-v2 gated; `repaired_export/` is immutable input. | Gate must pass before promotion. |
| **8** | **Optional post-correction QA** — a fresh sample against the new export, and/or the §7 rejected-population QA for generator recall. | Separate measurement, separate denominator. |
| **9** | **Experiment D.** | Runs after step 4 if Round 0 is clean; after step 7 if it was not. |

Steps 1–4 are the ~2.5 hours that unblock Experiment D. Steps 5–8 are
conditional, and step 6 may never happen at all.
