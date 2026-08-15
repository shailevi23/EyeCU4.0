# Ball annotation work — session log

Ten working sessions, 2026-08-13 to 2026-08-15, from "plan a ball QA" to a
frozen Stage-A experiment design that is blocked on hardware rather than on
method. Every figure below is read from a committed artifact; nothing is
recalled.

The through-line is a single question that kept changing shape: **how many
footballs is this dataset missing, and does it matter?** The answer moved from
28% to 8.3% to "depends what you mean by football" to "we still cannot measure
the thing we would need to measure".

---

## Session map

| # | Commit | What it settled |
| --- | --- | --- |
| 1 | `6202bc9` | Measure before correcting — plan only |
| 2 | `b6fbe2e` | Round 0 built: SRSWOR, 300 images, one inclusion probability |
| 3 | `181c7e5` | Interval matched to the design; interim reports no CI |
| 4 | `7ac120e` | Ontology revisit: what *kind* of ball was each finding? |
| 5 | `048bbe2` `b4b735a` `d23da40` `034baa4` | GT flag channel, selection fix, build stamp, NEXT IMAGE |
| 6 | `042ea1d` | Ontology written down; pp census built |
| 7 | `e0a0697` | Model proposals; "reviewed" stops meaning "clean" |
| 8 | `39dc745` `b5f5846` | Calibration 23%; tiling 27.4%; both rejected |
| 9 | `fdbb815` | Stratification; a claim of mine retracted |
| 10 | `902851d` `016adb0` `b44b0e3` | Pre-training audit; three more corrections |

---

## 1 — Measure first, correct second

The plan's first draft sampled images the *candidate generator* rejected. That
estimates the generator's recall, not the dataset's defect rate: the population
is model-defined, so every inclusion probability is conditional on detector
behaviour and the denominator is not the dataset.

**Round 0 was redesigned to consult no model at all.** Candidate generation
became a conditional Round 1.

The precedent that forced this: the role cleanup answered 4,153 of 4,153
candidates, and a QA of what the triage *rejected* still found **6.40% missed**.
Completing a queue is evidence about the queue.

## 2 — The sample

Stratified allocation was replaced with **SRSWOR, n=300 from N=1,232, every
image at exactly 300/1232**. The stratified version was self-weighting only to
within integer rounding (π ranged 0.2000–0.2667), which makes the estimator
"positives/300, approximately" — and every later reader has to decide whether
the approximation matters.

Run, split and GT state are still recorded, as *description*. A test proves they
are inert: drawing from a population with those fields blanked returns the
identical 300 images.

## 3 — The interval

Two reporting corrections, both from the user.

**The interval must match the design.** Clopper–Pearson is exact for sampling
*with* replacement from an infinite population. 300 images were drawn *without*
replacement from exactly 1,232, so the count is hypergeometric. Inverting that
distribution gives endpoints that are integer counts of images — which is what
the estimand actually is.

**`--interim` must not print a CI.** It was showing `0/300 → [0.00%, 1.22%]`
beside a caption explaining the 300 unanswered images were not negatives. The
caption does not survive being quoted.

> A property that looks like a bug and is not: at x=3 the lower bound is **M=4**,
> above the observed count. If the population held exactly 3 positives, drawing
> all 3 in a 300-image sample has probability 0.0143 — itself a tail event.

**Round 0 result, frozen:** **84 / 300 = 28.0%**, exact finite-population 95% CI
**[23.62%, 32.71%]** = **[291, 403]** images of 1,232. 128 missing footballs.
`sha256 abeda1ff…`

## 4 — What kind of ball?

The reviewer noticed during the sweep that many of the 128 were ball-boy balls,
spares behind the goal, touchline balls. If so, "28% missing" and "28% active-ball
failure" are very different claims and only the first is supported.

**Result: 25 ACTIVE, 103 NON_ACTIVE, 0 UNSURE.**

- Active-ball defect rate: **25/300 = 8.3%** — a derived secondary endpoint, not
  a replacement for the frozen 28%.
- **70.2% of the 84 positive images were NON_ACTIVE only.**
- pp any-missing 55.0% vs plain 7.6%; but pp *active* 11.6% vs plain 5.8%. The
  pp problem was mostly spares.
- **31.2% of zero-ball images were missing the active ball**, against 2.1% where
  a ball was already annotated.

Two design points worth keeping: the queue was built from the **effective** fold
(six images were re-answered; the raw log would have yielded 135 objects, not
128), and object identity is `sha256(image + geometry)` rather than list
position, so a re-answer cannot silently reattach an answer to a different ball.

## 5 — Four UI corrections

Recorded because each was a real defect, not polish.

**`n` was left unbound on purpose.** It reads as both NEXT and NON-ACTIVE;
either binding would mislabel real findings without ever failing.

**The blue GT was unclickable.** `#ov` had no `position`, so it sat in normal
flow *below* the image and made the stage taller than the picture; `#hit`'s
`inset:0` then resolved against that taller box and every click mapped to the
wrong coordinate. Fixed, plus `[`/`]` stepping, because a 4–8 px ball is two or
three screen pixels at fit zoom.

**The running server was three commits behind.** The GT controls were missing
from the live UI and the code was fine — the process had booted before that code
existed. A build stamp was added. Its first version hashed `__file__` at request
time, so a stale server advertised the *new* hash while serving the *old* page —
worse than no stamp. It now hashes the in-memory page.

**`J` skipped everything already answered**, so with 128/128 done there was no
way to page back. Added `M`/→ NEXT IMAGE.

> I also killed processes on port 8745 by port number without checking
> ownership, and one was the user's. Nothing was lost — every answer is in the
> append-only log — but it was careless.

## 6 — The ontology, written down

The source is **MIXED**: 22 existing annotations are human-confirmed spare or
ball-boy balls, so it is not ACTIVE_ONLY; 103 other visible spares were
unannotated, so it is not ALL_VISIBLE. With no consistent rule in the data, any
tool inferring the ontology could justify either answer.

    BALL_DETECTOR_ONTOLOGY      = ALL_VISIBLE_PHYSICAL_FOOTBALLS
    ACTIVE_MATCH_BALL_SELECTION = DOWNSTREAM_TEMPORAL_SELECTOR

The load-bearing clause is the negative one: **`EXISTING_NON_ACTIVE_BALL_GT` is
provenance, never a deletion instruction.** A later exporter reading it as
"remove" would delete 22 valid annotations while believing it was cleaning up.

pp membership is read from the **filename**, not the ledger run label — one pp
frame has no run recorded, so a run-based population gives 485/356 instead of
486/357.

## 7 — Model proposals, and a naming correction

The user's correction: reviewed images are **`HUMAN_REVIEWED_PARTIAL`**, never
gold or clean. A human-drawn ball is a confirmed positive; existing blue GT is
inherited and unvalidated; `NO_MISSING_BALL` says the reviewer found nothing to
*add*, not that what is there is right. One existing annotation turned out to be
a player.

488 unmatched proposals were generated over the 101 unresolved images. Matching
is **centre distance, not IoU** — at 4–8 px a one-pixel offset moves IoU by tens
of points.

The residual QA could not be built as specified and the file says so: the
intended frame was images the generator proposed *nothing* in, which at conf
0.03 is **2 of 101**. Raising the threshold would manufacture a pool by
discarding real recall, and the result would measure an artefact. Frame widened
to 20 images across two labelled strata; 40 was not available.

## 8 — Calibration, and the end of model assistance

Against the strongest reference available — **248 footballs a human drew** —
the generator recalls **23.0%** at its lowest threshold. No threshold reaches
50%, let alone the 80% target. Tiering was rejected: the shortfall is not in a
tail that can be discarded, it is everywhere.

A second detector (`yolov8n` COCO sports-ball) covered **1 of 35** — 2.9%. Not
complementary, so unioning adds false proposals and no recall.

Tiled inference was then tested properly — 2×2, 3×3, a 2× pyramid, 20% overlap,
every union:

| method | recall | unmatched proposals |
| --- | --- | --- |
| FULL | 23.0% | 1350 |
| 2×2 | 15.3% | 794 |
| 3×3 | 12.1% | 552 |
| PYRAMID | 25.8% | 1384 |
| **ALL** | **27.4%** | 1603 |

**MODEL_ASSISTED PP COMPLETION NOT VIABLE.**

Before trusting a negative that consequential, the pipeline was checked against a
reference the detector *can* see: the same 3×3 pyramid covered **20 of 27
existing blue annotations (74.1%)** on 30 images. Coordinate mapping, crop
handling and dedup all work. 19 tests pin tile coverage, round-trip mapping and
order-independent dedup.

## 9 — Stratification, and a retraction

I had written that the missed balls are "an object the detector was trained to
treat as background". **That was not measured, and it is retracted.**

- The source *does* annotate non-active balls (22 confirmed) — no exclusion
  policy to inherit.
- The detector recalls **50.0% of source GT in the same top band** where it
  recalls **29.8% of human additions** — not blind to that region.
- No training set was ever inspected.

Scale is ruled out too, and more sharply than the aggregate showed. Within the
top band, human additions are **larger** than source GT — **8.25 px vs 6.25 px** —
yet recalled far worse. If resolution were the barrier that would be reversed.

Supported statement: **a distribution/context shift whose cause is not
established.**

Temporal duplication is real but smaller than my spot check suggested: 55.6% of
additions sit in a group spanning ≥2 frames, 24.6% in groups of ≥4, and 248
boxes resolve to ~156 distinct groups (110 singletons) — **1.59 boxes per
group**.

Training concentration: **362 of 376 additions (96.3%) are pp**. Checked against
review effort (pp is 79% reviewed, plain 23%) using Round 0's unbiased random
sample: **0.88 balls/image pp vs 0.08 plain**, an 11× gap. Real, not an artefact.

## 10 — Pre-training audit, and three more corrections

**Ball semantics are UNKNOWN, not ALL_VISIBLE.** `LABELING.md:148` says *"Ball
only when visually identifiable"* — visibility, never *which* ball. Its ball-boy
exclusion covers the **people**. Completeness **NOT_MEASURED** for TRAIN and VAL
alike; no ball QA has ever run on EyeCU's own data.

**VAL cannot score the A11 increment.** 208 images, 111 balls, **0 multi-ball
frames**. `RESULTS.md` had already said so: *"a correctly detected second
football is already scored as a false positive here."* Temporal GT is the same —
`ANNOTATION.md:12` instructs labelling a second football, but all 104 frames
carry 0 or 1 box.

Three corrections, two of them mine:

- **Source holdouts were being consumed.** The audit declared keremberke
  train/valid/test preserved, then proposed feeding all 300 Round-0 images into
  training. Restricting to source-train drops the arms to **207 images** and the
  additions from 25/103/128 to **22/74/96**.
- **The optimizer claim was wrong.** I said label counts could make
  `optimizer=auto` resolve differently. Verified against local 8.4.116,
  `trainer.py:299`: `iterations = ceil(len(dataset)/max(batch,nbs)) * epochs` —
  image count, not boxes. All arms share one image set, so auto resolves
  identically anyway. Pinning `AdamW / 0.00125 / 0.9 / 0.0` is for
  reproducibility, not to prevent divergence. (`accumulate` is 21, not the 22 I
  wrote.)
- **The split is not video-disjoint.** IDs are unique but fully interleaved:
  **49% of pp consecutive-id pairs cross a split boundary**, 46% for plain. An
  NCC probe confirms these are often the same scene — cross-split adjacent pairs
  median **0.530** with 3/40 above 0.90, against a random baseline of **0.196**
  with 0/40. No match/video identifier exists anywhere, so original-video
  identity is **UNKNOWN** and the 14-object valid supplement is a
  **PROVENANCE UNKNOWN DEVELOPMENT DIAGNOSTIC**.

---

## Where it stands

**Stage-A frozen:** 1,030 images (823 EyeCU + 207 external), A10 +22 active,
A11 +74 non-active (96 total), epochs 80, patience 0, AdamW/0.00125/0.9/0.0,
batch 3, final-epoch primary readout.

**Blockers:**

1. **No local GPU** — torch is 2.8.0+cpu. 80 epochs × 1,030 images at imgsz 960
   is not viable, and any GPU environment differs from the Colab baseline.
2. **No metric scores the A11 increment** — 0 multi-ball frames in either VAL
   (208) or temporal GT (104).
3. **A11 is 74 boxes against 673 existing EyeCU ball boxes (~11%)** on 207 of
   1,030 images. With (2), a null result would be uninterpretable.
4. **A00's base is MIXED** — no arm may be called ACTIVE_ONLY or
   ALL_VISIBLE-clean.

**Untouched throughout:** frozen Round-0 result `abeda1ff…`, promoted export
`54128d4f…`, ontology policy, sealed TEST. No training performed. 1,065 tests
pass.

---

## What the sessions kept teaching

**Completion is not evidence.** A finished queue proves the queue was answered.
Twice now — 4,153 role candidates, then 488 ball proposals — the interesting
number came from the population that was *rejected*.

**Say what a number cannot support.** The 28% and the 8.3% are both true and
measure different things. The interim CI, the tier boundaries, the 40-image
residual sample and the "~98 missing footballs" extrapolation were each removed
or relabelled because they claimed more than the data carried.

**Tests decay toward snapshots.** `KNOWN_MODES` needed hand-editing every pass;
three Round-0 tests encoded "the round is not finished"; two PP tests asserted
absolute event counts on images the human was still reviewing. Each was rewritten
as the invariant it was reaching for.

**Verify the negative before reporting it.** Both major negatives — 23% recall
and 27.4% tiled — were checked against a reference the detector *can* see. A
coordinate bug would have looked identical.
