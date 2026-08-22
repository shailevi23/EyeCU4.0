# M5 FINAL REPORT -- one-shot sealed TEST detection evaluation

## Precheck

All 8 minimum blocking identity checks passed (`PRECHECK_RESULT.json`):
manifest hash, `FINAL_TEST_FRAME_LIST.json` hash, `TEST_DETECTION_ANNOTATIONS.json`
hash, `M5_EVALUATION_CONTRACT.md` hash, `labels_frozen=true`,
`production_predictions_run=false` (pre-run state), `best_A_960.pt` SHA256,
`yolo-sn-ball.pt` SHA256 -- all MATCH. Inference proceeded.

## Execution

One `TwoBranchDetector` (raw `best_A_960.pt` for player/goalkeeper/referee +
raw SN3D `yolo-sn-ball.pt` for ball, both at the frozen `accept_confidence=0.25`,
no `BallTemporalSelector`, no CBIoU tracking) forward pass per TEST frame,
directly on the 60 frozen `TEST_DETECTION_ANNOTATIONS.json` images
(`experiments/records/experiment_M5/m5_raw_predict.py`), CPU,
`torch.set_num_threads(1)`, `cv2.setNumThreads(1)`. 60 frames, 122.4s total.
Raw output: `RAW_PREDICTIONS.json`.

Separately, one fresh full-production-pipeline run per sequence (`FootballAnalysisPipeline`,
40 raw frames/sequence, `skip_frames=2`) for the structural E2E check only --
this does not touch or affect the scored detection metric.

One metric computation pass over the saved `RAW_PREDICTIONS.json` against
`TEST_DETECTION_ANNOTATIONS.json` (`m5_metrics.py`): standard COCO-style
P/R at IoU>=0.50, AP50, AP50-95 (10-threshold, 101-point interpolated),
per class, per sequence, and pooled. No threshold sweep, no confidence
substitution, no new metric definition.

## Critical finding -- GT quality defect in `como_2-0_sassuolo` (discovered post-metric, non-visually)

After computing the metrics, `como_2-0_sassuolo`'s per-class numbers were
near-zero (player recall 3.4%) while `manchester_city_v_liverpool` and
`youth_2` were plausible and internally consistent (player AP50 0.97 and
0.79 respectively). This asymmetry was investigated using **only** the
already-saved JSON numbers -- zero TEST images opened, satisfying the
"ZERO manual/vision TEST image reads" rule.

Evidence (`GT_QUALITY_FINDING.json`): `como_2-0_sassuolo`'s GT player boxes
average **578px² in area**, while the frozen detector's own raw predictions
on those same frames average **192px²** -- GT boxes ~3x larger than the
model's tight convention. `manchester_city_v_liverpool`'s GT (666px²) and
predictions (546px²) are the same order of magnitude (ratio 1.22, normal
detector variance). Spot-checked box pairs in `como_2-0_sassuolo` frame 38
confirm this is not a coordinate/scale bug (top-left corners are frequently
within 1-3px of each other) but a **systematic looseness** in the GT boxes
themselves -- e.g. IoU 0.186 for a referee box whose corner aligns within
1px but whose GT area is ~5x the matching prediction's.

**Root cause**: `como_2-0_sassuolo` was the sequence the assistant
hand-annotated in-session by reading a printed coordinate grid overlay and
estimating box corners by eye (self-documented throughout
`ANNOTATIONS_DRAFT.json`'s notes as carrying "+/-8-10px" uncertainty,
occasionally extrapolating an edge from "typical player height" rather than
the observed edge). `manchester_city_v_liverpool` and `youth_2` were
annotated by mouse-drag in the purpose-built local browser tool, which does
not share this failure mode. This is a defect introduced during M4
Section 6/7, not a bug in this M5 evaluator and not a property of the
frozen detector being measured.

**Per the M5 contract's Section 8 ("if a genuine methodological/software
failure prevents a valid result: document the exact bug and STOP")**: GT
was **not** modified, no frame/class was excluded, no rerun occurred. The
numbers below are reported exactly as computed, for all 60 frames and all
4 classes, with this finding attached rather than hidden or silently
corrected.

## Results (as computed, unmodified)

| class | GT | Pred | TP | FP | FN | P | R | AP50 | AP50-95 |
|---|---|---|---|---|---|---|---|---|---|
| player | 782 | 918 | 440 | 478 | 342 | 0.4793 | 0.5627 | 0.3125 | 0.1633 |
| goalkeeper | 31 | 27 | 6 | 21 | 25 | 0.2222 | 0.1935 | 0.1636 | 0.0780 |
| referee | 63 | 79 | 29 | 50 | 34 | 0.3671 | 0.4603 | 0.2724 | 0.1291 |
| ball (raw SN3D) | 48 | 37 | 21 | 16 | 27 | 0.5676 | 0.4375 | 0.2979 | 0.1699 |

**Pooled mAP50 = 0.2616, mAP50-95 = 0.1351 -- INVALID as a generalization
measurement**, per the finding above (one-third of the population
dominates this pooled figure with a GT artifact, not a detector property).

### Per sequence (the informative view, given the finding above)

**como_2-0_sassuolo -- NOT representative of detector quality** (GT defect):
player P=0.027/R=0.034/AP50=0.003, goalkeeper 0/0/0, referee P=0.024/R=0.040/AP50=0.017, ball 0/0/0.

**manchester_city_v_liverpool -- no equivalent defect detected, most trustworthy sample**:
player P=0.959/R=0.983/AP50=0.972/AP50-95=0.487, goalkeeper P=0.455/R=0.625/AP50=0.624/AP50-95=0.295,
referee P=0.815/R=0.815/AP50=0.781/AP50-95=0.362, ball P=0.313/R=0.313/AP50=0.106/AP50-95=0.017.

**youth_2 -- no equivalent defect detected**:
player P=0.705/R=0.851/AP50=0.793/AP50-95=0.508, goalkeeper P=1.0/R=0.111/AP50=0.119/AP50-95=0.095
(1 of 9 GT goalkeepers detected -- small-sample, high-resolution 1920x1080 footage, single positive detection),
referee P=0.545/R=0.545/AP50=0.442/AP50-95=0.301, ball P=1.0/R=0.941/AP50=0.941/AP50-95=0.655.

### Multi-ball descriptive (3 frames, `ALL_VISIBLE_PHYSICAL_FOOTBALLS`)

6 GT ball instances across the 3 multi-ball frames; 3 matched, 3 missed, 0 false positives.

## E2E structural acceptance -- PASS (all 3 sequences)

`E2E_ACCEPTANCE.json`: `overall_all_pass: true`. Per sequence: pipeline
completed (20 frames processed each after skip_frames=2 on a 40-raw-frame
clip); CBIoU never absorbed a ball box into a human track
(`ball_excluded_from_human_tracks: true` all 3); ball branch active;
`BallTemporalSelector` states all legal (`observed`/`recovered_low_conf`/
`interpolated_short_gap`/none seen illegal); possession path executed
without exception on all 3 (20/20 frames scored each); output/statistics/
provenance files written; `speed_distance_calibrated: false` and the
`_UNCALIBRATED`-suffixed fields present, confirming calibration/speed/
distance correctly stayed marked unsupported. No TEST accuracy claim was
made for tracking, possession, team assignment, calibration, speed, or
distance -- structural only, as scoped.

## Final component claims (unchanged scope, M5 contributes exactly two rows)

| component | status |
|---|---|
| Human detector (player/goalkeeper/referee) | **HELD-OUT TEST result from M5 -- but the pooled figure is contaminated; use manchester_city_v_liverpool/youth_2 per-sequence numbers, not the pooled number, as the trustworthy read** |
| Ball detector (raw SN3D) | **HELD-OUT TEST result from M5 -- same caveat** |
| CBIoU | development evaluation only (M3); structurally exercised, not accuracy-scored, in M5 |
| BallTemporalSelector | development evaluation only (M3); structurally exercised, not accuracy-scored, in M5 |
| Possession | CLOSED-LIMITATION (P1/P1.1, unchanged) |
| Team assignment | IMPLEMENTED BUT UNVALIDATED (unchanged) |
| Metric world coordinates | NOT VALIDATED (M2.1, unchanged) |
| Speed | UNSUPPORTED (unchanged) |
| Distance | UNSUPPORTED (unchanged) |
| Events | UNSUPPORTED / DEFERRED (unchanged) |

## Verdict

**C -- EVALUATION INVALID for the pooled/aggregate held-out detection
claim**, because a genuine, evidenced GT-quality defect in one of three
TEST sequences (self-introduced during M4, discovered here through pure
numeric review, not tuning) prevents that pooled number from meaning
anything about the frozen detector. This is not a software crash, but it
is exactly the class of "genuine methodological failure preventing a valid
result" the M5 contract's Section 8 anticipates. The `manchester_city_v_liverpool`
and `youth_2` per-sequence numbers show no equivalent defect and stand as
partial, informative (not pooled, not final-project-grade) held-out
evidence: on those two sequences the detector is measurably imperfect but
functional (player AP50 0.79-0.97, referee AP50 0.44-0.78, goalkeeper and
ball weaker and noisier on small per-sequence GT counts).

## What was not done (compliance)

- Zero TEST image reads by the assistant (0 GET/Read calls on any TEST frame).
- Zero annotation, contact sheets, overlays, or error galleries produced.
- Zero repo audits beyond the 8 precheck hash checks.
- Zero model/threshold tuning, before or after seeing results.
- Zero full test-suite runs.
- One prediction pass per sequence; no rerun after results were viewed.
- GT was not modified in response to the finding above.
- M4's `TEST_DETECTION_ANNOTATIONS.json` remains byte-identical to its
  frozen hash (re-verified: `4702b60f...`).
