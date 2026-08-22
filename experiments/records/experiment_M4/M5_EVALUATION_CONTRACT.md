# M5 -- one-shot sealed TEST detection-metrics evaluation contract

Prepared as part of M4 Section 11. **Not executed.** This document commits
to the exact metrics, scope, and reporting rules before any TEST prediction
is ever run, so the eventual M5 run cannot be shaped by having already seen
results.

## Scope

Detection only. This contract does **not** claim, compute, or imply
accuracy for tracking, possession, team assignment, calibration, speed, or
distance -- those remain governed by their own existing status records
(`experiments/records/experiment_M2/calibration/M2_1_CALIBRATION_STATUS.json`,
`experiments/records/experiment_P1/P1_POSSESSION_RESULT.json`, the M3
`component_status` block) and are unaffected by M5.

## Input

- Frozen system: `experiments/records/experiment_M3/SYSTEM_FREEZE_MANIFEST.json`
  (re-verify identity immediately before running -- see Section 12 of this
  same M4 pass, and repeat that same verification again at the start of M5
  itself before touching TEST).
- Frozen GT: `TEST_DETECTION_ANNOTATIONS.json`
  (sha256 `4702b60fbdb173773e6bc7246d45587c1446093559c96792d3ae864e4d6896cb`),
  60 frames, `player`/`goalkeeper`/`referee`/`ball` boxes,
  `ALL_VISIBLE_PHYSICAL_FOOTBALLS` ball ontology.
- Predictions: the frozen production detector(s) run once, in inference-only
  mode, on exactly these 60 frames -- `best_A_960.pt` for
  player/goalkeeper/referee, the SN3D ball branch for ball -- through the
  same code path `full_pipeline.py`/`run_pipeline.py` already use in
  production. No retraining, no threshold tuning, no per-class confidence
  adjustment based on any TEST result.

## Metrics (fixed, no substitutions)

- **Precision, Recall** per class, at the production `accept_confidence`
  thresholds already frozen in `SYSTEM_FREEZE_MANIFEST.json`
  (`human_detector.accept_confidence = 0.25`,
  `ball_detector.accept_confidence = 0.25`) and IoU >= 0.50 for a match.
- **mAP50** and **mAP50-95** per class, standard COCO-style IoU sweep,
  computed only from the frozen detector's raw confidence-ranked output --
  no confidence threshold substitution to make a class look better.
- All four reported **per class** (player, goalkeeper, referee, ball) and
  **per match** (como_2-0_sassuolo, manchester_city_v_liverpool, youth_2),
  never only pooled -- a pooled-only number would hide a match or class
  that fails while the average looks fine.
- Multi-ball frames (3 of 60, per `GT_DESCRIPTIVE_STATS.json`) are scored under the same
  `ALL_VISIBLE_PHYSICAL_FOOTBALLS` rule the GT was built under: every GT
  ball box is a separate ground-truth instance to match, not merged into
  one "the ball" slot.

## Explicitly forbidden in the M5 report

- No invented confidence interval, standard error, or significance test on
  a 60-frame, 3-match sample -- report the point estimate and the raw
  counts (TP/FP/FN per class) it came from, nothing more.
- No accuracy claim for anything beyond detection (see Scope).
- No comparison against VAL-set numbers presented as if they were
  equivalent -- TEST and VAL are different frames under a different
  provenance chain; a side-by-side table is fine, an implied
  "improvement"/"regression" framing across the two is not, since VAL was
  used during development and TEST was not.
- No post-hoc exclusion of a frame or class because its number looks bad.
  All 60 frames, all 4 classes, are scored and reported, every time.

## Output

A single frozen `M5_DETECTION_RESULT.json` (P/R/mAP50/mAP50-95 per class
per match, TP/FP/FN counts, the exact detector confidence/IoU thresholds
used, and the frozen manifest hash the predictions were produced under),
plus a short markdown summary. Both hashed and recorded the same way every
other M4/M3 artifact has been, with supersession (never silent overwrite)
if M5 is ever repeated.

## Gate

M5 may only start after a fresh M3 freeze-identity re-verification (source
tree SHA, manifest SHA, both weight SHAs) passes, exactly as Section 12 of
this M4 pass just did. If any of those four hashes has changed since this
M4 close, M5 does not run until a new freeze is produced.
