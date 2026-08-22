# EyeCU 4.0 — Final Presentation Outline

## 1. Problem statement
- Detect/track players, goalkeepers, referees, ball in match video.
- Assign teams from jersey colour; produce per-track statistics.
- Fully offline, locally trained detector.

## 2. Final architecture
- Two branches merging only at PlayerBallAssigner:
  - Human branch: best_A_960.pt (YOLO26s@960; player/goalkeeper/referee)
    → CBIoU human tracking (ball never enters CBIoU) → team assignment.
  - Ball branch: SN3D_BASE, yolo-sn-ball.pt (YOLO11l@1280, ball only)
    → BallTemporalSelector v1.
- Both branches → PlayerBallAssigner → possession/statistics → reports.
- See [PIPELINE_DIAGRAM.md](PIPELINE_DIAGRAM.md).

## 3. Detector / human / ball path
- Human classes: strongest, most reliable signal in the system.
- Ball: production ball detection uses the separate SN3D_BASE YOLO11l ball
  branch, not EyeCU's own detector — weakest, most variable path in the
  system throughout every evaluation stage.
- BallTemporalSelector v1 sits on the ball branch only, resolving each
  frame's ball state (observed / recovered_low_conf /
  interpolated_short_gap / unknown). Development-only evaluation.

## 4. Tracking path
- CBIoU is human-only, the default association backend for the human
  branch (legacy ByteTrack available); the ball does not enter CBIoU.
- Development evaluation only — no held-out tracking/association benchmark.
- Team assignment runs on the human branch, implemented but unvalidated
  against held-out labels.

## 5. Possession / calibration limitations
- Possession: closed-limitation, not held-out validated.
- Calibration: speed/distance are uncalibrated units, never validated
  against real-world reference — do not read as m/s or km/h.
- Speed/distance/events: unsupported, deferred.

## 6. Evaluation methodology
- Frozen TEST split, 3 sequences, held out from training/validation.
- Fixed shared metric implementation (IoU, 101-point AP) across milestones.
- Access-state artifacts track TEST reads / GT changes / inference runs per
  milestone for auditability.

## 7. The M5 → M5.1 correction story
- M5: frozen detector run once on TEST; Como scored anomalously near zero.
- Root cause: Como GT was transcribed via grid-reading, not mouse-drawing —
  an annotation-process defect, not a detector failure.
- M5.1: blind re-annotation of Como (tool never showed old GT/predictions/
  IoU); QC clamp; merged with untouched manchester/youth_2 GT; re-scored
  against the same frozen M5 predictions (zero new inference).
- Result: pooled mAP50 0.2616 → 0.6175 — confirms GT-defect hypothesis.
- M5.1 reported transparently as corrected-GT evaluation, not a sealed
  one-shot test.

## 8. Final metrics
| metric | value |
|---|---|
| mAP50 (pooled) | 0.6175 |
| mAP50-95 (pooled) | 0.2697 |
| Player | 0.9220 / 0.3861 |
| Goalkeeper | 0.5218 / 0.2056 |
| Referee | 0.6906 / 0.3145 |
| Ball | 0.3357 / 0.1726 |

## 9. Production-ready vs development-only
- Production-ready (held-out supported): human detector.
- Supported but weak: raw ball detector.
- Development-only: CBIoU, BallTemporalSelector.
- Development-measured (not held-out): team assignment — 46/46 on a
  post-freeze NON-TEST benchmark. Unvalidated: calibration.
- Closed-limitation: possession (goalkeeper may now be the recorded
  possessor; team credit never fabricated for them).
- Unsupported/deferred: speed, distance, events.

## 10. Post-freeze development findings (NON-TEST, after closure)
- Team-assignment benchmark: legacy assigner 46/46 (100%) vs. two
  lightweight alternatives, both weaker on at least one match — legacy kept.
- Contamination finding: 10/57 human-labelled tracks were `MIXED_TRACK`
  (identity changed mid-track) — development evidence only, not a global
  tracking error rate or a same-team ID-switch measurement.
- Automatic tracklet-consistency guard: designed and evaluated against a
  frozen adoption gate; **no candidate passed** → **not adopted**. Raw
  CBIoU output is unchanged. Explains why the known Bayern track #4 issue
  is documented, not silently patched.
- Output-FPS default fixed (uses the pipeline's real effective FPS, not a
  fixed guess).

## 11. Final limitations
- Ball remains the weakest class under every evaluation, corrected or not.
- Only frame-level detection has held-out validation; tracking/possession/
  team assignment/calibration do not.
- Headline number reflects one documented GT-repair cycle, not an untouched
  sealed test — always cite as such.
- Track contamination (§10) is a measured, unresolved development
  limitation, not silently corrected.

## 12. Closing
- **Verdict: B — Defensible final project result with material held-out
  generalization limitation.**
- Project status: COMPLETE. No M6. Post-freeze NON-TEST development
  (§10) does not reopen M4/M5/M5.1. No further experiments planned.
