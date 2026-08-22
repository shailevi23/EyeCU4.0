# EyeCU 4.0 — Final Project Report

**Status:** COMPLETE
**Final evaluation:** M5.1 corrected held-out detection evaluation
**Verdict: B — Defensible final project result with material held-out generalization limitation.**

This report is documentation only. It draws exclusively on artifacts already
frozen in `experiments/records/` and `docs/`. No models were run, no metrics
were recomputed, and no annotations or production code were touched in
producing it.

---

## 1. Problem statement

Detect and track players, goalkeepers, referees and the ball in football
match video; assign team from jersey colour; and produce per-track
statistics (position, uncalibrated speed/distance, possession). The goal was
a fully offline pipeline running on a locally trained detector, evaluated
honestly against held-out match footage — including reporting negative and
partial results rather than only the favorable ones.

## 2. Final architecture

The production system is two separate branches that merge only at the
player-ball assignment stage:

```
video ---> HUMAN BRANCH: best_A_960.pt (YOLO26s @960; player/goalkeeper/referee)
                 -> CBIoU HUMAN tracking (ball does NOT enter CBIoU)
                 -> team assignment
video ---> BALL BRANCH: SN3D_BASE, yolo-sn-ball.pt (YOLO11l @1280; ball only)
                 -> BallTemporalSelector v1 (observed / recovered_low_conf /
                    interpolated_short_gap / unknown)

  both branches -> PlayerBallAssigner -> possession/statistics
                -> annotated video + JSON reports
```

Three semantic human classes are preserved end to end: `player`,
`goalkeeper`, `referee`. Goalkeeper is never collapsed into `player` — its
kit deliberately differs from its own team's, and team assignment excludes
it. EyeCU's own detector produces ball boxes as a byproduct of training, but
**those boxes are not the production ball source** — see below.

**Human detector:** `best_A_960.pt`, a YOLO26s model trained at 960px. Two
alternatives were trained and evaluated for comparison (`best_B_1280.pt`,
more accurate on the ball but 1.7x cost; `best_C_960.pt`, a data-side
ablation that was rejected). All three are kept for reproducibility; only
`best_A_960.pt` is the production human-branch candidate.

## 3. Detector / human / ball path

- **Human classes (player, goalkeeper, referee):** the single largest
  source of reliable signal in the system. Held-out (TEST) evaluation in
  M5.1 supports this path as the strongest, most consistent part of the
  detector.
- **Ball:** production ball detection uses the separate SN3D_BASE YOLO11l
  ball branch (`yolo-sn-ball.pt`, @1280, ball only) — not EyeCU's own
  detector. This branch remains the weakest and most variable path in the
  system, consistent with the known difficulty of small, fast,
  frequently-occluded objects for this architecture (see
  `docs/research/ball_architecture_audit.md`).
- **BallTemporalSelector v1:** sits on the ball branch only, resolving a
  single reported ball state per frame (`observed`, `recovered_low_conf`,
  `interpolated_short_gap`, or `unknown`) from the SN3D branch's raw
  detections. Evaluated only in development, not against held-out TEST.

## 4. Tracking path

- **Association:** CBIoU (confidence-and-box IoU association) is
  **human-only** — the default tracker backend for the human branch (a
  `legacy` ByteTrack path exists for rollback); the ball never enters
  CBIoU. CBIoU has been evaluated in development only — it has no held-out
  TEST evaluation of its own, since TEST evaluation to date has measured
  per-frame detection quality (mAP), not multi-frame tracking/association
  quality.
- **Team assignment:** runs on the human branch only (jersey-colour
  clustering, goalkeeper excluded). Not held-out TEST validated. It has,
  however, since been measured on a **post-freeze, NON-TEST development
  benchmark** (two development matches, human-labelled tracks): the legacy
  assigner scored **46/46 (100%)** on the clean, single-identity tracks in
  that benchmark. That benchmark also surfaced a separate, more consequential
  finding — see "Post-freeze development findings" below. Status:
  **implemented; development-measured; not held-out TEST validated.**

## 5. Possession / calibration limitations

- **Possession:** implemented, but evaluation was closed out as a known
  limitation rather than validated against held-out ground truth. Status:
  **CLOSED-LIMITATION** — present in the pipeline, not a supported claim.
  Post-freeze, a goalkeeper may now be recorded as the ball possessor
  (`has_ball=True`) under the same geometry/threshold already used for
  field players — previously only `tracks['players']` was searched, so a
  goalkeeper could never receive possession at all. Team-control credit for
  a goalkeeper possession stays **UNKNOWN**; it is never fabricated from
  goalkeeper kit colour, since goalkeepers are deliberately excluded from
  jersey `TeamAssigner`. Team-possession accuracy itself remains
  unvalidated.
- **Pitch calibration:** speed and distance are computed in pipeline units
  that have never been validated against a real-world reference (no
  measured pitch-to-pixel calibration confirmed on held-out data). The
  README explicitly warns not to quote these as m/s or km/h. Status:
  **NOT VALIDATED.**
- **Speed / distance / events:** downstream of calibration and possession
  respectively; both remain **UNSUPPORTED / DEFERRED** for this project.

## 6. Evaluation methodology

Detection evaluation used a frozen TEST split (`manchester_city_v_liverpool`,
`youth_2`, `como_2-0_sassuolo`) held out from training and validation, scored
with a fixed metric implementation (`iou`, `ap_101point`, `evaluate` in
`experiments/records/experiment_M5/m5_metrics.py`) shared unchanged between
M5 and M5.1. Access-state artifacts (`TEST_ACCESS_STATE.json`) record, per
milestone, whether TEST images were read, whether GT was modified, and
whether new inference was run, to keep the held-out evaluation auditable.

## 7. The M5 → M5.1 correction story

- **M5** ran the frozen detector once against TEST and scored it against the
  then-current GT (`TEST_DETECTION_ANNOTATIONS.json`). Como's numbers came
  back anomalously near zero across every class, including player.
- Investigation (`GT_QUALITY_FINDING.json`, `M5_1_ERRATUM.md`) traced this to
  a GT **annotation-process defect**: Como's ground truth had been
  transcribed by grid-reading rather than direct mouse-drawing, unlike the
  other two sequences, producing systematically corrupted boxes — not a
  detector failure.
- **M5.1** performed a **blind re-annotation** of the 20 affected Como
  frames in a purpose-built local tool that never displayed the original
  GT, predictions, confidences, or IoU results — eliminating the risk of the
  correction being fitted to the detector's known outputs. QC clamped 2/480
  boundary-overshoot boxes (same mechanical defect class as an earlier M4
  clamp pass); re-run showed 0 errors.
- The corrected GT (`TEST_DETECTION_ANNOTATIONS_CORRECTED.json`) was merged
  with the untouched manchester/youth_2 records and scored against the
  **same, unchanged, frozen M5 predictions** (`RAW_PREDICTIONS.json`,
  reused byte-for-byte — zero new inference). `manchester_city_v_liverpool`
  and `youth_2` scores are exactly unchanged, confirming no
  cross-contamination.
- Result: pooled mAP50 rose from 0.2616 (M5, defective GT) to **0.6175**
  (M5.1, corrected GT), decisively confirming the GT-defect hypothesis
  rather than a detector improvement.
- **M5.1 is explicitly not a pristine one-shot sealed test.** It is reported
  transparently as a corrected-GT evaluation following a blinded, auditable
  post-hoc GT repair, using the original frozen predictions. This
  distinction is preserved in every artifact and must be preserved in any
  future citation of these numbers.

## 8. Final metrics (M5.1, authoritative)

| metric | value |
|---|---|
| **mAP50 (pooled)** | **0.6175** |
| **mAP50-95 (pooled)** | **0.2697** |

| class | AP50 | AP50-95 |
|---|---|---|
| Player | 0.9220 | 0.3861 |
| Goalkeeper | 0.5218 | 0.2056 |
| Referee | 0.6906 | 0.3145 |
| Ball | 0.3357 | 0.1726 |

The original M5 pooled metric (mAP50 0.2616 / mAP50-95 0.1351) remains
preserved in `experiments/records/experiment_M5/` for history but is
**invalid / superseded for reporting** due to the Como GT defect. M5.1's
numbers above are the authoritative detection results for this project.

## 9. Production-ready vs development-only

**Held-out (TEST) supported:**
- Human detector (player / goalkeeper / referee) — supported, strongest path.
- Raw ball detector — supported, but materially variable and the weakest
  class; do not treat as production-reliable.

**Development evaluation only (no held-out validation):**
- CBIoU association / tracking — scientific status unchanged by later
  post-freeze work (see §11a).
- BallTemporalSelector.

**Implemented but unvalidated / unsupported:**
- Team assignment — implemented; 46/46 (100%) on a post-freeze NON-TEST
  development benchmark (§11a); not validated against held-out labels.
- Possession — closed as a limitation, not validated. Goalkeepers may now
  be the recorded possessor; team-control credit is never fabricated for
  them (§5).
- Pitch/metric calibration — not validated.
- Speed, distance, events — unsupported, deferred.

## 11a. Post-freeze development findings

After project closure (§11), further **POST-FREEZE, NON-TEST** development
work measured and hardened components that were previously unvalidated.
None of this reopens or reruns M4/M5/M5.1; TEST was never accessed. Full
detail: [POST_FREEZE_SYSTEM_PATCH.md](../provenance/POST_FREEZE_SYSTEM_PATCH.md),
[experiments/post_freeze/team_assignment_v2/](experiments/post_freeze/team_assignment_v2/),
[experiments/post_freeze/tracklet_guard_v1/](experiments/post_freeze/tracklet_guard_v1/).

- **Team assignment benchmark:** 57 long-lived player tracks across two
  NON-TEST development matches were human-labelled. On the 46 tracks
  labelled as a single, clean identity, the legacy `TeamAssigner` scored
  **46/46 (100%)** against two lightweight alternative candidates (a robust
  color descriptor and a SigLIP-embedding candidate), both of which showed
  a material weakness on at least one match. Legacy was kept as the
  production default; nothing was replaced.
- **Track contamination finding:** of those same 57 tracks, **10 were
  human-labelled `MIXED_TRACK`** — the track's own visual identity was not
  consistent throughout its lifetime (evidence of an underlying tracking ID
  mixing two different people). This is development evidence from a
  specific, deliberately-selected benchmark of long-lived tracks; it is
  **not** a global tracking error rate, not a per-frame error rate, and
  does not measure same-team ID switches. One previously-observed failure
  (Bayern track #4, visibly on the wrong team in the demo video) was
  blindly human-labelled `MIXED_TRACK`, independently confirming the
  earlier color-based contamination hypothesis for that specific track.
- **Automatic tracklet consistency guard:** an automatic detector for this
  kind of contamination was designed and evaluated post-freeze (a
  no-guard baseline, a color change-point detector, and a SigLIP
  change-point detector) against a frozen adoption gate. **No candidate
  passed the gate** (required recall/false-positive/both-match trade-off
  was not met), so **no guard was adopted** — the raw CBIoU track output is
  unchanged and unmodified.
- **Output FPS default:** `run_pipeline.py --fps` now defaults to the
  pipeline's own `effective_fps` (`source_fps / skip_frames`) instead of a
  fixed `15`, fixing a playback-speed mismatch in exported video; an
  explicit `--fps` value still overrides it.

## 10. Final limitations

- Ball detection is the system's weakest and most variable class under
  every evaluation performed, including the corrected one.
- Tracking, possession, team assignment, and calibration have never been
  measured against held-out ground truth — only the frame-level detector
  has.
- M5.1's held-out numbers are trustworthy for detection specifically, but
  do not extend to any downstream (tracking/possession/speed/event) claim.
- The correction from M5 to M5.1 was necessary and is fully documented and
  auditable, but it means this project's headline detection number reflects
  one GT repair cycle, not an untouched sealed test.

## 11. No further experiments

This report and its accompanying documents close the project. No M6 is
planned. No further model runs, GT changes, or tuning are authorized under
this closure; any future work would constitute a new, explicitly-labelled
milestone, per the immutability note in `TEST_ACCESS_STATE.json`. (The
post-freeze development work in §11a is exactly that kind of
explicitly-labelled, NON-TEST milestone — it does not reopen M5/M5.1.)

## 12. Final demo

`demo_outputs/final_e2e_demo/tracked_output_final_system.mp4` — Bayern
Munich 3-1 Chelsea (NON-TEST), rendered from the existing tracks cache
(zero YOLO/SN3D inference for this render; cache-hit confirmed): 640×360,
375 frames, 12.5 fps, 30.0s, H.264 (`avc1`). The source video itself is
640×360, so remaining visual softness is source-resolution-limited, not an
export defect (see `../provenance/VISUALIZATION_PATCH_V2.md` for the codec/FPS export
fixes that were already applied). The on-screen IDs are **tracking IDs**,
not jersey numbers.

---

See also: [README_FINAL_SUMMARY.md](README_FINAL_SUMMARY.md),
[FINAL_PRESENTATION_OUTLINE.md](FINAL_PRESENTATION_OUTLINE.md),
[PIPELINE_DIAGRAM.md](PIPELINE_DIAGRAM.md), [FINAL_PROJECT_STATUS.md](FINAL_PROJECT_STATUS.md),
[POST_FREEZE_SYSTEM_PATCH.md](../provenance/POST_FREEZE_SYSTEM_PATCH.md).
