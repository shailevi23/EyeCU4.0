# Experiment M2 — minimal defensible pitch calibration — final disposition

```
STATUS      CLOSED
VERDICT     B — M2 PARTIAL
```

Replaces the guessed `pixels_per_meter` scale path with a real, validated,
fail-closed image→pitch calibration, scoped to individually frozen
stable-camera segments. Possession (P1/P1.1) was not touched. No TEST access.
M3 not started.

## 1. What was inspected first (narrow, not a repo audit)

- `trackers/speed_distance.py` — the only place `pixels_per_meter` (default
  12.0, a guess) exists in production code. Already labelled
  `UNCALIBRATED` in its own module docstring and every output field
  (`max_speed_kmh_UNCALIBRATED`, `total_distance_m_UNCALIBRATED`), and
  `full_pipeline.py` already sets `report['speed_distance_calibrated'] = False`
  at both call sites (lines ~290 and ~461). No path in the codebase claims a
  *validated* metric result from this constant — the existing mitigation
  (labelling) was already in place before this milestone.
- `trackers/camera_movement.py` — optical-flow **pixel** pan compensation
  (`position_adjusted`), unrelated to metric scale; left untouched.
- `trackers/football_tracker.add_position_to_tracks` — players already use
  `get_foot_position` = bottom-centre of bbox for `'position'`. This is
  exactly the ground-point convention M2 requires, already the input this
  milestone's calibration module consumes — no change needed there.
- FPS handling: `full_pipeline.py` reads the real source FPS via
  `get_video_info`, derives `effective_fps` from `skip_frames`, and passes it
  into `SpeedDistanceEstimator`. No guessed FPS in the pipeline. The 3
  candidate development sequences are native 25.0 fps
  (`data/tracking_val_gt/mot/EyeCU-val/*/seqinfo.ini`).
- Existing reusable utilities found and reused rather than rebuilt:
  `tools/detect_broadcast_cuts.py` (mad+hcorr dual-signal cut detector,
  already calibrated against a human-confirmed cut) for segment cut-freedom,
  and `trackers/camera_movement.CameraMovementEstimator` (existing optical
  flow) for quantifying camera drift inside a candidate segment.
- No production homography/pitch-calibration code existed anywhere in
  `trackers/` or `tools/` before this milestone. (`experiments/soccertrack_audit/`
  and `EyeCU_external_data/soccertrack_v2/` contain third-party dataset audit
  artifacts, not production code, and were not reused.)

## 2. Guessed-scale contract

**Found:** `trackers/speed_distance.py:SpeedDistanceEstimator.__init__` default
`pixels_per_meter=12.0`; consumed at `full_pipeline.py:290` and `:461-462`.

**Corrected:** nothing needed removing — both call sites already refuse to
claim calibration (see above). What M2 adds is the thing that was actually
missing: a path that CAN legitimately claim a metric result, and only inside
a validated segment. New module `trackers/pitch_calibration.py` never
imports `SpeedDistanceEstimator` or references `pixels_per_meter` (tested:
`tests/test_pitch_calibration.py::TestNoGuessedScaleFallback`). Outside a
calibrated segment, `CalibrationStore.image_to_pitch(...)` and
`short_window_displacement_and_speed(...)` return `None` — Python's
UNKNOWN — never a number, never a fallback to a guessed constant.

## 3. Calibration contract

- **Method:** manual homography on stable-camera segments, from the
  broadcast-visible **centre circle** (IFAB-fixed radius 9.15 m, independent
  of this stadium's unknown pitch length/width — the milestone explicitly
  forbids assuming those). See `tools/m2_circle_correspondences.py`'s module
  docstring for the full projective-geometry derivation (conic fitting,
  pole/polar centre construction, conjugate-diameter axis construction).
- **Coordinate convention:** pitch metres, local frame centred on that
  segment's own centre-circle centre, using an orthogonal local basis fixed
  by that segment's own correspondences (`axis_a` / `axis_b` — see each
  artifact's `coordinate_convention` field). Segments are **not** stitched
  into one global pitch frame; a position from one segment is only
  comparable to another position from the *same* segment.
- **Player ground point:** bottom-centre of the bbox — `((x1+x2)/2, y2)`,
  already what `add_position_to_tracks` computes for players. Never the bbox
  centre.
- **Validity rule:** a calibration applies only to the exact
  `(sequence, frame_range)` it was frozen for. Outside that range —
  including a camera cut, pan, or zoom, all of which are exactly "the frame
  is no longer inside the frozen range" — every metric query returns
  `None`. No global/whole-match calibration is claimed or attempted.

## 4. Segment selection (frozen before any calibration error was computed)

Recorded verbatim, with hash, at
`experiments/records/experiment_M2/calibration/SEGMENTS_FROZEN.json`
(sha256 `4709125254f15f92f3336eb6346f37852a2c3dfd8f41dffdd7ceffc094e53a39`).

| sequence | frames | cut-free (mad/hcorr) | camera drift | geometry |
|---|---|---|---|---|
| bayern_munich_3-1_chelsea_228 | 1-60 | yes (1-120 checked) | 5.3px / 1.0px over 60 frames — effectively locked off | centre circle + halfway line, both fully visible |
| women_1_239 | 141-170 | yes (111-171 checked) | 0.0px / 0.0px in the tested window | centre circle (partial arc) + touchline (direction only) |
| youth_premier_league_1133 | — excluded | n/a | n/a | camera fixed on a single pitch corner for the ENTIRE 300-frame clip (verified by sparse scan); no circle, no box, ever visible — only a touchline/goal-line junction and a 1 m corner arc too small to click reliably. Not forced. |

`women_1_239`'s first ~100 frames were rejected first: cumulative optical-flow
drift there reaches ~290px (the camera pans substantially following play),
disqualifying that portion despite visually resembling a stable shot in a
sparse frame-by-frame check. This is exactly the failure mode
`CameraMovementEstimator` was used to catch.

## 5. Correspondence protocol

Target of ≥8 with ≥6 fit/≥2 held-out was **not fully reached**, and this is
disclosed rather than hidden. What each segment's single available circle
+ reference-line combination actually supports, without assuming this
stadium's pitch length or width, is at most 5 exactly-derivable points
(circle centre + 4 cardinal points on two conic-conjugate diameters) — a
genuine hard limit of a one-circle, no-box, no-known-pitch-length view, not
a shortcut taken for convenience. Correspondence count actually used:

- **4 fit points** per segment: the two circle-conjugate-diameter pairs,
  non-collinear, well-distributed around the circle (not clustered).
- **1 held-out landmark** per segment: the circle's own centre, computed as
  the pole of the line at infinity w.r.t. the fitted conic — a construction
  that uses ONLY the conic, never the fit points, so for
  `bayern_munich_3-1_chelsea_228` (halfway line passes through the circle,
  `reference_line_mode: diameter`) it is a genuinely independent check. For
  `women_1_239` the halfway line itself was never visible in the stable
  window (only a touchline, which does not pass through the circle);
  `reference_line_mode: direction` was used instead, which builds the 4 fit
  points AS lines through this same centre point — there, the "held-out"
  landmark is **not independent** and is labelled as such
  (`held_out_landmark_independent: false` in the correspondences file,
  asserted by test). Reported error for it is therefore 0.0 by construction,
  not evidence.
- **Reconstruction check (the real, independent validation for both
  segments):** every other detected circle-boundary pixel (161 for bayern,
  260 for women, after removing the 4 used in fitting) reprojected to pitch
  space and checked against the known radius 9.15 m. This is not blind
  either — it is over the same fitted conic — but it is a much larger,
  spatially-spread set of genuinely-unused points and is the primary
  evidence behind the precision claim below.

Points were extracted algorithmically (colour + connected-component shape
filtering to separate thin line pixels from players' kit, then iterative
outlier-trimmed conic/line fitting — see `tools/m2_circle_correspondences.py`)
from two hand-specified rough regions per segment (a circle ROI, a
reference-line band). Manual work: 2 rough rectangles per segment, no
per-point clicking. `cv2.findHomography` (exact 4-point solve; 4 points is
what the geometry actually supports here, not a default left unexamined).

## 6. Geometric validation results

| | bayern (1-60) | women (141-170) |
|---|---|---|
| held-out centre landmark error | 1.666 m (**independent**) | 0.000 m (not independent, disclosed) |
| reconstruction: mean / median / p95 / max | 0.049 / 0.047 / 0.102 / 0.142 m | 0.123 / 0.124 / 0.243 / 0.293 m |
| reconstruction n points | 161 | 260 |
| homography condition number | 27497 | 22175 |
| degenerate? | no | no |

Notable: bayern's 1.67 m held-out error is concentrated entirely in one axis
(x-component of the error is 3.6e-8 m; the full 1.67 m is in y, the axis
perpendicular to the halfway line). The near-boundary reconstruction check on
the same segment is sub-15cm. This says the fitted homography is precise
near the fit points and near the circle boundary generally, but accumulates
more error over the ~9m extrapolation to the interior centre point — real,
useful information this validation step was specifically designed to surface
(section 7: reconstruction/landmark error, never player speed, as the
validation signal).

## 7. Error budget (position error → speed error, not an invented threshold)

Current production speed cadence
(`trackers/speed_distance.py:SpeedDistanceEstimator`, default
`frame_window=5`) at native 25 fps: `dt = 5/25 = 0.2 s`.

Displacement is computed from two independently-noisy endpoint positions, so
`displacement_error ≈ sqrt(2) * position_error`, and
`speed_error = sqrt(2) * position_error / dt`.

Using the **conservative, genuinely-independent** position-error estimate
(bayern's held-out landmark, 1.67 m — deliberately not the more optimistic
0.05-0.14 m near-boundary reconstruction number, since real player positions
are scattered across the pitch, not clustered on the circle boundary):

| window | dt | implied speed error |
|---|---|---|
| current default (5 frames) | 0.2 s | sqrt(2)*1.67/0.2 ≈ **11.8 m/s** — useless, larger than a sprint |
| 25 frames | 1.0 s | sqrt(2)*1.67/1.0 ≈ **2.4 m/s** — same order as real speed differences |
| 60 frames | 2.4 s | sqrt(2)*1.67/2.4 ≈ **1.0 m/s** |

**Chosen tolerance: minimum 25-frame (1.0 s) window before treating a speed
figure as anything more than illustrative**, because that is the smallest
window at which implied speed error drops below the current default's by an
order of magnitude and reaches the same scale as the quantity being
measured — not because 1.0 s is a round number. Even at that window the
~2.4 m/s implied error is not small enough to defensibly report a precise
speed value; see verdict below. This was derived from the measured
validation numbers and the existing cadence, not invented, and was not tuned
against any athlete speed value (section 8's explicit prohibition).

## 8. Real trajectory demonstration

`experiments/records/experiment_M2/calibration/trajectory_demo.json`. The
unchanged production chain (`FootballTracker`, `best_A_960.pt`) was run once
over `bayern_munich_3-1_chelsea_228` frames 1-60 (the calibrated segment).
Track id 1 (alive all 60 frames) was projected frame-by-frame through the
frozen calibration: 60 pitch-coordinate points, e.g. frame 1 →
`(-12.28, -1.86)` m, frame 60 → `(-11.77, -3.55)` m — a small, plausible
drift consistent with a player making minor positioning adjustments rather
than sprinting. Short-window demo over frames 1→26 (1.0 s at 25 fps):
displacement 2.15 m, speed 2.15 m/s — mechanically correct and a physically
plausible number, but see the error budget above for why its **precision**
is not yet asserted as validated.

## 9. Fail-closed behaviour (asserted by test, 25/25 passing)

- No calibration loaded → `None`.
- Frame outside a segment's `[start, end]` → `None`. Boundary frames
  (`start`, `end`) themselves are inclusive and DO return a value — tested
  explicitly.
- A camera cut/pan/zoom is modelled exactly as "the next frame falls outside
  the frozen segment boundary" — no separate camera-validity mechanism was
  built for M2, per the milestone's own allowance for manually frozen
  boundaries.
- Two different calibrated segments covering the same sequence are never
  silently chained across a short-window speed query — the segments must
  share one `artifact_sha256`, tested with two distinct segments 1 frame
  apart at the boundary.
- No `native_fps <= 0` or `None` is ever allowed to produce a result.
- Tampering with a frozen calibration artifact after the fact (any single
  homography coefficient) is caught by a SHA256 mismatch and raises
  `ValueError` rather than silently loading — tested for both segments.

## 10. P1.1 follow-up (documented only, not implemented)

Pitch calibration gives, for the first time, a real on-pitch/off-pitch
spatial signal: a person's ground point can now be checked against whether
it falls within plausible pitch bounds in a calibrated segment. This is a
plausible future input to P1.1's `NON_PARTICIPANT_OR_NONMATCH_BALL` failure
mechanism (100% of P1's false assignments traced to a stably-tracked
non-participant with zero GT overlap). This is recorded as a forward-looking
observation only — possession was not touched, re-tested, or re-scored in
M2, per explicit instruction.

## 11. M3 blocker note

P1.1 found that `youth_premier_league_1133`'s CBIoU track topology did not
reproduce identically between two runs of unchanged code (see
`experiment_P1_1_record.md` section "Determinism check"). Not investigated
here — M2's own segment selection excluded this sequence anyway for
unrelated geometry reasons (no circle/box ever visible), so the
nondeterminism did not block calibration work. Recorded as an explicit
reproducibility blocker to resolve before sealed TEST is touched.

## 12. Verdict

**B — M2 PARTIAL.** The homography is geometrically real and validated:
sub-15cm reconstruction error near the fit region on both segments, a
genuinely independent interior landmark check on one segment (1.67 m — not
negligible, but real and bounded, and diagnostic of where the residual error
lives), fail-closed behaviour proven outside calibration by test, and a real
production-tracker trajectory successfully projected into pitch metres with
a mechanically-correct short-window speed computation. What is **not** yet
supported is a *precise* speed/distance claim at the short-window scale this
milestone is scoped to: the conservative, independent position-error
estimate (1.67 m) implies speed uncertainty (~2.4 m/s at a 1.0 s window)
that is the same order of magnitude as the real speed differences a caller
would want to measure. Per verdict B's own definition: calibrated
**coordinates** are the validated M2 deliverable; speed/distance figures
from `short_window_displacement_and_speed` are mechanically functional and
demonstrated, but should be treated as illustrative, not precision-validated,
until correspondence precision improves.

## Explicit non-actions

Possession: not modified, not re-scored. Identity correspondence contract:
not touched. SN3D, BallTemporalSelector, CBIoU, detector thresholds: not
modified. No training. No new dataset. Sealed TEST: not accessed. M3: not
started. `youth_premier_league_1133` CBIoU nondeterminism: recorded, not
investigated.
