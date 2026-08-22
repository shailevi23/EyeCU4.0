# Experiment P1.1 — possession failure attribution — final disposition

```
STATUS               CLOSED
GATE                 B — errors materially split, no single small fix justified
SINGLE FIX IMPLEMENTED   NO
VERDICT               C — NO SINGLE FIX JUSTIFIED
```

P1 (verdict B, 46.43% exact-player correctness among mappable) is CLOSED and
unchanged. P1.1 attributes every P1 error to a root mechanism using only
existing frozen P1 rows, the frozen identity-correspondence table, and one
read-only diagnostic re-run of the unchanged production chain. No M1 contract
was reopened. No annotation was added or changed. No parameter of the
detector, SN3D, CBIoU, or PlayerBallAssigner was modified.

## 1. Provenance erratum (metadata only)

See `P1_ERRATUM_provenance.md`. `annotation_sha256` in
`P1_POSSESSION_RESULT.json` was `null` (a report-schema field the harness
never populated) and is now set to the correct, unchanged annotation hash
`df7d142c72963703d1848adc0d00906125e60b5279495830e71aff9e850e36d1`. No row, no
count, no metric was recomputed.

## 2. Diagnostic method

`tools/p1_1_attribution_diagnostics.py` re-runs the identical production chain
once per sequence (bayern, women, youth) and, for the frames named in the
P1.1 brief, dumps (a) the raw ball-candidate list `BallTemporalSelector`
chose from and (b) every predicted player track alive that frame with its
frozen P1 identity mapping and four foot-distance metrics to the selected
ball. Written to `P1_1_ATTRIBUTION_DIAGNOSTICS.json`. Nothing in the
production chain was modified; this only reads and records outputs the P1
harness computed but did not persist.

**Determinism check, performed before trusting any of this data**: the ball
branch (SN3D + `BallTemporalSelector`) reproduced bit-identical
`ball_state`/`ball_centre` for all 60 rows across both runs — fully
trustworthy. Player-track topology reproduced identically for
`bayern_munich_3-1_chelsea_228` and `women_1_239` (the exact track id assigned
in the original P1 run was present, at the same bbox, in the rerun every
time). For `youth_premier_league_1133`, it did **not**: e.g. the original run's
track 64 at frame 202 (support 10/41, voted GT 7) does not exist in the
rerun; a different track (57) occupies the same bbox at that one frame but
carries a completely different lifetime vote history (107/108 frames voting
GT 8). This means CBIoU produced different association topology for this one
sequence between two runs of otherwise-identical code — a reproducibility
finding, not something this task modifies or root-causes (CBIoU is frozen and
out of scope). Consequence: for `youth_premier_league_1133`, only the
deterministic ball-candidate evidence from the rerun is used; all
player-identity/tracking-availability claims for that sequence rely on the
**original, frozen** P1 rows and correspondence table only, never on the
rerun's player list.

## 3. Attribution counts

| category | count |
|---|---|
| BALL_OBJECT_SELECTION | 13 |
| PLAYER_ASSOCIATION_GEOMETRY | 0 |
| NON_PARTICIPANT_OR_NONMATCH_BALL | 10 |
| BALL_UNAVAILABLE | 3 |
| HUMAN_TRACKING_OR_MAPPING | 6 |
| UNCERTAIN | 0 |
| **total** | **32** |

(15 WRONG_PLAYER + 10 SHOULD_BE_UNKNOWN + 6 UNASSIGNED_PLAYER + 1 UNMAPPABLE = 32.)

By window:

| window | errors | attribution |
|---|---|---|
| bayern_munich_3-1_chelsea_228_w0 | 10 | 10 BALL_OBJECT_SELECTION |
| bayern_munich_3-1_chelsea_228_w1 | 1 | 1 HUMAN_TRACKING_OR_MAPPING |
| women_1_239_w0 | 5 | 4 HUMAN_TRACKING_OR_MAPPING, 1 BALL_UNAVAILABLE |
| women_1_239_w1 | 0 | — |
| youth_premier_league_1133_w0 | 10 | 10 NON_PARTICIPANT_OR_NONMATCH_BALL |
| youth_premier_league_1133_w1 | 6 | 3 BALL_OBJECT_SELECTION, 1 HUMAN_TRACKING_OR_MAPPING, 2 BALL_UNAVAILABLE |

## 4. Key findings that overturned the P1.1 brief's own working hypotheses

**Pattern B (bayern_w1:202, women_w0:77/78/80/81) is NOT a geometry-rule
defect.** The brief hypothesized `PlayerBallAssigner`'s two-bottom-corner rule
was picking the wrong of several *present* candidate players. Player-by-player
geometry dumps show the true controller's track was **not present in the
frame's player list at all** in every one of these 5 cases:

- `bayern_w1:202`: GT 7's track (id 6) is absent from the frame's player list.
  Its own lifetime record shows `n_frames: 203` of 204 possible frames in the
  sequence — it is missing from exactly one frame, and this is that frame.
  With track 6 gone, the nearest present track (23, mapped to GT 18) is 51.2px
  away and gets picked; every distance metric tried (bottom-left,
  bottom-right, bottom-centre, nearest-bottom-edge) gives track 23 the same
  answer, because there was nothing better to compare it against.
- `women_w0:77/78/80/81`: GT 10's track (id 36) is a fragmentary CBIoU track
  (`n_frames: 40` of 204) that is present at frames 79/82/83/84 (where P1
  scored it correct) and absent at 77/78/80/81 (where P1 scored it wrong,
  falling back to track 35 / GT 8, ~26-30px away). This alternation is
  independently confirmed by the original P1 rows themselves (which show
  `assigned_model_track` switching between 35 and 36 across the window), not
  only by the rerun.

No frame in the WRONG_PLAYER set showed the true controller present alongside
a closer wrong candidate under any of the four distance metrics computed.
`PLAYER_ASSOCIATION_GEOMETRY` is therefore empty (0) and there is no
geometry-rule correction to make.

**Pattern A (bayern_w0) is a compound of missed detection and a correctly
functioning (but mis-anchored) motion gate, not a simple "selector chose
wrong" case.** Raw candidates at frames 75/76/79/80/84 contain exactly **one**
object — the wrong one, 172-209px from GT 8's true foot — and nothing near the
truth at any confidence. At frames 77, 78 and 81 a second, low-confidence
candidate (0.105-0.112, near `CANDIDATE_CONF`) does appear only ~5px from GT
8's true foot — and `BallTemporalSelector` rejects it, keeping the established
wrong trajectory instead. This is not a bug in the selector: its job is to
reject detections outside the motion gate built from its own history, and by
frame 77 that history is already anchored on the wrong object (there was no
alternative at frame 75, the window's first frame). A possession-layer
continuity/provenance check applied after the fact would face the identical
problem — it has no way to know the established anchor is the wrong one
either, since both "keep tracking the established object" and "the true ball
just entered view" look the same from a pure motion-continuity standpoint.

**youth_w1** shows the same missed-then-compounded pattern (frames 196, 200,
201: one wrong-location candidate or zero candidates, no alternative near
GT 1/GT 3's true position at any point), plus one `HUMAN_TRACKING_OR_MAPPING`
case at frame 202 where the assigned track (support 10/41, purity computed
over a mostly-non-overlapping lifetime) is a low-quality CBIoU fragment
independent of the ball's location.

## 5. youth_w0 (10/10 NO_CONTROL false assignments) — mandatory finding

The assigned track (55) has `support: 0` across its **entire** 59-frame
lifetime in the frozen P1 correspondence table — it never overlaps *any* GT
box at `IOU_VOTE=0.30` at any point it exists. It is a real, stable CBIoU
track (a real detected person), but not a labelled match participant. This
matches the earlier P1 annotation observation (already on record from the M1
session, contact sheet `youth_premier_league_1133_w0_sheet0.jpg`) of an
untracked person holding a ball near the touchline banner throughout this
window; no new visual read was needed to confirm it, since the computational
evidence (zero GT overlap across the track's whole life) is already
dispositive.

Raw candidate evidence (deterministic, from the rerun) shows **two**
ball-like objects present across most of frames 75-84: the selected one
(near the touchline person, y≈142-145, confidence 0.30-0.43) and a second one
higher in frame (y≈99-101, confidence 0.10-0.33) that **also moves smoothly
and continuously** frame to frame — at frame 82 the selector even switches
which of the two it reports. Both trajectories are individually plausible
motion. This directly confirms the brief's warning: **a short-window
trajectory-continuity gate would not fix this case**, because the object
actually selected is not erratic — it is a real, smoothly-moving ball, just
one that is not being handled by a tracked match participant. The only signal
that would disqualify it is participant eligibility (is the nearby person a
labelled match participant), and that signal does not exist anywhere in the
production pipeline at inference time — GT identity is unavailable at
inference by construction, and no substitute signal (jersey/team
classification, pitch-boundary ROI) exists without new calibration, new
annotation, or retraining, all explicitly out of scope for this task.

## 6. Answers to the brief's three questions

- **Fraction of the 15 WRONG_PLAYER errors truly upstream ball-object
  selection**: 10/15 (66.7%) — bayern_w0 (8) + youth_w1:196,200 (2).
- **Fraction with a selected ball already geometrically consistent with the
  correct player**: 5/15 (33.3%) — bayern_w1:202 and women_w0:77/78/80/81.
  **All five** are `HUMAN_TRACKING_OR_MAPPING` (the correct player's predicted
  track was absent from the frame), not a geometry-rule defect.
- **Cause of the 10/20 NO_CONTROL false assignments**: 100%
  `NON_PARTICIPANT_OR_NONMATCH_BALL` — a real, stably-tracked CBIoU box for a
  non-participant person near a genuinely smooth, continuous secondary ball,
  with no eligibility signal in the pipeline to exclude it.

## 7. Decision gate

**Gate B.** Errors split across three structurally distinct, load-bearing
mechanisms, none of which a single small possession-only change can
defensibly fix under the hard constraints of this task:

- `BALL_OBJECT_SELECTION` (13/32, 41%) traces to the detector never emitting a
  candidate near the true ball in the majority of the affected frames, and to
  `BallTemporalSelector`'s continuity gate correctly rejecting the rare
  low-confidence true-ball candidates that do appear, because they fall
  outside a motion gate anchored on an already-wrong trajectory it has no way
  to distinguish from a legitimately established one. A possession-layer
  provenance/continuity re-check would face the same indistinguishability and
  is not expected to help; this is a detector/selector-level defect, both
  frozen and out of scope.
- `NON_PARTICIPANT_OR_NONMATCH_BALL` (10/32, 31%, **all** of the false
  assignments) requires a participant-eligibility signal that does not exist
  in the production pipeline at inference and cannot be constructed without
  retraining, new annotation, or calibration — all explicitly prohibited.
  A trajectory gate specifically will not help, since the offending
  detections are themselves smooth and continuous.
- `HUMAN_TRACKING_OR_MAPPING` (6/32, 19%) requires the missing player's track
  to exist in the frame at all — a CBIoU-level fix, explicitly out of scope
  ("Do NOT... modify CBIoU").

No possession-only change reaches more than one of these three buckets, and
the largest bucket (`BALL_OBJECT_SELECTION`) was shown not to be reachable by
the specific class of fix (a possession-layer continuity/provenance gate)
that was pre-authorized as a candidate remediation in the P1 record. Per
instruction, **nothing was implemented**.

## 8. Verdict

**C — NO SINGLE FIX JUSTIFIED.** Possession remains a declared measured
limitation (46.43% exact-player correctness among mappable frames, driven by
three separate, mostly out-of-scope upstream mechanisms) rather than a
possession-layer defect this milestone can close. Recommend proceeding to M2
with this limitation on record, rather than attempting a fix whose evidence
base does not support it.

## Explicit non-actions

PlayerBallAssigner: not modified. `max_distance`: unchanged at 70.
BallTemporalSelector: not modified. SN3D: not modified. CBIoU / human
detector: not modified. Identity correspondence contract: not reopened.
Possession annotations: not changed. No training. No new dataset. Sealed
TEST: not accessed. Visual reads used: 0 new reads (relied on the M1 session's
already-recorded contact-sheet observation for youth_w0, within the ≤4 budget
by construction). M2: not started.
