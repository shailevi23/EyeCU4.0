# Experiment P1 — close possession measurement — final disposition

```
STATUS      CLOSED
VERDICT     B — PlayerBallAssigner shows a clear, specific failure mechanism
M1 STATUS   CLOSED
```

## Scope

M1 only: fix the two defective evaluation contracts inherited from P0 (identity
correspondence, NO_CONTROL definition), complete the remaining 50 annotations
under the fixed contracts, score the UNCHANGED production chain once, report.
No parameter in PlayerBallAssigner, BallTemporalSelector, SN3D, CBIoU, or the
human detector was touched. No training. No TEST access. M2 not started.

## Frozen population

`data/possession_val_v1/POSSESSION_VAL_V1_FROZEN.json`
sha256 `9738780c1cad3311c9b137fad23ff7d1076fcf719591d342222045d6f47c5025`
(re-verified byte-identical to the value recorded at P0 time before any work
in this milestone began). 60 frames, 6 windows, unchanged.

## Contracts frozen before reading the remaining 50 labels

- `IDENTITY_CORRESPONDENCE_CONTRACT.md`
  sha256 `f8b003d4590a4f16e1d6021256dfb75b7cd8ae7ae439e32ba36f3ec493248ee9`
- `ANNOTATION_PROTOCOL_V2.md`
  sha256 `bf9d25e10546abf4ee09a2abe5be888ff7866b5336bcc3739dd990b20517c916`

Both documents live in this directory. Executable form of the identity
contract: `tools/possession_identity.py`. Renderer used for blind annotation:
`tools/render_possession_reads.py` (GT boxes/ids only, no model output).
Scoring harness: `tools/eval_possession_val_p1.py`.

## Why the contracts changed (see the documents for full detail)

1. P0 mapped a predicted player box to a GT identity per frame at
   `IoU >= 0.50`. A stable identity that dipped below that bar on a single
   frame lost its mapping for that frame and scored as a possession error even
   though possession never moved. Reproduced and fixed: see
   `TestP0FlickerPathology` in `tests/test_possession_identity_contract.py`.
2. P0's `NO_CONTROL` label for `women_1_239_w1:195` was justified in its own
   note by "nearest player >70px native" — the exact threshold
   `PlayerBallAssigner` is being evaluated against. Protocol v2 redefines
   `NO_CONTROL` physically (no algorithmic term permitted in the definition)
   and the frame was re-examined under it; the label did not change, but its
   justification is now physical (an isolated ball in open space, confirmed by
   locating the ball pixel-blob and measuring it against every GT box) rather
   than a restatement of the assigner's own rule.

Both changes were frozen and hashed **before** the remaining 50 labels were
read, per governance.

## Governance / auditability

- The original P0 result is preserved unmodified at
  `experiments/records/experiment_S1/p0/P0_POSSESSION_BASELINE.json` and is not altered by this
  record.
- `tools/eval_possession_val_p1.py` recomputes the OLD (P0) mapper on the same
  scored rows in the same run as the NEW mapper, so the two are always
  reported side by side rather than from two separate executions.
- Disclosed blinding exposure: the P1 operator read the P0 result file (which
  includes `assigned_gt_id`/`assigned_model_track` for all 60 frames under the
  now-discarded mapper) during orientation, before the 50 new labels were
  assigned. Mitigations and reasoning are recorded in
  `ANNOTATION_PROTOCOL_V2.md` under "Blinding exposure — disclosed". The 10
  original P0 labels were left untouched; only their *justification* was
  re-examined once (case 2 above), never their value, and never after seeing
  the P1 score.

## Annotation

Frozen at `data/possession_val_v1/POSSESSION_VAL_V1_ANNOTATIONS.json`,
sha256 `df7d142c72963703d1848adc0d00906125e60b5279495830e71aff9e850e36d1`.
60/60 labelled, 0 UNLABELLED. Distribution: PLAYER 35, NO_CONTROL 20, NO_BALL 0,
AMBIGUOUS 5. Every `PLAYER` row's `gt_player_track_id` was checked to exist in
GT on that exact frame (`tests/test_possession_identity_contract.py::
TestFrozenPopulation::test_player_labels_name_an_identity_that_exists_in_gt_on_that_frame`).

## Identity mapping validation (synthetic, before use on P1 data)

24/24 tests in `tests/test_possession_identity_contract.py` pass, covering:
stable track -> correct identity; two disjoint fragments of one player both
map to that player; near-50/50 correspondence -> `UNMAPPABLE` (`not_dominant`);
co-existing conflicting tracks -> the stronger claimant keeps the identity, the
other becomes `UNMAPPABLE`, and a tie makes both `UNMAPPABLE`; a demoted track
is never silently reassigned to its runner-up; the specific P0 flicker
scenario (bayern w1, frames 199-200) reproduced under the old mapper and shown
stable under the new one, including an adversarial single-frame-kill sweep.

## Production run

Unchanged chain: `SN3D_BASE -> BallTemporalSelector v1 -> CBIoU humans ->
PlayerBallAssigner(max_distance=70)`. Run once, full contiguous sequences from
frame 1 (same harness fix as P0: detector cache cleared per sequence, CBIoU
given full sequence context rather than window-local frames). Result:
`experiments/records/experiment_P1/P1_POSSESSION_RESULT.json`.

## Headline metrics

| | value |
|---|---|
| mappable PLAYER frames | 28 / 35 |
| exact-player correctness (mappable) | 13/28 = 46.43% |
| overall PLAYER coverage | 28/35 = 80.0% |
| UNMAPPABLE rate | 1/35 = 2.86% |
| false assignment on negatives (NO_CONTROL+NO_BALL) | 10/20 = 50.0% |

Full per-window, selector-state, error-category and P0-delta breakdowns are in
the JSON result file and in the assistant's final P1 report for this session.

## Dominant measured failure mechanism

Both the wrong-player errors (concentrated in `bayern_munich_3-1_chelsea_228_w0`,
8/10, and `youth_premier_league_1133_w1` frames 200-204) and the false
assignments on negatives (`youth_premier_league_1133_w0`, 10/10 of that window,
which is 100% of all false assignments measured) trace to the same root cause:
**when more than one ball-like object is present in a scene — a resting or
warm-up ball, a ball held by a person who is not a tracked game participant,
or a spurious detection elsewhere in frame — the production chain does not
distinguish the genuine in-play match ball from the other object, and
possession is assigned from proximity to whichever object the ball branch
happened to output that frame.** The recorded `ball_centre` in the result rows
is frequently 60-200px from the true controlling GT player's foot in these
frames, which is far outside plausible detector noise and consistent with the
branch having locked onto a different ball-shaped object.

## Verdict

**B — PlayerBallAssigner (as fed by the current production ball branch) shows
a clear, specific failure mechanism.** Not A: 46% exact-player correctness and
50% false-assignment on negatives are not a credible baseline result. Not C:
UNMAPPABLE and AMBIGUOUS rates are both low (2.86% and 8.3% of all frames
respectively) — the benchmark produces a clean signal, it is not too ambiguous
to support a claim. Not D: the evaluation contract itself performed exactly as
designed (the flicker case was fixed, UNMAPPABLE fired exactly where evidence
was genuinely weak, and the delta table shows precisely which P0 rows changed
and why).

## Recommended next action (NOT executed)

Make possession assignment provenance-aware to the ball branch's own
multi-candidate evidence: carry the full candidate list (already computed by
`tracker.ball_candidates` / `BallTemporalSelector`, not something new to
build) through to the assignment step, and require the selected candidate to
be plausible as the *same physical ball* over a short window (e.g. bounded
velocity/continuity against the immediately preceding accepted position)
before possession is assigned from it; when candidates disagree or the chosen
one is discontinuous with recent history, prefer UNKNOWN over forcing an
assignment. This is the smallest fix that targets the measured failure without
retraining or reopening detection/tracking. It is a recommendation for a future
milestone only; nothing has been executed under M1.

## Explicit non-actions

PlayerBallAssigner: not modified. `max_distance`: unchanged at 70.
BallTemporalSelector: not modified. SN3D: not modified. CBIoU / human
detector: not modified. No training. Sealed TEST: not accessed. M2: not
started. New possession labels: 50 (the frozen cap). No broad repo audit was
performed; no unrelated pre-existing test failures were investigated or
fixed.
