# POSSESSION_VAL_V1 ANNOTATION PROTOCOL — v2 (physical)

```
STATUS      FROZEN
SUPERSEDES  the P0 annotation protocol
APPLIES TO  POSSESSION_VAL_V1, frozen population sha256
            9738780c1cad3311c9b137fad23ff7d1076fcf719591d342222045d6f47c5025
FROZEN      before any of the remaining 50 labels was read
```

## Why v1 had to be replaced

The P0 protocol defined `NO_CONTROL` partly in algorithmic terms. The evidence
is in the P0 label file itself — `women_1_239_w1:195` carries the note:

> "ball in open space, **nearest player >70px native**"

70 px is `PlayerBallAssigner.max_distance`. A ground truth that is defined by
the same threshold the system under test uses is not a ground truth; it is a
restatement of the system's own rule, and any agreement it produces is
circular. The system could not fail such a frame except by a detector error,
and it could not be *credited* for succeeding on one either.

v2 removes every algorithmic term from the label definitions. Labels now
describe what is physically happening on the pitch, and nothing else.

## The four states

The population, the window list and the frame list are **unchanged and frozen**.
Only label definitions change, and only unlabelled frames gain labels.

### `PLAYER`

A ball is visible or unambiguously locatable in the frame, and **one specific
player has clear control of it**. Clear control means at least one of:

- the ball is in contact with, or immediately at, that player's feet or body,
  and that player is the one acting on it (dribbling, shielding, striking,
  trapping);
- the ball is held, caught, or being released by that player (goalkeeper,
  throw-in taker);
- the ball is in flight or rolling directly to that player, that player is
  visibly receiving it, and no other player is contesting the reception.

`gt_player_track_id` MUST be recorded: the GT identity of that physical player,
read from the GT boxes and ids drawn on the annotation image.

If control is clear but the controlling person **has no GT identity** (they are
not a tracked target — a coach, a substitute, a warm-up player, a person off
the field of play), the frame is **not** `PLAYER`. See `NO_CONTROL`.

### `NO_CONTROL`

**A ball is visible or unambiguously locatable in the frame, but no player has
clear control of it.**

This is a statement about the physical state of play. Typical instances:

- the ball is loose, rolling, or stationary in open space;
- the ball is in flight between players, and no receiver is yet receiving it;
- the ball has just left a player and no one has yet taken it;
- the ball is under the control of a person who is not a tracked target
  (a warm-up ball with a coach or substitute), so no *player* controls it;
- two or more players are near the ball but none of them has it — a genuine
  50/50 where nobody is in control yet. (If instead the annotator cannot tell
  *which of them* has it, that is `AMBIGUOUS`, not `NO_CONTROL`.)

`gt_player_track_id` is `null`. Expected system output: **UNKNOWN / unassigned**.

The definition MUST NOT and DOES NOT mention: pixel distances, 70 px, any
distance-to-player threshold, PlayerBallAssigner behaviour, detector output,
tracker output, or any model prediction. An annotator applying this definition
never measures a distance; they judge whether a player is in control.

### `NO_BALL`

No ball is present in, or locatable within, the visible frame at all — it is
out of shot, out of play and off camera, or fully occluded with no visible
evidence of where it is.

`gt_player_track_id` is `null`. Expected system output: **UNKNOWN / unassigned**.

### `AMBIGUOUS`

The visual evidence is not sufficient to decide between the states above, or
the ball is present and contested and the annotator cannot say which player
controls it, or there is more than one ball-like object and the match ball
cannot be identified.

`gt_player_track_id` is `null`. **Excluded from all primary correctness
metrics**; reported as a count only.

`AMBIGUOUS` is the required answer whenever the annotator would otherwise be
guessing. It is not a failure of the benchmark; a fabricated label is.

## What the annotator may and may not see

MAY see:

- the frame image;
- GT human boxes and GT track ids drawn on it;
- the frame index;
- neighbouring frames of the same frozen window, for motion context.

MUST NOT see, or use:

- PlayerBallAssigner output for the frame;
- any model possession prediction;
- ball detector or BallTemporalSelector output;
- predicted player boxes or predicted track ids;
- any scoring result.

### Blinding exposure — disclosed

The P1 operator read `experiments/records/experiment_S1/p0/P0_POSSESSION_BASELINE.json` during
orientation, **before** these 50 labels were assigned. That file lists, for all
60 frames including the then-unlabelled 50, the fields `assigned_model_track`
and `assigned_gt_id` under the **old, defective** mapping contract, and
`ball_state`.

This is a real and irreversible weakening of the blind. It is recorded here
rather than glossed. Mitigations actually applied:

- every label below was assigned from the rendered frame image carrying GT
  boxes and GT ids only, with the P0 result file closed and not re-consulted
  during annotation;
- the P0 file's `assigned_gt_id` values are products of the mapper this
  milestone exists to discard, so they are not a reliable oracle to copy even
  if one wished to;
- the annotation was frozen and hashed before the production chain was re-run,
  so labels could not be adjusted to fit the P1 score.

Reader should treat agreement between P1 labels and P0 model output with
proportionate caution on the 50 new frames.

## Manual work cap

Exactly the 50 already-frozen unlabelled frames. No frame is added, replaced,
dropped, or re-selected. The 10 P0 labels are re-examined only where the v2
definitions change their meaning; any such change is recorded explicitly in the
P1 record with its reason.

## Freeze rule

Labels are frozen and hashed before the production chain is run. Labels are
**never** revised after a score is seen.
