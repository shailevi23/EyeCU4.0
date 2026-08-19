# EYECU SEQUENCE-LEVEL IDENTITY CORRESPONDENCE CONTRACT — v1

```
STATUS      FROZEN
SCOPE       possession scoring only
APPLIES TO  POSSESSION_VAL_V1 (frozen population sha256
            9738780c1cad3311c9b137fad23ff7d1076fcf719591d342222045d6f47c5025)
```

This contract is **not** HOTA. It is **not** IDF1. It does not measure tracker
identity continuity, and it must never be reported as a tracking metric. It
answers exactly one question:

> When PlayerBallAssigner hands possession to predicted track `T` on frame `f`,
> **which physical player** — as named by the VERIFIED MOT identity GT — did it
> hand possession to?

## Why the P0 contract had to be replaced

P0 mapped a predicted player box to a GT identity **per frame**, by
`IoU >= 0.50` against that frame's GT boxes only. That is a single-frame
decision with no memory, and it fails in a way that corrupts possession
scoring rather than tracking scoring:

- A predicted box that sits on the same physical player for twenty consecutive
  frames can dip below IoU 0.50 on one frame — a limb leaving the box, a
  partial occlusion, a box-convention disagreement between the detector's
  visible-extent box and the GT's full-body box. Under P0 the mapping
  *disappears* on that frame (`assigned_gt_id: null`) and the frame scores as
  a possession error, although possession never moved and the assigner never
  changed its mind.
- The observed instance in the P0 result: sequence
  `bayern_munich_3-1_chelsea_228`, window `w1`. Predicted track `6` maps to GT
  identity `7` on frames 195, 196, 197, 198, 201, 203, 204 — and to `null` on
  frames **199 and 200**. Frame 200 is a labelled `PLAYER` frame with
  `gt_player_track_id = 7`. It was scored as a failure by a mapping flicker,
  not by a possession failure.
- Symmetrically, P0 had no way to say *"I cannot tell"*. A missing mapping was
  silently indistinguishable from a wrong player. Those are different facts
  about the assigner and must be counted separately.

This contract therefore accumulates evidence across the frames a predicted
fragment actually exists on, and makes **UNMAPPABLE** an explicit, first-class
outcome that is *not* counted as wrong-player.

## Declared constants

These were fixed **before** any of the remaining 50 labels were read and before
this contract was run against possession correctness even once.

```
IOU_VOTE            0.30    per-frame vote threshold (a vote, not a decision)
MIN_SUPPORT_FRAMES  2       absolute minimum frames of evidence per fragment
MIN_SUPPORT_RATE    0.50    fraction of a fragment's evaluable frames that must
                            overlap SOME GT identity
DOMINANCE_RATIO     2.0     top votes >= 2.0 x runner-up votes
DOMINANCE_MARGIN    2       top votes - runner-up votes >= 2
```

Justification, none of it derived from possession outcome:

- `IOU_VOTE = 0.30`. A per-frame overlap here is only one vote among many, so
  it can be set lower than a threshold that would decide a match on its own.
  0.30 tolerates the exact perturbations that made 0.50 flicker (occlusion,
  visible-extent vs full-body box convention) while still being far above the
  incidental overlap between two *different* players, who in these sequences
  are separated by more than a box width in the overwhelming majority of
  frames. Sensitivity across `IOU_VOTE` in `{0.20, 0.30, 0.40, 0.50}` is
  reported as audit evidence in the P1 record; the declared value is 0.30
  regardless of what that sweep shows.
- `MIN_SUPPORT_FRAMES = 2`. One frame of evidence is precisely the defect being
  removed. A fragment that exists on only one evaluable frame is **UNMAPPABLE**
  by construction.
- `MIN_SUPPORT_RATE = 0.50`. A fragment that overlaps no GT identity on most of
  its own frames is not describing a GT-tracked person; forcing it onto one
  would be inventing a correspondence.
- `DOMINANCE_RATIO` / `DOMINANCE_MARGIN`. Both must hold. The ratio rejects a
  near-tie between two identities; the absolute margin stops a 2-vote-to-1-vote
  fragment from passing on ratio alone.

## The algorithm

Inputs, per sequence, over the **full contiguous run range** used for scoring:

- `P[t] = {frame -> bbox}` — every predicted player track emitted by the
  production human tracker.
- `G[g] = {frame -> bbox}` — VERIFIED MOT identity GT.

### Step 1 — per-frame votes

For predicted track `t`, its *evaluable frames* `F(t)` are the frames where `t`
exists **and** GT has at least one box. For each frame `f` in `F(t)`:

- compute IoU between `P[t][f]` and every GT box on `f`;
- let `g*` be the argmax IoU (ties broken by lowest GT id, deterministic);
- if `IoU(g*) >= IOU_VOTE`, record one vote `votes(t, g*) += 1`; otherwise no
  vote is recorded for that frame.

`support(t)` is the sum of `votes(t, g)` over all `g` — the number of `t`'s own
frames on which it overlapped any GT identity at all.

### Step 2 — provisional decision per fragment

Let `top` be the identity with the most votes and `runner` the second
(0 votes if none). Track `t` is **UNMAPPABLE** unless all of:

```
|F(t)|      >= MIN_SUPPORT_FRAMES
support(t)  >= MIN_SUPPORT_FRAMES
support(t)  >= MIN_SUPPORT_RATE * |F(t)|
votes(top)  >= DOMINANCE_RATIO * votes(runner)
votes(top) - votes(runner) >= DOMINANCE_MARGIN
```

Otherwise `t -> top`, with `purity(t) = votes(top) / support(t)` recorded.

### Step 3 — simultaneous conflict resolution

Two predicted tracks **conflict** if they were provisionally mapped to the same
GT identity `g` **and** they exist on at least one common frame. Two fragments
of the same physical player that never co-exist do **not** conflict and both
keep `g` — this is the fragment tolerance the contract requires.

For each conflicted group on `g`:

- the member with the strictly highest `votes(., g)` keeps `g`;
- every other member becomes **UNMAPPABLE**;
- if the highest `votes(., g)` is tied, **every** member of the group becomes
  **UNMAPPABLE** — no silent, no forced identity.

Demoted tracks are **not** reassigned to their runner-up. Losing a contest is
not evidence for a different identity.

### Step 4 — output

A frozen map `predicted_track_id -> (gt_id | UNMAPPABLE)` per sequence, plus,
per track: `|F(t)|`, `support`, full vote vector, `purity`, and the decision
reason (`mapped`, `no_support`, `low_support_rate`, `not_dominant`,
`too_short`, `conflict_lost`, `conflict_tied`).

## Properties this contract deliberately has

- **Fragment-tolerant.** Evidence is accumulated over exactly the frames a
  fragment exists on. A short-lived fragment is judged on its own frames, not
  penalised for not living long enough to meet some global quota.
- **Many-to-one is legal.** N temporally disjoint fragments of one physical
  player all map to that player.
- **One-to-one is enforced where it matters.** Co-existing fragments cannot
  both claim one identity.
- **Refusal is a result.** `UNMAPPABLE` is reported separately and is **never**
  counted as `wrong player`.
- **Stable under single-frame perturbation.** Flipping any one frame's IoU from
  above to below `IOU_VOTE` changes `votes(top)` by at most 1 and cannot, on
  its own, move a fragment across the dominance test unless it was already at
  the margin.

## Known limitation, declared

Evidence is accumulated over a fragment's **whole** lifetime. A predicted track
that genuinely follows player A for most of its life and then switches to
player B is mapped wholly to A, and B-frames scored under it would be
attributed to A. The dominance test only converts this to `UNMAPPABLE` when the
switch is close to balanced. Mitigation: per-track `purity` is reported for
every track used in scoring, so the size of this exposure is visible rather
than assumed. This limitation is inherent to the fragment-level correspondence
the milestone specifies and is not repaired here.

## Prohibited uses

- These constants must not be changed on the basis of possession correctness.
- This contract must not be used, cited, or compared as HOTA, IDF1, MOTA, or
  any tracking-accuracy figure.
