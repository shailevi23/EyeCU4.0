# M4 -- TEST detection annotation protocol (frozen before any visual TEST access)

This protocol is written and hashed **before** any of the 60 FINAL TEST
frames are opened/viewed by the annotator (the assistant). It is not
adjusted after viewing any frame. `human_annotation_accessed` is `false`
until after this file and its `.sha256` sidecar exist.

## Scope

Exactly the 60 frames listed in `FINAL_TEST_FRAME_LIST.json`
(sha256 `2ea02e1839726a6b4ad2fc2dbd979b58fa65f038e828d556eb0b290e8a47a62f`),
already extracted to `experiments/records/experiment_M4/candidates/<sequence>/`.
No other TEST frame is annotated.

## Classes (reused verbatim from `docs/guides/LABELING.md`, same ontology
already used for the frozen TRAIN/VAL annotations, class ids match the
existing project convention)

```
0 player
1 goalkeeper
2 referee
3 ball
```

## Ball ontology -- explicit deviation from the TRAIN/VAL convention

`docs/guides/LABELING.md` implicitly treats "the ball" as singular
("Ball only when visually identifiable... A missing ball is honest").
For TEST detection GT, the ontology is deliberately widened:

**`ALL_VISIBLE_PHYSICAL_FOOTBALLS`** -- every real, physical, football-shaped
object visible in the frame gets its own `ball` box, regardless of whether
it is the ball currently in active play. This includes, if present: a second
match ball near the touchline, a warm-up/practice ball, a ball boy's spare
ball, a ball a substitute is holding. This does **not** include: a ball
depicted only as a 2D graphic (advertising hoarding, broadcast graphic
overlay, a ball printed on clothing/kit). The test is physical presence in
the filmed scene, not visual likeness.

Rationale: a single-active-ball convention silently encodes a selection
judgement (which ball is "the" ball) into ground truth. M5 evaluates raw
detection (P/R/mAP), which must not be contaminated by that judgement --
the detector is not asked to guess intent, only to find every real ball.

## Other classes -- rules reused verbatim from `docs/guides/LABELING.md`

- **Occlusion**: label a partially occluded player whenever identifiable as
  a person; box only the visible extent; do not guess hidden-limb position.
  If it cannot be told to be a person, do not label it.
- **Goalkeeper stays goalkeeper** regardless of kit similarity to teammates.
- **Referee stays referee**, including assistant referees on the touchline
  and at long range; never assigned a team.
- **Not players**: coaches, substitutes, medical staff, ball boys,
  spectators -- left unlabelled.
- **Keep boxes tight**: enclose the visible extent only, no margin.
- **Empty / no-object frames are valid** and are recorded with an empty
  object list, not skipped.
- **Duplicate boxes** on one object are not permitted.

## Box coordinate convention

Pixel-space `[x1, y1, x2, y2]` (top-left / bottom-right corners, integer
pixel coordinates in the native resolution of the extracted candidate
frame), matching the convention already used elsewhere in this project's
tracking/annotation tooling (e.g. `FootballTracker` track records,
`data/tracking_val_gt` sequence GT) -- not the normalized YOLO `.txt`
format used for the TRAIN/VAL training set, since these 60 frames are a
one-off detection-metrics GT set, not a training split.

## Method (assistive, non-model)

The annotator (the assistant) visually inspects each full frame, and may
programmatically crop/zoom sub-regions (pure image-processing crop/resize,
no model inference) to resolve small or ambiguous objects (chiefly the
ball, and distant players/referees) before recording a box. Crops are a
magnification aid only; they do not change what counts as visible.

**No frozen production model is invoked at any point in this process.**
`best_A_960.pt`, the SN3D ball detector, `BallTemporalSelector`, CBIoU
tracking, and the full pipeline are not run on any TEST frame during
annotation. There are no draft/prelabel boxes of any kind -- every box is
placed by direct visual read.

## Per-frame record schema

```json
{
  "sequence": "<sequence name>",
  "frame_number_1based": <int>,
  "file": "<path>",
  "objects": [
    {"class": "player|goalkeeper|referee|ball", "bbox": [x1, y1, x2, y2]}
  ],
  "notes": "<optional annotator note, e.g. reason for a judgement call>"
}
```

## Failure handling

If a frame genuinely cannot be defensibly annotated (e.g. corrupt/unreadable
image, or a scene where visible-object judgement cannot be made responsibly
even with crops), the problem is recorded and annotation **stops** for a
milestone-level decision -- the frame is not silently dropped or swapped
after having been viewed, since that would reintroduce content-dependent
selection after visual access, which Section 4's frozen replacement rule
was specifically designed to avoid.

## Completeness requirement

Every player, goalkeeper, referee, and every visible physical football in
the frame must receive a box. This is exhaustive per-frame annotation, not
a sample of salient objects.
