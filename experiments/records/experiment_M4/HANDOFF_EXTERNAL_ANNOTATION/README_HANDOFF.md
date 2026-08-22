# M4 external annotation handoff -- 40 remaining sealed-TEST frames

Manual visual annotation inside this session was stopped for cost reasons
after 20 of 60 frames (all of `como_2-0_sassuolo`). This folder is a
self-contained handoff so the remaining **40 frames** (`manchester_city_v_liverpool`,
20 frames; `youth_2`, 20 frames) can be annotated externally in CVAT or
Roboflow, then returned and merged.

Nothing about the frozen population changed: this handoff draws from the
already-frozen `FINAL_TEST_FRAME_LIST.json`
(sha256 `2ea02e1839726a6b4ad2fc2dbd979b58fa65f038e828d556eb0b290e8a47a62f`)
and the already-frozen `TEST_ANNOTATION_PROTOCOL.md`
(sha256 `2dfaed2eb0057f9c4a0e5ae94baf66a80e5334853ef4970f2ba23ed9a94ebd6d`).
No frame was added, dropped, or replaced to produce this handoff.

## What's in this folder

- `images/manchester_city_v_liverpool/*.jpg` (20 files)
- `images/youth_2/*.jpg` (20 files)
- `obj.names` -- the 4 classes, in the exact order used throughout this
  project (`0 player, 1 goalkeeper, 2 referee, 3 ball`), matching
  `docs/guides/LABELING.md`'s existing TRAIN/VAL convention.
- `HANDOFF_MANIFEST.json` -- unambiguous identity mapping for all 40 files:
  `sequence`, `frame_number_1based`, `frame_index_0based`, and the exact
  filename each frame must keep.
- This README.

Filenames already encode identity unambiguously:
`<sequence>_<frame_number_1based, 6-digit>.jpg`, e.g.
`manchester_city_v_liverpool_000038.jpg`. Do not rename files.

## Ontology (frozen, reused verbatim from `TEST_ANNOTATION_PROTOCOL.md`)

```
0 player
1 goalkeeper
2 referee
3 ball
```

**Ball = `ALL_VISIBLE_PHYSICAL_FOOTBALLS`.** Label every real, physical,
football-shaped object visible in the frame as `ball` -- not just the one
in active play. This includes a second match ball, a warm-up ball, a spare
ball a substitute is holding, etc. Do **not** label a ball depicted only as
a 2D graphic (advertising hoarding, broadcast overlay, a ball printed on
clothing). The test is physical presence in the filmed scene.

## Other rules (reused verbatim from the frozen protocol)

- **Occlusion**: label a partially occluded player whenever identifiable as
  a person; box only the visible extent; do not guess hidden-limb position.
- **Goalkeeper stays goalkeeper** regardless of kit similarity to teammates.
- **Referee stays referee**, including assistant referees on the touchline
  and at long range; never assigned a team.
- **Not players**: coaches, substitutes, medical staff, ball boys,
  spectators -- leave unlabelled.
- **Keep boxes tight**: enclose the visible extent only, no margin.
- **Empty frames are valid** -- if a frame genuinely has no player/GK/ref/ball,
  submit it with zero boxes, don't drop it.
- **No duplicate boxes** on one object.
- **Exhaustive, not sampled**: every player, goalkeeper, referee, and every
  visible physical football in the frame must get a box.

## Drafting aid (optional, explicitly scoped)

A **non-EyeCU, generic** pretrained model (e.g. a stock COCO/person detector
built into CVAT or Roboflow) may be used only as a drafting aid to speed up
box placement. This is different from what was forbidden during the
in-session portion of M4 (no EyeCU production model -- `best_A_960`, SN3D,
`BallTemporalSelector`, CBIoU, or the full pipeline -- may ever be run on
TEST). If a generic drafter is used:
- Every frame still requires full manual completeness review afterward.
- Every visible football specifically must be manually checked -- generic
  detectors are unreliable on small/motion-blurred balls and are not
  trained on the `ALL_VISIBLE_PHYSICAL_FOOTBALLS` ontology (they'll miss
  secondary/non-match balls entirely).
- Class corrections (e.g. drafter mislabels a goalkeeper as a player) must
  be fixed by hand, same as the existing TRAIN/VAL workflow already
  documents in `docs/guides/LABELING.md` ("Goalkeeper stays goalkeeper").

## CVAT import

1. Create a new task, upload all 40 images from `images/` (both subfolders --
   CVAT tasks are flat, so the sequence name is already embedded in each
   filename for identity, no need to preserve the folder split).
2. Task labels: create exactly 4 labels named `player`, `goalkeeper`,
   `referee`, `ball` (rectangle/bbox type), in this order.
3. Annotate per the rules above.
4. Export: **Export as "YOLO 1.1"** format. This produces one `.txt` per
   image (normalized `cx cy w h`) plus an `obj.names`/`obj.data` pair --
   use the `obj.names` in this folder (or verify the exported one matches
   it exactly in order) so class indices line up.

## Roboflow import

1. Create a new project (Object Detection), upload all 40 images from
   `images/`.
2. Create the same 4 classes in the same order.
3. Annotate per the rules above.
4. Export in **YOLOv8** (or any YOLO-txt variant) format, or COCO JSON --
   either is fine as long as image filenames are preserved unchanged so
   `HANDOFF_MANIFEST.json` can map every returned box back to
   `(sequence, frame_number_1based)`.

## What must come back

Return, as a single zip or folder:
- The per-image label files (YOLO `.txt` per image, or one COCO/CVAT JSON --
  whichever the tool produced), covering all 40 images, filenames unchanged.
- The `obj.names`/class list actually used, so class-index order can be
  verified against this folder's `obj.names` before merging.

On return, these will be converted back into the same pixel-bbox JSON record
schema already used for the 20 frames done in-session
(`{sequence, frame_number_1based, file, objects: [{class, bbox: [x1,y1,x2,y2]}]}`)
and merged into the single annotation set before Sections 8-14 of M4 (QC,
stats, freeze, M5 contract prep, final report) resume.

## Current M4 access-state (unchanged by this handoff)

- `machine_leakage_accessed = true`
- `human_annotation_accessed = true` (20/60 frames annotated in-session;
  frames covered by this handoff have NOT been visually inspected by the
  assistant)
- `labels_frozen = false` -- annotation is still open, not yet complete
- `production_predictions_run = false`
- `evaluation_results_viewed = false`
- M4 Sections 8 onward (QC, stats, final freeze, M5 contract prep, final
  report) are paused pending the return of these 40 externally-annotated
  frames.
