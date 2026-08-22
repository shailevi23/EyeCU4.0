# M4 checkpoint -- manual visual annotation STOPPED by explicit user instruction, handoff prepared

Stopped for cost reasons (per-frame visual annotation tool usage). This
supersedes the prior checkpoint's "resume in-session" plan: **do not resume
manual visual annotation in-session** unless the user explicitly says so
again. Nothing frozen/hashed has been touched or invalidated.

## Done and frozen (Sections 0-5) -- unchanged

- Freeze precheck: manifest/source-tree/weight SHAs verified MATCH before any TEST access.
- `INITIAL_TEST_FRAME_LIST.json` (+ `.sha256`) -- 60 candidate frames, deterministic selection.
- `candidates/` -- all 60 candidate frames extracted, `candidates_manifest.json`.
- `LEAKAGE_SCREEN_RESULT.json` (+ `.sha256`) -- 0/60 leaks found against 2231-image TRAIN+VAL pool.
- `FINAL_TEST_FRAME_LIST.json` (+ `.sha256`) -- identical to initial list (0 replacements needed).
- `TEST_ANNOTATION_PROTOCOL.md` (+ `.sha256`) -- frozen before any visual TEST access.
- `machine_leakage_accessed = true`, `human_annotation_accessed = true`.

## Manual annotation: 20 of 60 done in-session, then STOPPED

File: `experiments/records/experiment_M4/ANNOTATIONS_DRAFT.json` (draft,
mutable, not hashed) -- **all 20 frames of `como_2-0_sassuolo` are complete**
(frame_number_1based = 38, 113, 188, 263, 338, 413, 488, 563, 638, 713, 789,
864, 939, 1014, 1089, 1164, 1239, 1314, 1389, 1464). Verified valid JSON,
20 frame records, all from one sequence -- que sequence is fully done.

The user's stop instruction referenced "the 3 annotations already completed,"
which undercounts the actual state (20, not 3) -- flagged and corrected in
the response at the time, not silently gone along with. The handoff below
covers the true remaining count (40 frames), not 57.

## Handoff package prepared for the remaining 40 frames

`experiments/records/experiment_M4/HANDOFF_EXTERNAL_ANNOTATION/`:
- `images/manchester_city_v_liverpool/*.jpg` (20) and `images/youth_2/*.jpg` (20)
  -- copied via `cp` from `candidates/`, never opened/viewed by the assistant.
- `obj.names` -- the 4 classes in project order (player, goalkeeper, referee, ball).
- `HANDOFF_MANIFEST.json` -- sequence/frame_number_1based/frame_index_0based/
  filename mapping for all 40 files.
- `README_HANDOFF.md` -- full ontology + bbox rules (reused verbatim from
  `TEST_ANNOTATION_PROTOCOL.md`), CVAT and Roboflow import instructions,
  what must be returned, and current access-state.

None of the 40 handoff frames have been visually inspected by the assistant
at any point (before or after the stop instruction).

## Next step to resume

This task is now blocked on external annotation of the 40 handed-off frames
(outside this session, in CVAT/Roboflow, per `README_HANDOFF.md`). When the
labels come back:
1. Convert the returned CVAT/Roboflow export (YOLO txt or COCO JSON) into
   the same per-frame record schema used in `ANNOTATIONS_DRAFT.json`
   (`{sequence, frame_number_1based, file, objects: [{class, bbox: [x1,y1,x2,y2]}]}`),
   using `HANDOFF_MANIFEST.json` to resolve identity, and merge with the
   existing 20 in-session records.
2. Section 8: completeness/validity QC (visual spot-check + machine checks:
   in-bounds, x2>x1/y2>y1, valid class ids, no exact-duplicate boxes,
   exactly 60 frame records).
3. Section 9: GT-only descriptive stats (no model comparison).
4. Section 10: freeze `TEST_DETECTION_ANNOTATIONS` + `TEST_ACCESS_STATE`
   with SHA256s, set final access-state flags, labels immutable after.
5. Section 11: prepare (do not execute) the M5 evaluation contract.
6. Section 12: re-verify freeze integrity (source-tree/manifest/weight SHAs unchanged).
7. Section 13/14: final M4 report per the exact template, then STOP (no M5 execution).

If the user instead wants the remaining 40 frames done in-session after all,
resume the adaptive-rigorous-pass method documented in this checkpoint's
prior revision (grid overview + targeted crops, ~2-4 image reads/frame) --
but only on explicit instruction, given the stop just issued.
