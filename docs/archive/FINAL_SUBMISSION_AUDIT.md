# Final Submission Audit (Stage A)

Documentation-only audit and correction pass. No production code, models,
GT, or scientific metrics were touched. TEST was not accessed; no inference
of any kind was run.

## Files reviewed

- `FINAL_PROJECT_REPORT.md`
- `README_FINAL_SUMMARY.md`
- `FINAL_PRESENTATION_OUTLINE.md`
- `PIPELINE_DIAGRAM.md`
- `FINAL_PROJECT_STATUS.md`
- `README.md`
- `SUPPORTED_OUTPUT_CONTRACT.md` — **does not exist in the repository** (see Blockers)
- `POST_FREEZE_SYSTEM_PATCH.md`
- `VISUALIZATION_PATCH_V2.md` (read as a directly-referenced file from `FINAL_PROJECT_REPORT.md`)

## Files changed

- `FINAL_PROJECT_REPORT.md` — team-assignment status updated to disclose
  the post-freeze 46/46 development benchmark (without claiming held-out
  validation); possession section extended with the goalkeeper-possession
  correction; new §11a "Post-freeze development findings" summarizing the
  team-assignment benchmark, the 10/57 `MIXED_TRACK` finding (with its
  explicit scope limits), the tracklet-guard non-adoption, and the FPS
  default fix; new §12 "Final demo" with the verified render facts and the
  tracking-ID-not-jersey-number clarification; "See also" now links
  `POST_FREEZE_SYSTEM_PATCH.md`.
- `README_FINAL_SUMMARY.md` — status table's team-assignment/CBIoU/
  possession rows updated; new tracklet-consistency-guard row (not
  adopted); new "Post-freeze development" and "Final demo" sections;
  "Where to look" now links `POST_FREEZE_SYSTEM_PATCH.md` and the two
  `experiments/post_freeze/` benchmark directories.
- `FINAL_PROJECT_STATUS.md` — reporting-rule bullets extended with the
  post-freeze team-assignment measurement, the tracklet-guard non-adoption,
  and the goalkeeper-possession correction; closing line now points to
  `POST_FREEZE_SYSTEM_PATCH.md`.
- `FINAL_PRESENTATION_OUTLINE.md` — §9 updated for team assignment/
  possession; new §10 "Post-freeze development findings" (team-assignment
  benchmark, contamination finding with its scope limits, guard
  non-adoption, FPS fix); §10/§11 renumbered to §11/§12 accordingly, §12
  "Closing" updated to reference the new section.
- `PIPELINE_DIAGRAM.md` — new "Post-freeze note" paragraph (goalkeeper
  possession, guard non-adoption); the diagram itself was already correct
  (two branches, ball excluded from CBIoU) and was **not** redrawn.
- `README.md` — "What it does today" ASCII diagram replaced (it previously
  showed one linear detector→CBIoU→team-assignment→ball-selection chain,
  which reads as the ball passing through the same detector/association as
  humans; now shows the actual two-branch architecture with the ball
  explicitly bypassing CBIoU); added a pointer to the final report/verdict
  and to `POST_FREEZE_SYSTEM_PATCH.md`; CLI options table gained
  `--overlay-mode`, `--team-assignment-backend`, `--fps` rows (previously
  undocumented); Documentation table gained rows for the six root-level
  final-submission files, which it previously did not link at all; test
  count corrected from a stale "111 fast + 9 slow" to the actual ~1220
  fast / ~10 slow (pre-existing staleness from the project's long history,
  not introduced by recent work — corrected here since README quality was
  in scope).

No changes were made to `VISUALIZATION_PATCH_V2.md` — it was already
internally consistent with the authoritative facts and needed no edits.

## Inconsistencies found (and corrected above)

1. **Stale team-assignment claim.** All five primary docs still said "team
   assignment: implemented but unvalidated" with no mention of the
   post-freeze 46/46 development measurement, the 10/57 `MIXED_TRACK`
   finding, or the tracklet-guard evaluation/non-adoption — none of that
   post-freeze work was reflected anywhere in submission-facing docs before
   this audit. Fixed by adding it everywhere, consistently worded as
   **development-measured, not held-out TEST validated**.
2. **Missing goalkeeper-possession disclosure.** No doc mentioned that a
   goalkeeper can now be the recorded ball possessor, or the "team credit
   stays unknown, never fabricated" rule. Fixed.
3. **`README.md`'s architecture diagram implied a single combined
   detector/association path** ("detector → CBIoU association → team
   assignment → ball temporal selection"), which — read literally — puts
   ball temporal selection *after* CBIoU/team assignment in one chain, the
   exact "ball → CBIoU" / "one combined detector path" pattern flagged as
   incorrect. `FINAL_PROJECT_REPORT.md`, `README_FINAL_SUMMARY.md`,
   `FINAL_PRESENTATION_OUTLINE.md`, and `PIPELINE_DIAGRAM.md` already
   showed the correct two-branch architecture and needed no diagram fix —
   only `README.md` was stale. Fixed.
4. **No submission doc referenced `POST_FREEZE_SYSTEM_PATCH.md` or the
   `experiments/post_freeze/` benchmarks at all.** A reader following only
   the primary docs would never learn this work existed. Fixed by adding
   cross-links from every primary doc.
5. **No doc mentioned the official final demo file, its verified render
   facts, or that on-screen IDs are tracking IDs (not jersey numbers).**
   Fixed in `FINAL_PROJECT_REPORT.md` and `README_FINAL_SUMMARY.md`.
6. **`README.md`'s CLI options table was missing `--overlay-mode`,
   `--team-assignment-backend`, and `--fps`** entirely (all real,
   already-shipped flags). Fixed.
7. **`README.md`'s test count** ("111 fast tests + 9 slow") undercounts the
   actual suite by roughly 10x — clearly drifted over the project's
   history, not something recent. Fixed to the measured count.

## What was verified as already correct (no change needed)

- M5.1 numbers (per-class GT/predictions/TP/FP/FN/precision/recall/AP50/
  AP50-95, pooled mAP50/mAP50-95) match the authoritative values exactly in
  `FINAL_PROJECT_REPORT.md` and `README_FINAL_SUMMARY.md`.
- Como GT repair, reuse of the original M5 `RAW_PREDICTIONS.json`, "zero new
  inference in M5.1", "not a pristine one-shot sealed test", and "original
  M5 pooled metric superseded but preserved for history" are all disclosed
  correctly, consistently, and were not altered.
- No doc claims the working source tree is still byte-identical to the
  M3/M5 freeze; nothing needed correcting on that point.
- `PIPELINE_DIAGRAM.md`'s Mermaid diagram already correctly shows two
  branches merging only at `PlayerBallAssigner`, with the ball never
  entering CBIoU or team assignment.
- Claim boundaries (possession CLOSED-LIMITATION, calibration NOT
  VALIDATED, speed/distance/events UNSUPPORTED/DEFERRED) were already
  correctly stated everywhere they appeared.
- "Pristine"/"sealed" language is used only to correctly *deny* that status
  for M5.1 — no overclaim found.

## Remaining blockers

- **`SUPPORTED_OUTPUT_CONTRACT.md` does not exist in this repository.** It
  is named as a primary file in this audit's scope but was never created in
  any prior session. Per this task's own rules (documentation corrections
  only; no scientific/production redevelopment), a new authoritative
  contract document was **not fabricated** here — creating one would mean
  inventing its content rather than auditing existing text. Flagging this
  for the user to either supply/confirm its intended content or confirm it
  is not actually part of this submission.

No other genuine blockers were found. All other inconsistencies identified
above were text-only corrections and have been applied.
