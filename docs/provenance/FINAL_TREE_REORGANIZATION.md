# Final Tree Reorganization

Structural pass only, on branch `chore/final-repository-cleanup`, on top of
the prior evidence/hygiene cleanup (`FINAL_REPOSITORY_CLEANUP.md`). No
scientific evidence deleted, no algorithms touched.

## What moved

**Root markdown → `docs/final/`** (final submission document set):
`FINAL_PROJECT_REPORT.md`, `README_FINAL_SUMMARY.md`, `FINAL_PROJECT_STATUS.md`,
`FINAL_PRESENTATION_OUTLINE.md`, `PIPELINE_DIAGRAM.md`.

**Root markdown → `docs/provenance/`** (post-freeze/audit engineering docs):
`POST_FREEZE_SYSTEM_PATCH.md`, `VISUALIZATION_PATCH_V2.md`,
`FINAL_REPOSITORY_CLEANUP.md` (the prior cleanup's own report).

**Demo renders → `demo_outputs/archive/`**: `tracked_output_team_v2_final.mp4`
and `tracked_output_viewer_v2_final_h264.mp4` — both cited from documentation
(team-assignment-v2 FINAL_RESULTS and VISUALIZATION_PATCH_V2 respectively) so
kept, but moved out of `final_e2e_demo/` so that directory contains only the
official demo (`tracked_output_final_system.mp4`) plus what's required to
reproduce it (`render_final_h264.py`, `cache/`, `reports/`,
`visualizations/`, `final_report.json`, `processing_stats.json`).

All moves used `git mv` — one canonical copy of every file, no leftovers.

## References updated

- `README.md` — all final-doc table links and the repository-layout tree
  point at the new `docs/final/`, `docs/provenance/` paths.
- `CLAUDE.md` — final-submission-doc pointers and key-directories section
  updated to the new paths.
- Every cross-link between the moved docs themselves: same-directory
  siblings kept bare filenames (still valid); cross-directory links
  (`docs/final/*` ↔ `docs/provenance/*`) rewritten to `../final/...` /
  `../provenance/...`.
- `docs/provenance/VISUALIZATION_PATCH_V2.md` and
  `experiments/post_freeze/team_assignment_v2/FINAL_RESULTS.md` — updated
  the two demo-mp4 paths they cite to `demo_outputs/archive/...`.
- `docs/archive/FINAL_SUBMISSION_AUDIT.md` left untouched — it is historical
  prose describing edits made while those docs still lived at root; not a
  live link.

Verified with `git grep` that no remaining live reference (link or bare
repo-relative path) points at a pre-move location; the one root-relative
mention outside `docs/` is an absolute `C:\Users\...\Desktop\...` path in
`experiments/records/experiment_B2/B2_FINAL_RESULTS.md` describing where a
run originally executed — historical text, not a repo link.

## Root, before → after

Before: `Anotated Data/`, `EyeCU_StageA_inputs/`, `FINAL_PRESENTATION_OUTLINE.md`,
`FINAL_PROJECT_REPORT.md`, `FINAL_PROJECT_STATUS.md`, `FINAL_REPOSITORY_CLEANUP.md`,
`LICENSE`, `PIPELINE_DIAGRAM.md`, `POST_FREEZE_SYSTEM_PATCH.md`, `README.md`,
`README_FINAL_SUMMARY.md`, `VISUALIZATION_PATCH_V2.md`, `best_A_960.pt`,
`best_B_1280.pt`, `best_C_960.pt`, `data/`, `demo_outputs/`, `docs/`,
`experimental/`, `experiments/`, `eyecu_football_v1.pt`, `full_pipeline.py`,
`input-videos/`, `models/`, `notebooks/`, `pytest.ini`, `requirements.txt`,
`rf_trackers/`, `run_pipeline.py`, `tests/`, `third_party/`, `tools/`,
`trackers/`, `yolov8n.pt` — 31 entries, 8 of them markdown reports.

After: same list minus the 7 moved markdown files — 24 entries, only
`README.md` and `CLAUDE.md` are markdown. See root classification table in
the chat response for the full per-entry justification.

## Left as-is, with reason

- `experiments/stage_a/` — kept under `experiments/`, unresolved run status
  is a scientific question out of scope for this pass, per instruction.
- `experimental/event_detection/` — kept; its own `README.md` already states
  "Not production code," satisfying the future-work separation requirement
  without adding a redundant top-level `experimental/README.md`.
- `Anotated Data/`, `notebooks/`, `input-videos/`, `EyeCU_StageA_inputs/` —
  data-adjacent root entries; not part of the documentation-tree reorg this
  pass targets, and moving them would be a data migration explicitly out of
  scope.
- `best_A_960.pt` and the other root `*.pt` files — left at root; runtime
  default paths depend on them.

## Verification

- `python -m py_compile run_pipeline.py full_pipeline.py trackers/*.py` — OK.
- `pytest tests/test_goalkeeper_possession.py tests/test_output_fps_contract.py`
  — 10/10 passed.
- All required assets confirmed present at their (possibly new) canonical
  paths — see the chat response's VERIFICATION section for the full list.
- `git grep` swept for every pre-move filename; no dangling references.

## Boundary

- TEST accessed: NO
- Inference runs: 0
- Algorithms changed: NO
- Metrics recomputed: NO
