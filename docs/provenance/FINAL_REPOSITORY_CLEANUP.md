# Final Repository Cleanup

Executed on branch `chore/final-repository-cleanup`, commit `dabfb21`.
Repository architecture/hygiene only — no algorithms, models, or metrics
changed.

## Before → after

| | Before | After |
|---|---|---|
| Tracked files | 661 | 1,410 |
| Untracked (non-ignored) | 21 top-level items | 0 |
| Ignored paths | ~93 | ~93 (rules widened, same governed trees) |

Working tree stayed ~74 GB throughout (dominated by `EyeCU_external_data/`
59 GB and `eye_env/` 3.4 GB — both untouched, already ignored, not
repository content). `.git` grew to ~445 MB after committing ~155 MB of
newly-tracked evidence (annotation grids, benchmark JSON, demo renders).

## Removed (untracked, never in git history — no data loss)

- `demo_outputs/final_e2e_demo/tracked_output.mp4`,
  `tracked_output_viewer_v2.mp4` — pre-H.264-fix demo renders, superseded by
  `tracked_output_final_system.mp4`; grep confirmed zero references in any
  current `.md`.
- `demo_outputs/overlay_preview/*.jpg` (10 files) — polish/debug preview
  frames, zero doc references.
- `FINAL_CLEANUP_PLAN.md` — the prior, shallow cleanup pass that stopped
  before execution; superseded by this document.
- Stray `__pycache__/` directories outside `eye_env/` (repo root, trackers/,
  tests/, tools/, rf_trackers/, third_party/, and inside experiment dirs).

~11 MB reclaimed on disk from the demo files; `__pycache__` cleanup is
regenerated automatically and untracked either way.

## Organized (moved, no content changes)

- `EyeCU_B2_results/` → `experiments/records/experiment_B2/raw_runs/` —
  the raw Ultralytics training-run evidence behind the B2 NOT-ADOPT verdict.
- `EyeCU_R1_results/` → `experiments/records/experiment_R1/`.
- `EyeCU_S1_results/` → `experiments/records/experiment_S1/` — two docs in
  `experiments/records/experiment_P1/` cited the old root path
  (`EyeCU_S1_results/p0/P0_POSSESSION_BASELINE.json`); both updated to the
  new path so the citation stays valid.
- `new.md`, `EYECU_FACT_CHECKED_TRACKING_TODO_FOR_CLAUDE.md` → `docs/archive/`
  — pre-dated the FINAL_* doc set, same genre as the existing
  `docs/archive/TODO_legacy.md`.
- `FINAL_SUBMISSION_AUDIT.md` → `docs/archive/` — its own audit trail,
  content already reflected in the current final docs.

## Tracked (were legitimate untracked deliverables/evidence)

- Root final-submission docs: `FINAL_PROJECT_REPORT.md`,
  `README_FINAL_SUMMARY.md`, `FINAL_PROJECT_STATUS.md`,
  `FINAL_PRESENTATION_OUTLINE.md`, `PIPELINE_DIAGRAM.md`,
  `POST_FREEZE_SYSTEM_PATCH.md`, `VISUALIZATION_PATCH_V2.md`.
- `trackers/overlay.py`, `trackers/team_assigner_v2.py` — confirmed live
  imports from `football_tracker.py`/`camera_movement.py` and
  `full_pipeline.py` (v2 team-assignment rollback backend), not dead code.
- `tests/test_goalkeeper_possession.py`, `tests/test_output_fps_contract.py`
  — both import real, current production symbols; both pass (10/10).
- `experiments/post_freeze/` (team_assignment_v2, tracklet_guard_v1) —
  post-freeze benchmark evidence, including the tracklet-guard result that
  supports its documented non-adoption.
- `experiments/records/experiment_M4/`, `experiment_M5/`, `experiment_M5_1/`,
  `experiment_B2/B2_PREFLIGHT_RECORD.md` — GT construction, frozen
  predictions, hashes, final reports.
- `demo_outputs/` (after pruning above) — official demo plus the two
  duplicate/intermediate renders that `VISUALIZATION_PATCH_V2.md` and
  `experiments/post_freeze/team_assignment_v2/FINAL_RESULTS.md` cite by name.

## .gitignore

Kept the existing 240-line project-specific ignore file intact (it already
correctly governs `EyeCU_external_data/`, `data/`, bake-off outputs, and the
Keremberke review package with careful allow-lists — verified with
`git check-ignore -v` spot checks). Appended a standard hygiene block:
`*.pyc`/`*.pyo`, `.mypy_cache/`, `.ruff_cache/`, coverage artifacts,
`.DS_Store`/`Thumbs.db`/`Desktop.ini`, `.idea/`, `.vscode/*.local.json`,
`.env.*`, `*.key`, `.claude/settings.local.json`, `*.log`/`*.tmp`/`*.temp`.

No previously-tracked files needed `git rm --cached` — nothing generated or
secret was ever committed. No secrets found anywhere in the tree (only
`.env`, untracked and already ignored).

## Claude Code structure

- Added `CLAUDE.md` (none existed) — architecture, entry point, key
  directories, authoritative model paths, scientific boundaries (frozen
  detector research, no TEST access without explicit authorization), and
  the two targeted tests to run.
- Added `.claude/skills/repo-hygiene/SKILL.md` for future maintenance passes.

## Verification

- `python -m py_compile run_pipeline.py full_pipeline.py trackers/*.py` — OK.
- `pytest tests/test_goalkeeper_possession.py tests/test_output_fps_contract.py`
  — 10/10 passed.
- Grep across all `.md` for every deleted/moved filename — no dangling
  references remain (one hit in `experiment_B2/B2_FINAL_RESULTS.md` is an
  absolute `C:\Users\...\Desktop\...` path documenting where the run
  originally happened, not a repo-relative link — left as historical text).
- Required final assets all present: `best_A_960.pt`,
  `models/third_party/soccernet_v3d/yolo-sn-ball.pt`, all seven root
  FINAL_*/README docs, M5.1/post-freeze/tracklet-guard evidence,
  `demo_outputs/final_e2e_demo/tracked_output_final_system.mp4`,
  `experiments/records/experiment_S1/` (relocated Bayern possession
  baseline evidence).
- Spot-checked `git check-ignore -v` on every new `.gitignore` pattern
  against the exact staged evidence set — nothing scientific got
  accidentally ignored.

## Left untouched, on purpose

- Six pre-existing unstaged edits present before this cleanup began
  (`README.md`, `experiments/external_sources/keremberke_review/SECOND_PASS_GATE.json`,
  `full_pipeline.py`, `run_pipeline.py`,
  `trackers/{camera_movement,football_tracker,player_ball_assigner}.py`) —
  in-progress work unrelated to this cleanup, not reverted or committed.
- `experiments/stage_a/STAGE_A_INPUT_MANIFEST.json` — a preflight manifest
  for a Colab data-augmentation run with no result/verdict file anywhere in
  the repo. Genuinely unresolved whether that run executed or was
  abandoned before it started; kept and now tracked as-is rather than
  guessed at.
- `experimental/event_detection/event_detector.py` — zero production
  callers, but has its own README and is referenced from
  `docs/archive/TODO_legacy.md` and `experiments/post_freeze/team_assignment_v2/CURRENT_IMPLEMENTATION.md`
  as explicitly out-of-scope future work, not orphaned/dead code.
- `data/backups/pseudo_meta_*/`, `data/pseudo_meta/` empty per-match
  placeholder directories — pre-existing scaffolding, untracked by git
  either way (git does not track empty directories), not part of this
  session's untracked-file surface.

## Boundary

- TEST accessed: NO
- Model/detector inference run: 0
- Production algorithms changed: NO
- Scientific metrics recomputed: NO
