# CLAUDE.md

EyeCU 4.0 — football video analysis (player/ball tracking, team assignment,
possession). University coursework project, detector research frozen.

## Architecture

```
video ─┬─▶ HUMAN: best_A_960.pt (YOLO26s@960)  → CBIoU tracking → team assignment
       └─▶ BALL:  yolo-sn-ball.pt (YOLO11l@1280) → BallTemporalSelector
              both → PlayerBallAssigner/possession → annotated video + JSON
```

Entry point: `run_pipeline.py` → `full_pipeline.py` (`FootballAnalysisPipeline`),
which drives `trackers/` (football_tracker, team_assigner [+ team_assigner_v2
rollback backend], camera_movement, player_ball_assigner, overlay, ...).

## Key directories

- `trackers/` — production runtime code
- `tests/` — active test coverage for current implementation
- `experiments/records/` — frozen scientific evidence (GT, predictions,
  manifests, hashes, final reports) backing every milestone (M1–M5.1, B2,
  P1, R1, S1, etc.) and every rejected candidate
- `experiments/post_freeze/` — post-freeze evaluations (team assignment v2,
  tracklet guard) run after the system froze; non-adoption decisions live here
- `demo_outputs/final_e2e_demo/tracked_output_final_system.mp4` — the
  official final demo; other renders in that directory are cited duplicates
- `docs/results/RESULTS.md` — measured results including negative ones
- `docs/archive/` — superseded planning/audit documents kept for history

## Authoritative model paths (runtime contracts, not committed — obtain per
directory READMEs)

- `best_A_960.pt` — production human detector
- `yolo-sn-ball.pt` — production ball detector (`models/third_party/soccernet_v3d/`)

## Scientific boundaries

- Detector research is frozen. Do not retrain, retune, or reopen closed
  milestones without explicit instruction.
- **Never access TEST** (the held-out test split) unless explicitly
  authorized by the user for that specific task.
- Do not recompute or alter frozen scientific metrics; corrections get a new
  dated record, not an edit to a frozen one.

## Tests

Targeted runs only — do not run the full suite speculatively:

```bash
pytest tests/test_goalkeeper_possession.py tests/test_output_fps_contract.py -v
```

## Final submission docs

`README.md` (start here) → `FINAL_PROJECT_REPORT.md` (full) /
`README_FINAL_SUMMARY.md` (one-pager) / `FINAL_PROJECT_STATUS.md` (terse) /
`FINAL_PRESENTATION_OUTLINE.md` / `PIPELINE_DIAGRAM.md`. Post-freeze work:
`POST_FREEZE_SYSTEM_PATCH.md`, `VISUALIZATION_PATCH_V2.md`.
