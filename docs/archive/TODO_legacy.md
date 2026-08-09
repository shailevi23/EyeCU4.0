> **ARCHIVED — not the current roadmap.**
> This was the cleanup and refactor plan; its engineering work is complete and
> its measured outcomes live in [../results/RESULTS.md](../results/RESULTS.md).
> What remains to do is in
> [../coursework/COURSEWORK_PLAN.md](../coursework/COURSEWORK_PLAN.md).
> Kept for the decisions it records, not for its instructions.

---

# Claude Efficiency and Token-Cost Rules

* Begin with a short execution plan containing only:

  * files likely to change
  * dependencies to inspect
  * tests to run
* Use targeted repository searches such as `rg`, `git grep`, and symbol references before opening large files.
* Read only the relevant functions or line ranges first. Expand to the full file only when required.
* Do not repeatedly reopen unchanged files.
* Do not paste complete source files into responses. Show only:

  * concise findings
  * relevant diffs
  * errors requiring attention
  * final test results
* Batch related searches and file reads instead of inspecting one symbol at a time.
* Maintain a compact working summary containing:

  * confirmed architecture
  * completed changes
  * remaining tasks
  * unresolved risks
* Update that summary instead of re-analyzing the repository from scratch after every change.
* Use existing project patterns and dependencies before introducing new abstractions or packages.
* Prefer small edits to broad rewrites.
* Do not generate optional documentation, comments, wrappers, or abstractions unless they improve correctness or maintainability.
* Run the narrowest relevant tests after each change, then run the complete regression suite before completion.
* Do not rerun expensive video-processing tests when unrelated files changed.
* Reuse saved smoke-test fixtures, cached test videos, and frozen detector outputs when valid.
* Never reuse caches when the video, model, detector settings, tracker settings, or `skip_frames` changed.
* Do not reduce token usage by skipping:

  * dependency checks before deletion
  * security fixes
  * regression tests
  * error handling
  * validation of generated reports
  * comparison against the saved baseline
* When blocked, report the exact blocker, evidence, and safest next action instead of exploring unrelated alternatives.
* At completion, return only:

  1. concise change summary
  2. files changed or deleted
  3. tests and results
  4. before/after performance
  5. remaining risks
* Avoid narrating routine tool calls or repeating completed TODO items.
* Optimize for fewer repository reads and smaller responses, not fewer correctness checks.

---

# EyeCU 4.0 Cleanup TODO

## 1. Safety

* [x] Create branch: `cleanup/local-detector`
* [x] Tag current state: `pre-cleanup`
* [ ] **Revoke the exposed Roboflow API key.** ← needs your Roboflow account; no script can do it
* [x] Remove all hardcoded API keys.
* [x] Read `ROBOFLOW_API_KEY` only from environment variables.
* [x] Make Roboflow disabled by default. (`--use-roboflow` is now opt-in; was opt-out `--no-roboflow`)

## 2. Cleanup

Before deleting anything:

* [x] Search for all imports/references. (AST cross-reference of every def/class + pyflakes)
* [x] Confirm required functionality is migrated or unused.
* [x] Run relevant regression tests. (30-frame smoke test after each removal)

Then:

* [x] Delete unreachable code after `return stats` in `full_pipeline.py`.
* [x] Remove the duplicate detector pass in the advanced pipeline.
* [x] Remove duplicate orchestrator: `trackers/football_analysis.py`.
* [x] Remove outdated `SETUP_README.txt`.

Also removed in this pass:

* [x] The whole legacy path — `_process_video_legacy()`, `process_frame()`, `_detect_players()`,
  `_track_players()`, `_detect_interruptions()`, `_process_tracked_player()`,
  `_visualize_results()`, and the `--mode legacy` CLI option. `full_pipeline.py`: 801 → ~440 lines.
* [x] `detector.tflite` — inspected first: it is MediaPipe **BlazeFace** (2019 face detector),
  not a football model. Belonged to the deleted mesh/face code.
* [x] Unused `MatchRecorder` / `TrackingEvaluator` / `PerformanceAnalyzer` instantiation
  (constructed, never called — the only usages were commented out). This also stops the
  pipeline creating a stray empty `match_1.db`.
* [x] 5 `if __name__ == "__main__"` demo blocks in `trackers/` — every one loaded a
  nonexistent `input_video.mp4`, so none could run.
* [x] `TeamAssigner.assign_team_colors()` — superseded by `assign_teams_to_tracks()`.
* [x] ~30 unused imports; `__all__` added to `trackers/__init__.py`.
* [x] Empty `bodies/ faces/ meshes/ evaluation/ tracking_videos/` output dirs (legacy-only).

## 3. Archive legacy modules

Recoverable from the `pre-cleanup` tag; removed from the working tree:

* [x] `player_detection.py`
* [x] `player_tracking.py`
* [x] `id_loss_handler.py`
* [x] `face_body_crop.py`
* [x] `mesh_reconstruction.py`
* [x] `db_handler.py`
* [x] `test_mesh_pipeline.py`
* [x] `test_mesh_pipeline_mock.py`
* [x] `visualize_mesh_results.py`

Do not delete the purpose of `eval_benchmark.py`.

Later replace it with validated detection/tracking metrics.

* [x] `eval_benchmark.py` file removed (orphaned — nothing imported it).
* [ ] **Rebuild its purpose**: a real evaluation suite (mAP/recall per class, duplicate-box
  rate, ID switches, IDF1/HOTA, FPS) — see "Model Evaluation" and Phase 5 below.

## 4. Keep as active pipeline

Keep:

```text
run_pipeline.py
full_pipeline.py
trackers/
```

Production flow:

```text
local detector
→ duplicate suppression
→ tracker
→ team assignment
→ statistics
```

## 5. Detector refactor

* [x] Replace `roboflow_detector.py` with a generic detector interface.
  New `trackers/detector.py`: `BaseDetector` ABC + `create_detector()` factory.
  The old 634-line file mixed cloud, local and jersey OCR in one class; deleted.
* [x] Add `LocalDetector`. Ultralytics YOLO, no network, the production path.
* [x] Keep Roboflow only as an optional labelling/benchmark tool.
  `RoboflowDetector` is opt-in and takes a `LocalDetector` fallback, so a failed
  request degrades instead of aborting the run.
* [x] Preserve four classes:

  * `player`
  * `goalkeeper`
  * `referee`
  * `ball`
* [x] Do not convert `goalkeeper` into `player`.
  The old code did exactly this (`class_name = 'player'  # Normalize goalkeeper`).
  Goalkeepers now get their own `tracks["goalkeepers"]` list, their own colour,
  and stay out of jersey-colour team clustering.

Also dropped here: the EasyOCR jersey-number code and `process_video_batch()`,
both dead and both on the section 7 "do not work on" list.

**Known gap:** `player_ball_assigner` still only considers `tracks["players"]`,
so a goalkeeper holding the ball is not credited with possession. Revisit in Phase 4.

## 6. Fix bugs

* [x] Store returned tracks in `FootballTracker.tracks`.
* [x] Ensure `player_statistics.json` is generated. (needed a second fix too: numpy
  `int64` track ids are not valid JSON keys, so `json.dump` was still failing)
* [~] Calculate unique player IDs instead of using first-frame count.
  Half done. The report no longer claims a player count: it reports
  `unique_track_ids` and `track_statistics`, which is honest about being a
  count of tracks. The true physical-player count stays **unresolved** — with
  current ID churn (145 ids / 300 frames) any number derived from tracks would
  be wrong. Blocked on the fine-tuned detector and tracker work.
* [x] Read real video FPS.
  `get_video_info()` supplies it; reported as `source_fps`. Falls back to 30
  with a warning only if the container has no usable rate.
* [x] Use:

```text
effective_fps = source_fps / skip_frames
```

  Was hardcoded `fps = 30` in two places, so every speed was inflated by
  `skip_frames` (at the default skip=2, exactly 2x). Tested across
  skip_frames 1/2/3/5.

* [x] Stop copying the last ball box indefinitely.
* [x] Limit ball interpolation to a configurable frame gap.
  `--max-ball-gap` (default 15 frames). Past the cap the ball is reported unknown
  instead of frozen. Verified with `--max-ball-gap 3`: max consecutive hold was
  exactly 3, then the ball went unknown for the remaining frames.
  (Same run also showed COCO `yolov8n` detecting the ball in **1 of 60 frames** —
  this is the ball-recall problem the fine-tuned model has to solve.)
* [x] Create cache keys from:

  * video hash
  * model hash
  * detector settings
  * tracker settings
  * `skip_frames`

  `trackers/cache_utils.py`. The key is part of the filename
  (`tracks_<key>.pkl`), so incompatible caches coexist rather than overwrite.
  `max_frames` is included too — it changes which frames are in the run.
  Video identity is size + mtime + 1 MiB head/tail rather than a full hash;
  hashing a whole match would cost more than the run it protects.

### ⚠️ Speed / distance are UNCALIBRATED

`pixels_per_meter=12.0` in `trackers/speed_distance.py` is a guess, and it is
applied as one constant for the whole frame — which cannot hold for broadcast
footage, where perspective and zoom change the pixels-per-metre ratio across the
image and over time.

* Every km/h and metre figure is currently **relative only**, not real-world.
* Marked in the output: fields are named `*_UNCALIBRATED`, and
  `final_report.json` carries `speed_distance_calibrated: false`.
* Fixing it needs pitch homography from known line markings. Out of scope for
  now; do not quote these numbers until it is done.

## 7. Ignore for now

Do not work on:

* [ ] Jersey OCR
* [ ] Face recognition
* [ ] MediaPipe / 3D mesh
* [ ] Persistent player database
* [ ] Cross-match ReID
* [ ] Streamlit / UI
* [ ] Highlight generation
* [ ] Event detection — see `experimental/` below

### Preserved experimental code (NOT production)

`experimental/event_detection/event_detector.py` — goal/shot/sprint event
detection. The one unique piece kept from the retired standalone MVP; the rest
of it either duplicated production code or was on the "do not work on" list.

* Not imported by `run_pipeline.py`, `full_pipeline.py` or `trackers/`.
* Self-contained: numpy, cv2, stdlib only.
* Preserved for later evaluation. Integrating it is **out of scope** — it is
  covered by the "do not work on" list above.
* Never executed against real tracking data; it imports, which is not the same
  as it working.

## 8. Regression tests

Now **automated** under `tests/` (pytest). 41 tests, ~5s.
`pytest` runs everything; `pytest -m "not slow"` skips the 9 that load real
YOLO weights and decode video (~2s for the remaining 32).

* [x] Pipeline runs locally without network access. `test_runs_without_network`
* [x] Pipeline runs without Roboflow, EasyOCR or MediaPipe.
* [x] 30-frame smoke test completes. (integration run, 12 frames)
* [x] Output video is created.
* [x] `final_report.json` is created. `test_final_report_is_created`
* [x] `player_statistics.json` is created. `test_player_statistics_is_created`
* [x] Four detector classes survive the full pipeline.
  `test_all_four_classes_survive_the_pipeline`,
  `test_four_classes_survive_a_real_run`,
  `test_goalkeeper_is_not_merged_into_players`
* [x] Referees are never assigned to a team.
  `test_referees_are_never_assigned_a_team`, plus the same for goalkeepers and
  a counter-test that outfield players *do* get one, so it cannot pass by
  assigning nothing.
* [x] One detector inference occurs per processed frame.
  `test_exactly_one_inference_per_processed_frame`, `test_no_duplicate_detector_pass`
* [x] Different videos/models cannot reuse incompatible cache files.
  `tests/test_cache_safety.py` (12 tests) + two integration tests that assert
  two distinct cache files are actually written.
* [x] Ball becomes unknown after the configured missing-frame limit.
  `test_ball_becomes_unknown_after_max_gap`, parametrised over gaps 1/3/8.
* [x] Speed remains consistent when `skip_frames` changes.
  `test_speed_is_consistent_across_skip_frames` over 1/2/3/5, plus a guard test
  that the old hardcoded-FPS behaviour really was 2x wrong.
* [ ] Team-assignment accuracy does not regress.
* [ ] Duplicate-box rate does not regress.
* [ ] ID-switch count does not regress.
* [x] Local processing remains near or above the existing FPS baseline.
  300-frame benchmark (`yolov8n.pt`, `--imgsz 640`, `--skip-frames 2`), 3 runs:
  8.96 / 7.80 / 8.41 FPS — avg **8.39**, min 7.80, max 8.96 (14% spread).
  Earlier 15-frame figures (3.6–7.1) were dominated by model-load overhead and
  are not a usable baseline; this replaces them.

## 9. Final architecture requirement

Final project must contain:

* One CLI
* One main pipeline
* One detector interface
* One tracker interface
* One evaluation suite

`full_pipeline.py` must not import legacy OCR, face, mesh, database or custom ReID modules.

Status:

* [x] One CLI — `run_pipeline.py`.
* [x] One main pipeline — `full_pipeline.py`. Both duplicate orchestrators are
  gone: `trackers/football_analysis.py` and the retired MVP's second class,
  which was also named `FootballAnalysisPipeline`.
* [x] One detector interface — `trackers/detector.py`.
* [x] One tracker interface — `sv.ByteTrack` via `trackers/football_tracker.py`.
* [ ] One evaluation suite — still to build; see "Model Evaluation" below.
* [x] `full_pipeline.py` imports no OCR, face, mesh, database or ReID module.

---


---

# Everything beyond cleanup has moved

The sections that used to follow — Phase 2 (dataset), Guides 1–3, Phase 3
(training experiments), Phases 4–6 — are superseded. Keeping two copies of the
plan guarantees they drift apart, so they were removed rather than left to rot.

| What you want | Where it lives now |
|---|---|
| What to do next, in priority order | [../coursework/COURSEWORK_PLAN.md](../coursework/COURSEWORK_PLAN.md) |
| Frame extraction, drafting, annotation rules, split definitions | [../guides/LABELING.md](../guides/LABELING.md) |
| Measured results, failure modes, bugs found | [../results/RESULTS.md](../results/RESULTS.md) |
| SoccerNet / SoccerTrack / Roboflow assessment | [../research/EXTERNAL_DATASETS.md](../research/EXTERNAL_DATASETS.md) |
| Training experiments | `notebooks/EyeCU_Train_Colab.ipynb` |

The record above this line is kept because it documents what was removed from
the codebase and why — useful when something looks conspicuously absent.
