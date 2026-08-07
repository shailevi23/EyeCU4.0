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
* [ ] Calculate unique player IDs instead of using first-frame count.
* [ ] Read real video FPS.
* [ ] Use:

```text
effective_fps = source_fps / skip_frames
```

* [x] Stop copying the last ball box indefinitely.
* [x] Limit ball interpolation to a configurable frame gap.
  `--max-ball-gap` (default 15 frames). Past the cap the ball is reported unknown
  instead of frozen. Verified with `--max-ball-gap 3`: max consecutive hold was
  exactly 3, then the ball went unknown for the remaining frames.
  (Same run also showed COCO `yolov8n` detecting the ball in **1 of 60 frames** —
  this is the ball-recall problem the fine-tuned model has to solve.)
* [ ] Create cache keys from:

  * video hash
  * model hash
  * detector settings
  * tracker settings
  * `skip_frames`

## 7. Ignore for now

Do not work on:

* [ ] Jersey OCR
* [ ] Face recognition
* [ ] MediaPipe / 3D mesh
* [ ] Persistent player database
* [ ] Cross-match ReID
* [ ] Streamlit / UI
* [ ] Highlight generation

## 8. Regression tests

Verified manually so far; **not yet automated** — these need to become a real test
file so they run on every change.

* [x] Pipeline runs locally without network access.
* [x] Pipeline runs without Roboflow, EasyOCR or MediaPipe.
* [x] 30-frame smoke test completes.
* [x] Output video is created.
* [x] `final_report.json` is created.
* [x] `player_statistics.json` is created.
* [ ] Four detector classes survive the full pipeline.
* [ ] Referees are never assigned to a team.
* [ ] One detector inference occurs per processed frame.
* [ ] Different videos/models cannot reuse incompatible cache files.
* [ ] Ball becomes unknown after the configured missing-frame limit.
* [ ] Speed remains consistent when `skip_frames` changes.
* [ ] Team-assignment accuracy does not regress.
* [ ] Duplicate-box rate does not regress.
* [ ] ID-switch count does not regress.
* [ ] Local processing remains near or above the existing FPS baseline.

## 9. Final architecture requirement

Final project must contain:

* One CLI
* One main pipeline
* One detector interface
* One tracker interface
* One evaluation suite

`full_pipeline.py` must not import legacy OCR, face, mesh, database or custom ReID modules.

---

# Phase 2 — Football Dataset

Start this phase only after cleanup and regression tests are stable.

## Dataset TODO

* [ ] Reach approximately **1,000–1,500 useful labeled frames**.
* [ ] Use frames from multiple independent matches/clips.
* [ ] Include different:

  * teams
  * kits
  * stadiums
  * camera angles
  * day/night lighting
  * crowd/no-crowd footage
* [ ] Label only:

  * `player`
  * `goalkeeper`
  * `referee`
  * `ball`

### Dataset split

Split by original match/video source:

```text
70% train
15% validation
15% test
```

Never put frames from the same original match into different splits.

---

# Guide 1 — Get Images

Current dataset already contains multiple diverse short match clips.

For short ~1-minute videos, extract around every **3 seconds**:

```bash
python tools/extract_frames.py \
  --videos-dir input-videos \
  --out data/frames \
  --max-frames 300 \
  --interval-sec 3
```

Prefer diversity over large numbers of nearly identical frames.

Include:

* Normal gameplay
* Corners
* Tackles
* Crowded penalty areas
* Referees
* Goalkeepers
* Small/distant balls
* Motion blur
* Different camera angles
* Different lighting conditions

Do not use hundreds of consecutive near-identical frames.

Target:

```text
~1,000–1,500 useful frames
```

A first training run may start around **900–1,200 good labeled frames** if diversity and label quality are strong.

---

# Guide 2 — Model-Assisted Labelling

Use **Roboflow Annotate** or **CVAT**.

1. Upload extracted frames.
2. Use the existing Roboflow football detector to generate initial boxes.
3. Manually review predictions.
4. Add:

   * missed players
   * missed balls
   * missed referees
   * missed goalkeepers
5. Fix incorrect classes.
6. Remove duplicate boxes.
7. Check crowded and occluded scenes carefully.
8. Export in **Ultralytics YOLO format**.

Roboflow is allowed as a temporary **labelling assistant**.

It must not remain required for production inference.

---

# Phase 3 — Model Training & Selection

## Primary model decision

Main model:

```text
YOLO26s
imgsz=960
```

Use pretrained weights:

```python
YOLO("yolo26s.pt")
```

Do not train from scratch.

---

# Guide 3 — Train in Google Colab

Install:

```python
!pip install -U ultralytics
```

Check GPU:

```python
!nvidia-smi
```

Mount Google Drive and save training outputs/checkpoints there.

---

## Experiment A — Main Model

Train **YOLO26s @ 960px** first.

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")

model.train(
    data="/content/football_dataset/football.yaml",
    epochs=80,
    imgsz=960,
    batch=-1,
    patience=20
)
```

This is the main candidate.

---

## Experiment B — Small Object / Ball Test

Only after Experiment A:

Train/test the same model at:

```text
YOLO26s @ 1280px
```

Purpose:

Determine whether higher resolution significantly improves:

* ball recall
* distant player recall
* goalkeeper/referee detection

Compare the improvement against the FPS cost.

Do not automatically choose 1280 just because accuracy is slightly higher.

---

## Experiment C — Speed Model

Train/test:

```text
YOLO26n @ 960px
```

Purpose:

Determine whether YOLO26n provides similar accuracy with significantly better inference speed.

If accuracy is close enough to YOLO26s, YOLO26n may become the production model.

---

## Baseline

Keep one YOLOv8 experiment only as a historical baseline:

```text
YOLOv8s
```

Do not spend significant optimization time on YOLOv8.

Its purpose is to prove whether the new fine-tuned model actually improves the existing system.

---

# Models NOT to Prioritize

Do not spend time on these during the first training phase:

* [ ] YOLO12
* [ ] YOLO26l
* [ ] YOLO26x
* [ ] YOLO-World
* [ ] P2 architecture

P2 should only be investigated if:

```text
YOLO26s @ 960
+
YOLO26s @ 1280
+
better ball training data
```

still fail to provide acceptable ball detection.

---

# Model Evaluation

For every trained model record:

* [ ] mAP50
* [ ] mAP50-95
* [ ] Player precision
* [ ] Player recall
* [ ] Goalkeeper precision
* [ ] Goalkeeper recall
* [ ] Referee precision
* [ ] Referee recall
* [ ] Ball precision
* [ ] Ball recall
* [ ] Duplicate-box rate
* [ ] False positives
* [ ] FPS

Most important EyeCU metrics:

```text
ball recall
referee recall
goalkeeper recall
duplicate detections
FPS
```

Do not select a model using overall mAP alone.

---

# Model Selection Rule

Select the best balance of:

```text
Detection accuracy
+
Ball recall
+
Role classification
+
Low duplicate rate
+
Inference speed
```

All candidates must be evaluated on the **same unseen test matches**.

---

# If YOLO26s @ 960 Works Well

* [ ] Save `best.pt`.
* [ ] Test on completely unseen match footage.
* [ ] Integrate into `LocalDetector`.
* [ ] Run full EyeCU regression suite.
* [ ] Compare new detection results against Roboflow baseline.
* [ ] Remove Roboflow from production inference.
* [ ] Benchmark ByteTrack using the new detector.

---

# If Ball Detection Is Still Weak

Do these in order:

1. [ ] Compare YOLO26s @ 1280.
2. [ ] Add more difficult ball examples.
3. [ ] Add frames where the model misses the ball.
4. [ ] Retrain using active learning.
5. [ ] Only then investigate:

   * P2 architecture
   * dedicated ball detector

Do not change architecture before improving the training data.

---

# Optional Performance Optimization

If YOLO26s is accurate but too slow:

* [ ] Test YOLO26n @ 960.
* [ ] Compare accuracy/FPS directly.
* [ ] Optionally investigate knowledge distillation:

```text
YOLO26s
   ↓ teacher

YOLO26n
   ↓ student

fast production model
```

Do this only after the standard YOLO26n benchmark exists.

---

# Phase 4 — Detection Post-Processing

After selecting a detector:

* [ ] Tune confidence thresholds.
* [ ] Add duplicate-box suppression.
* [ ] Do not aggressively suppress legitimate overlapping players.
* [ ] Measure duplicate-box rate before and after changes.

---

# Phase 5 — Tracking

Only after detector output is stable:

1. [ ] Benchmark current ByteTrack.
2. [ ] Tune ByteTrack parameters.
3. [ ] Compare with BoT-SORT.
4. [ ] Test BoT-SORT + ReID if needed.

Compare:

* ID switches
* IDF1
* HOTA if available
* track fragmentation
* FPS

Do not change detector and tracker simultaneously during experiments.

Use frozen detector outputs when comparing trackers.

---

# Phase 6 — Deployment

After choosing the final detector/tracker:

* [ ] Export final detector to ONNX.
* [ ] Benchmark ONNX Runtime.
* [ ] Benchmark OpenVINO for CPU deployment.
* [ ] Benchmark TensorRT if NVIDIA GPU is available.
* [ ] Select production backend based on measured performance.
* [ ] Keep `.pt` model for future fine-tuning.

---

# Final Target Architecture

```text
Football video
      ↓
Fine-tuned YOLO26 detector
      ↓
Duplicate suppression
      ↓
ByteTrack / BoT-SORT
      ↓
Role filtering
      ↓
Team assignment
      ↓
Ball possession
      ↓
Speed / distance
      ↓
Reports
```

Primary training target:

```text
YOLO26s @ 960
```

Fallback accuracy experiment:

```text
YOLO26s @ 1280
```

Speed experiment:

```text
YOLO26n @ 960
```

Production goal:

```text
Fully local football-specific detector
with no Roboflow dependency during normal inference.
```
