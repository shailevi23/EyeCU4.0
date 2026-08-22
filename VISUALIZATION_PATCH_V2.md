# Visualization Patch V2 — Viewer Overlay (post-freeze)

**This patch occurred AFTER the final scientific evaluation.** It is a
rendering/presentation-layer change only, applied on top of the already-closed
project (see [FINAL_PROJECT_STATUS.md](FINAL_PROJECT_STATUS.md),
[FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md)).

## What changed

Rendering/overlay plumbing only:

- `trackers/overlay.py` (new) — viewer/debug drawing helpers: responsive UI
  scale, text-with-outline, rounded pill labels, edge-safe label placement
  (`place_label`, clamped to the frame's safe margin), HUD text fitting
  (`fit_text_to_width`), player/ball markers, compact HUD.
- `trackers/football_tracker.py` — `draw_annotations(..., overlay_mode=)`
  branches into a new clean "viewer" renderer vs. the original "debug"
  renderer (made responsive, not otherwise changed in substance).
- `trackers/camera_movement.py` — `draw_camera_movement(..., overlay_mode=)`;
  a no-op in viewer mode, responsive/bounded in debug mode.
- `full_pipeline.py`, `run_pipeline.py` — thread `overlay_mode` /
  `--overlay-mode {viewer,debug}` through the existing entry point.
- `demo_outputs/overlay_preview/smoke_check.py` (new) — cheap, synthetic,
  no-model-run static/rendering check (corner-case edge-safety, HUD text
  fit, ball-ring bounds, monotonic scaling, mode switching).
- Polish pass v3 (`trackers/overlay.py`): ID pills now default to just
  below the feet (edge-safe fallback to just above only when below would
  clip the frame or lose a 3-try collision check), pill/marker/ball-ring
  sizes cut ~11-15%, possessor halo made thinner/semi-transparent.
- `full_pipeline.py` (`save_output_video`) — **output encoding defect
  fixed**: the writer requested `codec='XVID'` against an `.mp4` container,
  which the FFMPEG backend rejects; OpenCV silently fell back to `mp4v`
  (MPEG-4 Part 2), a visibly softer/blockier codec than the source's own
  H.264 despite a higher nominal bitrate. Now requests `codec='avc1'`
  (H.264), verified in this environment to actually encode (readback
  confirms a real `h264` fourcc, not a silent fallback).
- `demo_outputs/final_e2e_demo/render_final_h264.py` (new) — renders the
  final native 640x360 demo from the ORIGINAL source frames + the existing
  tracks cache (zero inference; the script verifies the cache is
  `cache_format=2` with `tracks` and `ball_candidates` before running, and
  aborts without falling back to inference if it is missing/invalid), and
  writes output FPS as `effective_fps = source_fps / skip_frames = 12.5`
  instead of the CLI's fixed default of 15, so the 375 processed frames
  play at their real timing (~30s) instead of sped up. No frame
  interpolation or duplication.

## What did NOT change

- Detector weights, thresholds, and configuration: unchanged.
- CBIoU / association logic: unchanged.
- BallTemporalSelector logic and states (`observed`, `recovered_low_conf`,
  `interpolated_short_gap`, `unknown`): unchanged — the overlay only
  *displays* the state already assigned; it computes nothing.
- Team assignment and possession (`has_ball`, `compute_team_ball_control`)
  logic: unchanged — the overlay only reads `has_ball` / the possession
  array that the existing pipeline already produced.
- Calibration: unchanged; speed/distance remain uncalibrated and this patch
  does not touch `speed_distance.py`.
- No TEST data, GT, metrics, or model files were read, run, or modified.

## Effect on evaluation

**M5.1 metrics continue to refer to the original evaluated source** in
`experiments/records/experiment_M5_1/` (mAP50 = 0.6175, mAP50-95 = 0.2697,
per-class breakdown as reported there and in
[FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md) §8). This patch does
**not** retroactively change, invalidate, or supersede any M5/M5.1 result —
it changes only how an already-produced output video is drawn for a human
viewer.

**Viewer V2 (`--overlay-mode viewer`, the default) is the final demo /
presentation renderer** going forward; `debug` remains available for
engineering inspection. The final native-resolution render is
`demo_outputs/final_e2e_demo/tracked_output_viewer_v2_final_h264.mp4`
(640x360, 12.5 fps, 375 frames, real H.264/avc1) — drawn on the original
decoded source frames plus the existing cached tracks/ball_candidates, not
on the earlier mp4v-encoded video, so no prior compression damage is baked
in. No scientific result, prediction, threshold, or tracking/selector/team/
possession logic changed to produce it.

## Provenance

- Evaluated source (last commit before this and all other post-freeze
  working-tree changes): `829f586` — "M3 manifest: record the final
  post-repair source-tree hash" (current `HEAD` at the time of this patch).
- M5, M5.1, and this visualization patch all exist as **uncommitted
  working-tree files** on top of that commit; none of the three has been
  committed separately, so there is no distinct "visualization-patch commit"
  to cite yet — only the file paths listed above, changed after
  `experiments/records/experiment_M5_1/` was frozen.
- No broad git audit was performed to produce this record, per scope.
