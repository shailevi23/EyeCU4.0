# M3 — model / weight provenance

## Human detector

| | |
|---|---|
| checkpoint | `best_A_960.pt` (repo root) |
| SHA256 | `5eaf2e81d7f6b28fd0c665e769a5fb66ec71dc6be7f0d51576300a8370768e4a` |
| inference size | 960 |
| accept confidence | 0.25 |
| classes | player, goalkeeper, referee (ball comes from the second, SN3D branch — see below) |

The hash above was independently recomputed from the file on disk in this
session and matches the frozen provenance record at
`data/tracking_val_v1/manifest.json` (`detector.checkpoint_sha256`,
`imgsz: 960`, `confidence: 0.25`) exactly. As of this freeze,
`FootballAnalysisPipeline`'s constructor default, `run_pipeline.py`'s CLI
default, and `full_pipeline.py`'s own `__main__` example all default to this
checkpoint (see `PRODUCTION_CONFIG_VERIFICATION.md` — this was a genuine
mismatch found and fixed in this milestone, not merely re-verified).

## SN3D ball detector

| | |
|---|---|
| checkpoint | `yolo-sn-ball.pt` (third-party, not tracked in git) |
| SHA256 | `e8c1a900300893c34bf36c964c5854ed93603470e04a4a8eba73f70e4eea148b` |
| architecture | YOLO11l, 25,311,251 params, nc=1, `{0: 'ball'}` |
| inference size | 1280 |
| accept confidence | 0.25 (`BALL_ACCEPT_CONF`) |
| candidate-pool confidence | 0.10 (`BALL_CANDIDATE_CONF`) |
| source | SoccerNet-v3D official release v1.0.0 (`https://github.com/mguti97/SoccerNet-v3D`), asset `yolo-sn-ball.pt` |

SHA256 independently recomputed from the file on disk in this session and
matches `trackers/detector.py:SN3D_BALL_SHA256` and
`models/third_party/soccernet_v3d/README.md` exactly, and matches the value
given in this milestone's own instructions verbatim.

**Provisioning contract** (`trackers/detector.py:resolve_sn3d_ball_path`),
verified by reading the code, in order:
1. an explicit path argument,
2. the `EYECU_SN3D_MODEL_PATH` environment variable,
3. the repo-relative default `models/third_party/soccernet_v3d/yolo-sn-ball.pt`.

No search, no fallback to a different checkpoint. A missing file raises
`FileNotFoundError` with an actionable message (official-release URL, exact
expected path, expected SHA256). A present-but-wrong-hash file raises
`ValueError` from `verify_sn3d_ball_checkpoint` naming both the expected and
the actual digest. Both are fail-fast, not warnings. The 51 MB weight itself
is correctly excluded from git (`.gitignore:242`,
`models/third_party/**/*.pt`) — this freeze does not add it to the
repository; a clean checkout must provision it separately per the README
above before the ball branch can run at all, and will get a clear error if
it does not.

## BallTemporalSelector v1 (predeclared constants, unchanged since freeze)

```
ACCEPT_CONF                 0.25
CANDIDATE_CONF               0.10
MAX_RESCUE_GAP_SECONDS       0.6
GATE_BASE_PX                60.0
GATE_GROWTH_PX               40.0
MIN_HISTORY_FOR_VELOCITY       2
MAX_INTERP_GAP_SECONDS       0.4
CUT_MEAN_ABSDIFF            30.0
```

Verified against `trackers/ball_temporal.py` (module explicitly labelled
"PREDECLARED v1 SETTINGS") and enforced live by
`tests/test_experiment_temporal_candidate_recovery.py`. `BALL_CANDIDATE_CONF`
in particular stays 0.10 per T1's closed disposition (T1 tested 0.01 and it
was **not** adopted — see `experiments/records/experiment_T1/experiment_T1_record.md`).

## Human tracker

CBIoU (vendored Roboflow `CBIoUTracker`, `rf_trackers/core/cbiou/tracker.py`)
is the default/production association backend
(`trackers/football_tracker.py`'s `tracker_backend='cbiou'` default,
matched by both `FootballAnalysisPipeline` and `run_pipeline.py`). Not
tuned in this milestone; library defaults for its own thresholds
(`track_activation_threshold`, `minimum_iou_threshold_*`, etc.) are
unchanged. The ball never enters this backend — see
`trackers/football_tracker.py` lines ~395-402 and ~465-468 (ball detections
are filtered out before `boxes`/`class_ids` are built for the association
call, and are written to `tracks["ball"]` directly from the raw detector
output on a separate path).

## Class ontology

`player`, `goalkeeper`, `referee`, `ball` — `trackers/detector.py:CLASS_IDS`.
Unchanged.
