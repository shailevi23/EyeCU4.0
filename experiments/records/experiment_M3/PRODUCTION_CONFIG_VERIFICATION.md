# M3 — production config vs frozen measured config

Narrow inspection of the exact constructor/CLI defaults a normal user
invokes, against the frozen configuration measured in closed experiments.
Read-only investigation first; one genuine mismatch found and fixed.

| # | setting | expected (frozen) | actual current code | status |
|---|---|---|---|---|
| 1 | Ball detector backend | SN3D_BASE, `yolo-sn-ball.pt`, imgsz=1280, accept=0.25 | `trackers/detector.py`: `SN3D_BALL_IMGSZ=1280`, `BALL_ACCEPT_CONF=0.25`, `BALL_CANDIDATE_CONF=0.10`; `full_pipeline.py` constructor default `ball_detector_backend='sn3d'`; both `tools/eval_possession_val.py` and `tools/eval_possession_val_p1.py` pass `ball_detector_backend='sn3d'` explicitly | MATCH |
| 2 | BallTemporalSelector v1 constants | see `MODEL_PROVENANCE.md` | identical, module labelled "PREDECLARED v1 SETTINGS"; enforced live by `tests/test_experiment_temporal_candidate_recovery.py` | MATCH |
| 3 | Human tracker (association) default | CBIoU | `trackers/football_tracker.py` `tracker_backend='cbiou'`; `full_pipeline.py` `tracker_backend: str = 'cbiou'`; `run_pipeline.py --tracker` default `"cbiou"`; T2 gate record shows CBIoU passed all 9 adoption criteria at library defaults | MATCH |
| 4 | Ball excluded from CBIoU | comment "ball's single canonical path... written here and nowhere else" | present, unchanged, `trackers/football_tracker.py` ~395-402 and ~465-468 | MATCH |
| 5 | **Human detector checkpoint default** | `best_A_960.pt` (`data/tracking_val_v1/manifest.json`) | **was** `yolov8x.pt` in `FootballAnalysisPipeline.__init__`, `full_pipeline.py`'s own `__main__` CONFIG, and `run_pipeline.py --yolo-model` | **MISMATCH — fixed in this milestone** (see below) |
| 6 | `ball_candidate_pool` default | on | `full_pipeline.py`: `ball_candidate_pool: bool = True` | MATCH |
| 7 | SN3D checkpoint resolution order | explicit arg > `EYECU_SN3D_MODEL_PATH` > repo-relative default; fail-fast on missing/wrong hash | confirmed by reading `resolve_sn3d_ball_path` / `verify_sn3d_ball_checkpoint` — see `MODEL_PROVENANCE.md` | MATCH |

## The one real mismatch, and the fix

`FootballAnalysisPipeline.__init__`'s `yolo_model` parameter, the
`run_pipeline.py` CLI's `--yolo-model` flag, and `full_pipeline.py`'s own
`__main__` example `CONFIG` dict all defaulted to `'yolov8x.pt'` — a
generic, untrained-for-this-task COCO checkpoint — not the closed, measured
production human detector `best_A_960.pt`. A user running either normal
entry point (`python run_pipeline.py`, or `python full_pipeline.py`
directly) without explicitly overriding the model would silently get the
wrong detector, with no error, no warning, and predictions that do not
match anything ever measured in a closed experiment.

**Fixed**: all three defaults changed to `'best_A_960.pt'`. No other
behaviour changed — the parameter can still be overridden explicitly for an
intentional experiment. No test previously encoded the old value ('yolov8x'
does not appear anywhere in `tests/`), so nothing else needed updating for
consistency. Locked going forward by
`tests/test_production_config.py::TestHumanDetectorDefault`, which reads
the value out of `data/tracking_val_v1/manifest.json` rather than
hardcoding it, so the test and the frozen provenance record can never
silently drift apart from each other.

## Coverage gap noted, not fixed (not prediction-affecting)

`run_pipeline.py` has no CLI flag for `ball_detector_backend` — a CLI user
cannot select it explicitly, only inherit whatever `full_pipeline.py`'s
constructor default is (`'sn3d'`, correct). Not a value mismatch, so left
alone per "fix only real mismatches, no tuning."
