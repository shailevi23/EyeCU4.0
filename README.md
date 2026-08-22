# EyeCU 4.0 — football video analysis

Detects and tracks players, goalkeepers, referees and the ball in football
match video, assigns teams from jersey colour, and produces per-track
statistics. Runs fully offline on a locally trained detector.

University coursework project. Measured results, including the negative ones,
live in [docs/results/RESULTS.md](docs/results/RESULTS.md).

---

## What it does today

Two independent branches that merge only at possession assignment — the
ball never enters human association:

```
video ─┬─▶ HUMAN:  best_A_960.pt (YOLO26s@960; player/goalkeeper/referee)
       │             → CBIoU human tracking → team assignment
       └─▶ BALL:    SN3D_BASE, yolo-sn-ball.pt (YOLO11l@1280, ball only)
                     → BallTemporalSelector (provenance-tagged)

  both branches → PlayerBallAssigner / possession
                → annotated video + JSON reports
```

**Four semantic classes, preserved end to end:** `player`, `goalkeeper`,
`referee`, `ball`. Goalkeeper is never collapsed into player — its kit
deliberately differs from its own team's, and team assignment excludes it.
A goalkeeper may still be recorded as the ball *possessor*; it is never
given a fabricated team.

**Final scientific result, architecture detail, and status:** see
[docs/final/FINAL_PROJECT_REPORT.md](docs/final/FINAL_PROJECT_REPORT.md) /
[docs/final/README_FINAL_SUMMARY.md](docs/final/README_FINAL_SUMMARY.md) — held-out verdict
**B** (defensible, with a material held-out generalization limitation);
post-freeze NON-TEST development notes in
[docs/provenance/POST_FREEZE_SYSTEM_PATCH.md](docs/provenance/POST_FREEZE_SYSTEM_PATCH.md).

**Current detector candidate: `best_A_960.pt`** — YOLO26s trained at 960 px on
823 match-disjoint frames. On the frozen 208-image validation set: mAP50
**0.739**, player recall 0.916, referee recall 0.809, goalkeeper recall 0.515,
ball recall 0.486. Roughly **58 FPS** on a T4.

Two alternatives were trained and evaluated. `best_B_1280.pt` is more accurate
on the ball but costs 1.7×; `best_C_960.pt` was a data-side ablation and was
**rejected**. Both are kept for reproducibility. See
[docs/results/RESULTS.md](docs/results/RESULTS.md).

**Not calibrated:** speed and distance are reported in uncalibrated units and
are marked as such in the output. Do not quote them as m/s or km/h.

---

## Installation

Python 3.10+. A CUDA GPU is optional for inference and effectively required for
training.

```bash
pip install -r requirements.txt
```

Model weights are downloaded on first use. Roboflow is **optional** and used
only as a labelling aid; set `ROBOFLOW_API_KEY` in the environment if you want
it. Never hard-code a key.

---

## Usage

```bash
python run_pipeline.py --input input-videos/match.mp4 \
    --yolo-model best_A_960.pt --imgsz 960 --max-frames 300
```

Common options:

| flag | default | meaning |
|---|---|---|
| `--input` | required | input video path |
| `--yolo-model` | `yolov8x.pt` | detector weights — pass `best_A_960.pt` |
| `--imgsz` | 960 | detector inference size |
| `--conf` | 0.25 | detection confidence threshold |
| `--skip-frames` | 2 | process every Nth frame |
| `--max-frames` | all | cap frames processed |
| `--max-ball-gap` | 15 | frames the *tracker* may hold the last known ball box during association. Not the final ball output: `BallTemporalSelector` resolves the reported ball, and a frame it cannot resolve stays empty |
| `--tracker` | `cbiou` | association backend; `legacy` selects supervision ByteTrack for rollback |
| `--use-cache` | off | reuse cached detections/tracks |
| `--use-roboflow` | off | opt in to the hosted detector (labelling/benchmark only) |
| `--show-speed` / `--show-distance` | off | overlay uncalibrated speed/distance |
| `--overlay-mode` | `viewer` | clean tactical-camera render; `debug` keeps the engineering overlay |
| `--team-assignment-backend` | `legacy_color` | production default and benchmark winner (46/46 on the post-freeze dev benchmark); `v2` kept for rollback only |
| `--fps` | pipeline's `effective_fps` | output video FPS; defaults to `source_fps / skip_frames`, not a fixed guess |

Output lands in `--output-dir` (default `match_analysis_output/`): annotated
video, `visualizations/`, `reports/player_statistics.json`,
`final_report.json`, and a `cache/` keyed on video, model and settings.

Run `python run_pipeline.py --help` for the full list.

---

## Repository layout

```
run_pipeline.py            CLI entry point
full_pipeline.py           orchestrator
trackers/                  detector, tracking, team assignment, ball temporal logic
tools/                     dataset construction, labelling, evaluation, diagnostics
tests/                     ~1220 fast tests + ~10 slow
data/                      frames, labels, manifests, frozen splits (gitignored)
experiments/records/       per-experiment specs and training logs
experiments/post_freeze/   post-freeze benchmark evidence (team assignment v2, tracklet guard)
demo_outputs/final_e2e_demo/  official final demo render
docs/final/                final submission docs (report, summary, status, slides, diagram)
docs/provenance/           post-freeze/audit engineering documentation
docs/archive/              superseded/historical documents
experimental/              preserved, unintegrated code
```

---

## Documentation

| document | answers |
|---|---|
| [docs/final/FINAL_PROJECT_REPORT.md](docs/final/FINAL_PROJECT_REPORT.md) | the full final report — architecture, held-out result, limitations |
| [docs/final/README_FINAL_SUMMARY.md](docs/final/README_FINAL_SUMMARY.md) | one-page final summary and where to look |
| [docs/final/FINAL_PROJECT_STATUS.md](docs/final/FINAL_PROJECT_STATUS.md) | terse final status/verdict record |
| [docs/final/FINAL_PRESENTATION_OUTLINE.md](docs/final/FINAL_PRESENTATION_OUTLINE.md) | slide-by-slide presentation outline |
| [docs/final/PIPELINE_DIAGRAM.md](docs/final/PIPELINE_DIAGRAM.md) | architecture diagram with per-component validation status |
| [docs/provenance/POST_FREEZE_SYSTEM_PATCH.md](docs/provenance/POST_FREEZE_SYSTEM_PATCH.md) | post-freeze NON-TEST development (FPS, goalkeeper possession, team-assignment benchmark, tracklet guard) |
| [docs/provenance/VISUALIZATION_PATCH_V2.md](docs/provenance/VISUALIZATION_PATCH_V2.md) | viewer/overlay rendering patch and codec fix |
| [docs/provenance/FINAL_REPOSITORY_CLEANUP.md](docs/provenance/FINAL_REPOSITORY_CLEANUP.md) | repository cleanup/organization record |
| [docs/results/RESULTS.md](docs/results/RESULTS.md) | what was measured — detector results, failure modes, bugs found |
| [docs/coursework/COURSEWORK_PLAN.md](docs/coursework/COURSEWORK_PLAN.md) | what remains to finish the project |
| [docs/guides/LABELING.md](docs/guides/LABELING.md) | how to extract frames, draft labels and annotate |
| [docs/research/ball_architecture_audit.md](docs/research/ball_architecture_audit.md) | why the ball is hard for this architecture; detector freeze record |
| [docs/research/system_rescue_research.md](docs/research/system_rescue_research.md) | verified reference-repo findings and football-CV literature |
| [docs/research/EXTERNAL_DATASETS.md](docs/research/EXTERNAL_DATASETS.md) | SoccerNet / SoccerTrack / Roboflow assessment |
| [experiments/records/](experiments/records/) | experiment specs and raw training logs |
| [docs/archive/](docs/archive/) | superseded plans, kept as history |

---

## Tests

```bash
python -m pytest tests/ -m "not slow" -q
```

---

## License

See [LICENSE](LICENSE).
