# EyeCU 4.0 — football video analysis

Detects and tracks players, goalkeepers, referees and the ball in football
match video, assigns teams from jersey colour, and produces per-track
statistics. Runs fully offline on a locally trained detector.

University coursework project. Measured results, including the negative ones,
live in [docs/results/RESULTS.md](docs/results/RESULTS.md).

---

## What it does today

```
video → detector (YOLO26s @960) → ByteTrack association → team assignment
      → ball possession → annotated video + JSON reports
```

**Four semantic classes, preserved end to end:** `player`, `goalkeeper`,
`referee`, `ball`. Goalkeeper is never collapsed into player — its kit
deliberately differs from its own team's, and team assignment excludes it.

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
| `--max-ball-gap` | 15 | frames the last known ball box may be held before the ball is reported unknown |
| `--use-cache` | off | reuse cached detections/tracks |
| `--use-roboflow` | off | opt in to the hosted detector (labelling/benchmark only) |
| `--show-speed` / `--show-distance` | off | overlay uncalibrated speed/distance |

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
tests/                     111 fast tests + 9 slow
data/                      frames, labels, manifests, frozen splits (gitignored)
experiments/records/       per-experiment specs and training logs
docs/                      documentation (below)
experimental/              preserved, unintegrated code
```

---

## Documentation

| document | answers |
|---|---|
| [docs/results/RESULTS.md](docs/results/RESULTS.md) | what was measured — detector results, failure modes, bugs found |
| [docs/coursework/COURSEWORK_PLAN.md](docs/coursework/COURSEWORK_PLAN.md) | what remains to finish the project |
| [docs/guides/LABELING.md](docs/guides/LABELING.md) | how to extract frames, draft labels and annotate |
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
