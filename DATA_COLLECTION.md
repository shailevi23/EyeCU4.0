# Data Collection & Training — Runbook

Everything needed to go from raw match videos to a trained football detector.
The tooling is built and tested; the only missing input is **footage**.

Background: [TODO.md](TODO.md) — "Phase 2 — Football Dataset", "Phase 3 — Model Training & Selection".

---

## Before you start

| Requirement | Status |
|---|---|
| `tools/extract_frames.py` | ✅ built, tested |
| `tools/pseudo_label.py` | ✅ built, tested |
| `tools/build_dataset.py` | ✅ built, tested |
| `notebooks/EyeCU_Train_Colab.ipynb` | ✅ built |
| `requirements.txt` | ✅ added |
| Pipeline runs fully offline by default | ✅ verified |
| **4–6 full match videos** | ❌ **you must supply these** |
| Roboflow API key (optional, speeds up labelling) | ❌ needs rotating |

The repo only contains two short clips (`input-videos/short.mp4` 51s,
`08fd33_4.mp4` 30s) from what is effectively one domain. That is not enough
for 1,500 frames, and `build_dataset.py` will refuse to build a split from
fewer than 3 matches — by design, since a match-disjoint split is the whole
point.

**What to gather:** 4–6 matches, ideally differing in stadium, kit colours,
lighting (day / night / floodlit) and broadcast style. Full matches are best;
10–15 minute segments per match are workable.

---

## Step 1 — Extract frames

```bash
pip install -r requirements.txt

# put your videos in input-videos/, one file per match
python tools/extract_frames.py --videos-dir input-videos --out data/frames \
    --interval-sec 3 --max-frames 300
```

Each video becomes one `match_id` (from its filename) and frames are named
`<match_id>_<frame>.jpg` — that naming is what makes the match-disjoint split
possible later, so **do not rename or reshuffle the files**.

The sampler does three things beyond a plain `ffmpeg fps=1/5`:

- **routine coverage** — one frame every `--interval-sec` seconds
- **motion bursts** — extra consecutive frames when motion spikes above the
  local median (corners, tackles, fast pans — the hard cases Guide 1 asks for)
- **filtering** — drops near-duplicates (dHash) and the blurriest 15% of
  candidates. The blur threshold is computed **per video**, because broadcast
  sources differ enormously in intrinsic sharpness (measured here: median
  Laplacian variance 282 for `08fd33_4.mp4` vs 24 for `short.mp4` — one fixed
  threshold would have emptied out the softer video entirely).

Tuning to hit ~1,500 total: `--max-frames 300` × 5 matches = 1,500. If a match
yields too few, lower `--interval-sec`. The script warns if you finish under
1,000 frames or under 4 matches.

**Also deliberately include** (these are the classes that fail):
goalkeepers in kits close to the referee's, assistant referees near the
touchline, goalkeepers outside the box, distant/small players, and hard
negatives such as crowd, bench and advertising-board shots (an empty label file
is valid and useful).

## Step 2 — Pseudo-label

```bash
# Roboflow backend (keeps the goalkeeper class) — needs a key:
$env:ROBOFLOW_API_KEY = "your-new-key"
python tools/pseudo_label.py --frames data/frames --labels data/labels --backend roboflow

# or entirely offline:
python tools/pseudo_label.py --frames data/frames --labels data/labels \
    --backend local --model yolov8x.pt
```

Classes are fixed project-wide: `0 player`, `1 goalkeeper`, `2 referee`, `3 ball`.
Team identity is **not** a detector class — it stays in `trackers/team_assigner.py`.

⚠️ A local COCO model can only produce `player` and `ball`. Verified on the
sample clips: 1,673 player, 27 ball, **0 goalkeeper, 0 referee**. If you label
offline you will be drawing every goalkeeper and referee box by hand. Prefer
the Roboflow backend for this step — it is exactly the "temporary
labelling assistant" role TODO.md Guide 2 assigns it.

## Step 3 — Correct the labels by hand (the part that actually matters)

Upload `data/frames` + `data/labels` to **Roboflow Annotate** or **CVAT** and fix:

- every `goalkeeper`, `referee` and `ball` box — check all of them, not a sample
- missed players, especially small/distant ones
- duplicate boxes on one player (the failure this whole project is chasing)
- `person` mislabels: coaches, substitutes, medical staff, ball boys

Export back in **Ultralytics YOLO format** into `data/labels/`, preserving the
`<match_id>/` subfolders.

This is the slow step: budget roughly 1–3 minutes per frame, i.e. **25–75 hours
for 1,500 frames**. Splitting it across people is the usual answer.

## Step 4 — Build the split

```bash
python tools/build_dataset.py --frames data/frames --labels data/labels \
    --out data/dataset --zip
```

Produces `data/dataset/{train,val,test}/{images,labels}`, a `football.yaml`,
a `split_report.json`, and `football_dataset.zip` for Colab. It splits **whole
matches** at 70/15/15 by frame count and hard-fails if any match lands in two
splits. Exact ratios are impossible with whole matches — with 5 equal matches
you get roughly 60/20/20, which is fine and still leak-free.

Read the per-class instance counts it prints. If `val` or `test` has zero
goalkeepers or referees, those per-class scores will be meaningless and you
should re-balance which matches carry the rare classes.

## Step 5 — Train on Colab

Upload `football_dataset.zip` to Drive, open
[notebooks/EyeCU_Train_Colab.ipynb](notebooks/EyeCU_Train_Colab.ipynb) in Colab,
set `Runtime → GPU`, edit `ZIP_PATH`, and run top to bottom.

The notebook verifies the split has no leaked images, runs a 10-epoch pilot to
time the real run, then experiments A–D (YOLOv8s@640, YOLO26n@960, YOLO26s@960,
YOLO26s@1280), compares them on **val**, scores the winner **once** on the
held-out test matches, and exports ONNX to Drive.

Planning ranges: 1,500 frames at 960px on a Colab T4 is roughly
2–6 hours per 100-epoch run. Free-tier Colab disconnects — either run one
experiment per session or use Colab Pro. If `yolo26*.pt` is not yet in the
installed Ultralytics version the notebook skips that cell rather than crashing;
substitute `yolo11n.pt` / `yolo11s.pt`.

## Step 6 — Back in the repo

```bash
python run_pipeline.py --input input-videos/short.mp4 \
    --yolo-model eyecu_football.pt --max-frames 300
```

Compare FPS and detection quality against the Roboflow baseline in
[RESULTS.md](RESULTS.md), then continue with [TODO.md](TODO.md) Phase 4
(detection post-processing) and Phase 5 (tracking).

---

## Before running Step 2 with Roboflow

The old key `bzHGvvsL4gjqNOHIPR5J` was committed to this repo and must be treated
as compromised. Revoke it in the Roboflow dashboard and issue a new one — that
needs your account, no script can do it. Nothing reads a hardcoded key any more;
`ROBOFLOW_API_KEY` is the only source, and without it the pipeline runs local-only.
