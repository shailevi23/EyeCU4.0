# M3 — runtime / fresh-environment contract

## Runtime versions (this evaluation environment)

```
Python        3.10.0
PyTorch       2.8.0+cpu
Ultralytics   8.4.116
OpenCV        4.10.0
supervision   0.26.1
NumPy         2.2.6
device        CPU (torch.cuda.is_available() == False)
```

Not every pip package is frozen here — only the ones that can affect
detector/tracker numerical output, per instruction. Full dependency
declarations (with minimum versions) are in `requirements.txt`, already
present and not modified for this freeze:
`ultralytics>=8.3.0`, `supervision>=0.20.0`, `torch>=2.0.0`,
`torchvision>=0.15.0`, `opencv-python>=4.8.0`, `numpy>=1.24.0`,
`scipy>=1.10.0`, `pandas>=2.0.0`, `scikit-learn>=1.3.0`.

## Deterministic flags required for final TEST evaluation

No `torch.manual_seed` / `cudnn.deterministic` call exists anywhere in the
codebase today, and none was needed to resolve the CBIoU reproducibility
blocker (see `CBIOU_REPRODUCIBILITY.md` — the root cause was a design defect
in evaluation scripts sharing tracker state across sequences, not RNG or
GPU-kernel nondeterminism). Device is already CPU in this environment,
so classic CUDA nondeterminism was never a factor.

As a defensive margin only (pre-approved by instruction as a "deterministic
backend setting if practical and necessary," not required by any evidence
gathered in this milestone), M4/M5 TEST evaluation **must** run with:

```
torch.set_num_threads(1)
cv2.setNumThreads(1)
```

set once at process start, before any model is loaded. This changes
execution parallelism only — no algorithm, no threshold, no weight. It is
not baked into the shared library defaults (would needlessly slow down
ordinary development use); it is a requirement recorded in the freeze
manifest for the specific frozen TEST run.

## Fresh-environment path (clean checkout → first production run)

1. Clone the frozen commit / working tree (see `SYSTEM_FREEZE_MANIFEST.json`
   for the exact source-tree identity).
2. `pip install -r requirements.txt`.
3. Provision the third-party SN3D ball checkpoint: it is **not** in git
   (`.gitignore:242`, `models/third_party/**/*.pt`). Download
   `yolo-sn-ball.pt` from the official SoccerNet-v3D release v1.0.0
   (`https://github.com/mguti97/SoccerNet-v3D`) and place it at
   `models/third_party/soccernet_v3d/yolo-sn-ball.pt`, or set
   `EYECU_SN3D_MODEL_PATH` to wherever it was placed. Full instructions:
   `models/third_party/soccernet_v3d/README.md`.
4. `best_A_960.pt` (the human detector) **is** tracked in git at the repo
   root and needs no separate provisioning step.
5. Run the production entry point, e.g.:
   ```
   python run_pipeline.py --video "<path>" --output-dir <dir>
   ```
   No hidden experimental configuration is required — the CLI/constructor
   defaults, as of this freeze, already point at the correct closed
   checkpoints and backends (see `PRODUCTION_CONFIG_VERIFICATION.md`).

A bare `git clone` alone is deliberately **not** sufficient to run the ball
branch — the missing 51 MB third-party weight fails fast with an actionable
error (path, official-release URL, expected SHA256) rather than silently
falling back to a different detector. This is correct, existing behaviour
and was not changed for this freeze.

## Exact production command template

```
python run_pipeline.py \
  --video "<path-to-non-TEST-or-TEST-video>" \
  --output-dir "<output-dir>" \
  --skip-frames 2
```

(`--yolo-model`, `--imgsz`, `--tracker`, and the ball branch all take their
corrected production defaults; only `--video`/`--output-dir` and, for TEST,
`--skip-frames`/frame-range handling per the M4 plan need to be supplied.)
