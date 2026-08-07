#!/usr/bin/env python
"""
Generate first-pass YOLO labels for extracted frames.

These are PSEUDO-LABELS, not final labels. Import the frames + labels into
Roboflow Annotate or CVAT and manually verify every `goalkeeper`, `referee`
and `ball` box before training (ROADMAP.md section 3).

Backends:
  roboflow  hosted football detector (keeps the goalkeeper class).
            Requires ROBOFLOW_API_KEY in the environment.
  local     any local Ultralytics .pt model. A COCO model only produces
            player/ball (person -> player, sports ball -> ball); a fine-tuned
            football model produces all four classes.

Examples:
    $env:ROBOFLOW_API_KEY = "..."
    python tools/pseudo_label.py --frames data/frames --backend roboflow
    python tools/pseudo_label.py --frames data/frames --backend local --model yolov8x.pt
"""

import argparse
import base64
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

# Class order is fixed across the whole project -- see data/football.yaml.
CLASSES = ['player', 'goalkeeper', 'referee', 'ball']
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

# Alias table for whatever a source model happens to call things.
ALIASES = {
    'player': 'player', 'players': 'player', 'football player': 'player',
    'person': 'player', 'outfield player': 'player',
    'goalkeeper': 'goalkeeper', 'goal keeper': 'goalkeeper', 'gk': 'goalkeeper',
    'referee': 'referee', 'ref': 'referee', 'main referee': 'referee',
    'side referee': 'referee', 'assistant referee': 'referee',
    'ball': 'ball', 'sports ball': 'ball', 'football': 'ball', 'soccer ball': 'ball',
}

ROBOFLOW_URL = 'https://detect.roboflow.com'
DEFAULT_MODEL_ID = os.environ.get('ROBOFLOW_MODEL_ID', 'football-players-detection-3zvbc/12')


def norm_class(name: str):
    return ALIASES.get(str(name).strip().lower())


def to_yolo_line(cls_name: str, cx, cy, w, h, img_w, img_h):
    """Absolute centre-format box -> normalised YOLO line, clamped to the frame."""
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(float(img_w), cx + w / 2), min(float(img_h), cy + h / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    ncx = ((x1 + x2) / 2) / img_w
    ncy = ((y1 + y2) / 2) / img_h
    nw = (x2 - x1) / img_w
    nh = (y2 - y1) / img_h
    if nw <= 0 or nh <= 0:
        return None
    return f'{CLASS_ID[cls_name]} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}'


class RoboflowBackend:
    def __init__(self, model_id, confidence, overlap, retries=3):
        import requests  # imported lazily so the local backend has no dependency
        self.requests = requests
        self.api_key = os.environ.get('ROBOFLOW_API_KEY')
        if not self.api_key:
            sys.exit('ROBOFLOW_API_KEY is not set. Export it, or use --backend local.')
        self.url = f'{ROBOFLOW_URL}/{model_id}'
        self.params = {
            'api_key': self.api_key,
            'confidence': int(confidence * 100),
            'overlap': int(overlap * 100),
            'format': 'json',
        }
        self.retries = retries

    def predict(self, image_path: Path):
        payload = base64.b64encode(image_path.read_bytes())
        last = None
        for attempt in range(self.retries):
            try:
                r = self.requests.post(
                    self.url, params=self.params, data=payload,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'},
                    timeout=30)
                r.raise_for_status()
                out = []
                for pred in r.json().get('predictions', []):
                    name = norm_class(pred.get('class', ''))
                    if name:
                        out.append((name, pred['x'], pred['y'],
                                    pred['width'], pred['height']))
                return out
            except Exception as e:  # network flakiness is expected, see RESULTS.md
                last = e
                time.sleep(1.5 * (attempt + 1))
        print(f'  ! roboflow failed on {image_path.name}: {last}')
        return None


class LocalBackend:
    def __init__(self, model_path, confidence, overlap, imgsz):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = confidence
        self.iou = overlap
        self.imgsz = imgsz
        mapped = {i: norm_class(n) for i, n in self.model.names.items()}
        self.mapped = {i: n for i, n in mapped.items() if n}
        if not self.mapped:
            sys.exit(f'{model_path} has no classes that map to {CLASSES}.')
        print(f'  local model classes in use: '
              f'{sorted(set(self.mapped.values()))}')

    def predict(self, image_path: Path):
        res = self.model.predict(str(image_path), conf=self.conf, iou=self.iou,
                                 imgsz=self.imgsz, verbose=False)[0]
        out = []
        for box in res.boxes:
            name = self.mapped.get(int(box.cls))
            if not name:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append((name, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frames', default='data/frames',
                   help='Frames root produced by extract_frames.py.')
    p.add_argument('--labels', default='data/labels',
                   help='Output root for YOLO .txt labels (mirrors --frames).')
    p.add_argument('--backend', choices=['roboflow', 'local'], default='roboflow')
    p.add_argument('--model', default='yolov8x.pt', help='Local backend model path.')
    p.add_argument('--model-id', default=DEFAULT_MODEL_ID,
                   help='Roboflow model id, e.g. "project/version".')
    p.add_argument('--confidence', type=float, default=0.30)
    p.add_argument('--overlap', type=float, default=0.50, help='NMS IoU threshold.')
    p.add_argument('--imgsz', type=int, default=1280, help='Local backend inference size.')
    p.add_argument('--limit', type=int, help='Only label the first N frames (smoke test).')
    p.add_argument('--overwrite', action='store_true',
                   help='Re-label frames that already have a .txt file.')
    args = p.parse_args()

    frames_root = Path(args.frames)
    labels_root = Path(args.labels)
    if not frames_root.exists():
        sys.exit(f'Frames directory not found: {frames_root}. Run extract_frames.py first.')

    images = sorted(pth for pth in frames_root.rglob('*')
                    if pth.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if args.limit:
        images = images[:args.limit]
    if not images:
        sys.exit(f'No images under {frames_root}.')

    if args.backend == 'roboflow':
        backend = RoboflowBackend(args.model_id, args.confidence, args.overlap)
        print(f'Backend: roboflow ({args.model_id})')
    else:
        backend = LocalBackend(args.model, args.confidence, args.overlap, args.imgsz)
        print(f'Backend: local ({args.model})')

    counts = Counter()
    per_match = Counter()
    labelled = failed = skipped = 0

    for i, img_path in enumerate(images, 1):
        rel = img_path.relative_to(frames_root)
        out_path = (labels_root / rel).with_suffix('.txt')
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        preds = backend.predict(img_path)
        if preds is None:
            failed += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f'  ! unreadable image: {img_path}')
            failed += 1
            continue
        img_h, img_w = image.shape[:2]

        lines = []
        for name, cx, cy, w, h in preds:
            line = to_yolo_line(name, cx, cy, w, h, img_w, img_h)
            if line:
                lines.append(line)
                counts[name] += 1

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # An empty file is a valid YOLO label: it marks a hard negative frame.
        out_path.write_text('\n'.join(lines) + ('\n' if lines else ''))
        per_match[rel.parts[0] if len(rel.parts) > 1 else 'root'] += 1
        labelled += 1

        if i % 25 == 0 or i == len(images):
            print(f'  {i}/{len(images)} frames  '
                  f'(labelled {labelled}, skipped {skipped}, failed {failed})')

    summary = {
        'backend': args.backend,
        'source': args.model_id if args.backend == 'roboflow' else args.model,
        'confidence': args.confidence,
        'frames_labelled': labelled,
        'frames_skipped': skipped,
        'frames_failed': failed,
        'instances_per_class': {c: counts.get(c, 0) for c in CLASSES},
        'frames_per_match': dict(per_match),
    }
    labels_root.mkdir(parents=True, exist_ok=True)
    (labels_root / 'pseudo_label_summary.json').write_text(json.dumps(summary, indent=2))

    print('\nInstances per class:')
    for c in CLASSES:
        print(f'  {c:<11} {counts.get(c, 0)}')
    print(f'\nLabels written to {labels_root}')
    for rare in ('goalkeeper', 'referee', 'ball'):
        if counts.get(rare, 0) < 100:
            print(f'! only {counts.get(rare, 0)} `{rare}` instances -- '
                  f'add frames that contain them before training.')
    print('\nNEXT: import frames + labels into Roboflow/CVAT and correct every '
          'goalkeeper / referee / ball box by hand. Do not train on raw output.')


if __name__ == '__main__':
    main()
