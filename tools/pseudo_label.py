#!/usr/bin/env python
"""
Generate DRAFT YOLO labels with the hosted Roboflow football detector.

These are drafts, not truth. Every draft must be reviewed by a human before it
trains anything -- see LABELING.md.

Safety properties:
  * A label file the tool did not write is never touched.
  * A draft the tool wrote and a human then edited is never touched again
    (detected by content hash, not timestamps).
  * Per-box confidence and provenance are stored SEPARATELY from the labels,
    so the label files stay clean YOLO format.

Classes are fixed project-wide:
    0 player   1 goalkeeper   2 referee   3 ball

The API key is read only from ROBOFLOW_API_KEY. Nothing is hardcoded and no
key is ever written to disk.

Examples:
    export ROBOFLOW_API_KEY=...            # PowerShell: $env:ROBOFLOW_API_KEY="..."
    python tools/pseudo_label.py --dry-run
    python tools/pseudo_label.py --source youth_3 --limit 20
    python tools/pseudo_label.py --batch data/batches/batch_01.json
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

CLASSES = ['player', 'goalkeeper', 'referee', 'ball']
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

ALIASES = {
    'player': 'player', 'players': 'player', 'football player': 'player',
    'outfield player': 'player', 'person': 'player',
    'goalkeeper': 'goalkeeper', 'goal keeper': 'goalkeeper', 'gk': 'goalkeeper',
    'referee': 'referee', 'ref': 'referee', 'main referee': 'referee',
    'side referee': 'referee', 'assistant referee': 'referee',
    'ball': 'ball', 'sports ball': 'ball', 'football': 'ball', 'soccer ball': 'ball',
}

ROBOFLOW_URL = 'https://detect.roboflow.com'
DEFAULT_MODEL_ID = os.environ.get('ROBOFLOW_MODEL_ID',
                                  'football-players-detection-3zvbc/12')


def norm_class(name):
    return ALIASES.get(str(name).strip().lower())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def imread_unicode(path: Path):
    """cv2.imread fails on non-ASCII paths on Windows; read the bytes instead."""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def to_yolo(cls_name, cx, cy, w, h, img_w, img_h):
    """Absolute centre-format box -> normalised YOLO line, clamped to the frame."""
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(float(img_w), cx + w / 2), min(float(img_h), cy + h / 2)
    if x2 <= x1 or y2 <= y1:
        return None, None
    ncx, ncy = ((x1 + x2) / 2) / img_w, ((y1 + y2) / 2) / img_h
    nw, nh = (x2 - x1) / img_w, (y2 - y1) / img_h
    if nw <= 0 or nh <= 0:
        return None, None
    return (f'{CLASS_ID[cls_name]} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}',
            {'class': cls_name, 'cx': round(ncx, 6), 'cy': round(ncy, 6),
             'w': round(nw, 6), 'h': round(nh, 6)})


class RoboflowBackend:
    def __init__(self, model_id, confidence, overlap, retries=3, timeout=30):
        try:
            import requests
        except ImportError:
            sys.exit('The requests package is required: pip install requests')
        self._requests = requests

        key = os.environ.get('ROBOFLOW_API_KEY')
        if not key:
            sys.exit(
                'ROBOFLOW_API_KEY is not set.\n'
                '  PowerShell : $env:ROBOFLOW_API_KEY = "your-key"\n'
                '  bash       : export ROBOFLOW_API_KEY="your-key"\n'
                'Get a key at https://app.roboflow.com/settings/api\n'
                'The key is read from the environment only and is never stored.'
            )
        self.api_key = key
        self.model_id = model_id
        self.url = f'{ROBOFLOW_URL}/{model_id}'
        self.params = {'api_key': key,
                       'confidence': int(confidence * 100),
                       'overlap': int(overlap * 100),
                       'format': 'json'}
        self.retries = retries
        self.timeout = timeout

    def predict(self, image_path: Path):
        """Returns a list of (class, cx, cy, w, h, conf) or None on failure."""
        payload = base64.b64encode(image_path.read_bytes())
        last = None
        for attempt in range(self.retries):
            try:
                r = self._requests.post(
                    self.url, params=self.params, data=payload,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'},
                    timeout=self.timeout)
                if r.status_code in (401, 403):
                    sys.exit(f'Roboflow rejected the API key (HTTP {r.status_code}). '
                             f'Check ROBOFLOW_API_KEY and that it can access '
                             f'{self.model_id}.')
                if r.status_code == 404:
                    sys.exit(f'Roboflow model not found: {self.model_id}. '
                             f'Set --model-id or ROBOFLOW_MODEL_ID.')
                r.raise_for_status()
                out = []
                for p in r.json().get('predictions', []):
                    name = norm_class(p.get('class', ''))
                    if name:
                        out.append((name, p['x'], p['y'], p['width'], p['height'],
                                    float(p.get('confidence', 0.0))))
                return out
            except SystemExit:
                raise
            except Exception as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        print(f'  ! failed on {image_path.name}: {last}')
        return None


def load_targets(args):
    """Images to consider, from an explicit batch file or the frame manifest."""
    frames_root = Path(args.frames)

    if args.batch:
        batch = json.loads(Path(args.batch).read_text(encoding='utf-8'))
        rels = batch['images'] if isinstance(batch, dict) else batch
        return [(frames_root / r, r) for r in rels]

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(f'Manifest not found: {manifest_path}\n'
                 f'Run: python tools/dataset_manifest.py')
    data = json.loads(manifest_path.read_text(encoding='utf-8'))

    rels = []
    if 'images' in data:                      # data/manifest.json (per-image)
        rels = [row['image_path'] for row in data['images']]
    else:                                     # data/frames/manifest.json
        for m in data.get('matches', []):
            src = frames_root / m['match_id']
            rels += [p.relative_to(frames_root).as_posix()
                     for p in sorted(src.glob('*.jpg'))]

    if args.source:
        wanted = set(args.source)
        rels = [r for r in rels if r.split('/')[0] in wanted]
    return [(frames_root / r, r) for r in rels]


def classify_existing(label_path: Path, meta_path: Path):
    """
    'absent' | 'draft' (ours, untouched) | 'edited' (ours, human-modified)
    | 'foreign' (not written by this tool).
    """
    if not label_path.exists():
        return 'absent'
    if not meta_path.exists():
        return 'foreign'
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return 'foreign'
    current = sha256_text(label_path.read_text(encoding='utf-8'))
    return 'draft' if current == meta.get('label_sha256') else 'edited'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frames', default='data/frames')
    p.add_argument('--labels', default='data/labels',
                   help='YOLO .txt output (drafts, later human-corrected).')
    p.add_argument('--meta', default='data/pseudo_meta',
                   help='Confidence/provenance, kept out of the label files.')
    p.add_argument('--manifest', default='data/manifest.json')
    p.add_argument('--batch', help='JSON list of image paths (from select_batch.py).')
    p.add_argument('--source', action='append',
                   help='Restrict to this source match (repeatable).')
    p.add_argument('--limit', type=int, help='Stop after N images.')
    p.add_argument('--confidence', type=float, default=0.30,
                   help='Minimum detection confidence (default 0.30).')
    p.add_argument('--overlap', type=float, default=0.50,
                   help='NMS IoU threshold (default 0.50).')
    p.add_argument('--model-id', default=DEFAULT_MODEL_ID)
    p.add_argument('--dry-run', action='store_true',
                   help='Show what would be labelled; no API calls, no writes.')
    p.add_argument('--refresh-drafts', action='store_true',
                   help='Re-generate drafts that are still untouched. '
                        'Human-edited and foreign labels are still never touched.')
    args = p.parse_args()

    frames_root = Path(args.frames)
    labels_root = Path(args.labels)
    meta_root = Path(args.meta)

    targets = load_targets(args)
    if not targets:
        sys.exit('No images selected.')

    # Decide what to do with each target before touching the network.
    todo, states = [], Counter()
    for img, rel in targets:
        label_path = (labels_root / rel).with_suffix('.txt')
        meta_path = (meta_root / rel).with_suffix('.json')
        state = classify_existing(label_path, meta_path)
        states[state] += 1
        if state == 'absent' or (state == 'draft' and args.refresh_drafts):
            todo.append((img, rel, label_path, meta_path))
    if args.limit:
        todo = todo[:args.limit]

    print(f'{len(targets)} image(s) considered')
    print(f'  absent (will draft) : {states["absent"]}')
    print(f'  existing drafts     : {states["draft"]}'
          f'{"  (will refresh)" if args.refresh_drafts else "  (skipped)"}')
    print(f'  human-edited        : {states["edited"]}  (never touched)')
    print(f'  foreign labels      : {states["foreign"]}  (never touched)')
    print(f'  -> {len(todo)} to send to Roboflow')

    if args.dry_run:
        print('\n--dry-run: no API calls, nothing written.')
        for _, rel, _, _ in todo[:15]:
            print(f'    {rel}')
        if len(todo) > 15:
            print(f'    ... and {len(todo) - 15} more')
        return

    if not todo:
        print('\nNothing to do.')
        return

    backend = RoboflowBackend(args.model_id, args.confidence, args.overlap)
    print(f'\nBackend: roboflow ({args.model_id}) conf>={args.confidence}\n')

    counts, per_source = Counter(), Counter()
    written = failed = 0
    conf_by_class = {c: [] for c in CLASSES}

    for i, (img, rel, label_path, meta_path) in enumerate(todo, 1):
        preds = backend.predict(img)
        if preds is None:
            failed += 1
            continue

        image = imread_unicode(img)
        if image is None:
            print(f'  ! unreadable image: {rel}')
            failed += 1
            continue
        img_h, img_w = image.shape[:2]

        lines, boxes = [], []
        for name, cx, cy, w, h, conf in preds:
            line, box = to_yolo(name, cx, cy, w, h, img_w, img_h)
            if line:
                lines.append(line)
                box['confidence'] = round(conf, 4)
                boxes.append(box)
                counts[name] += 1
                conf_by_class[name].append(conf)

        # An empty file is a valid YOLO label: it marks a hard negative.
        text = '\n'.join(lines) + ('\n' if lines else '')
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(text, encoding='utf-8')

        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({
            'image': rel,
            'source': rel.split('/')[0],
            'model_id': args.model_id,
            'confidence_threshold': args.confidence,
            'overlap': args.overlap,
            'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'image_size': [img_w, img_h],
            'status': 'draft',
            'label_sha256': sha256_text(text),
            'boxes': boxes,
        }, indent=2), encoding='utf-8')

        written += 1
        per_source[rel.split('/')[0]] += 1
        if i % 25 == 0 or i == len(todo):
            print(f'  {i}/{len(todo)}  (written {written}, failed {failed})')

    summary = {
        'model_id': args.model_id,
        'confidence_threshold': args.confidence,
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'frames_written': written,
        'frames_failed': failed,
        'instances_per_class': {c: counts.get(c, 0) for c in CLASSES},
        'mean_confidence_per_class': {
            c: round(float(np.mean(v)), 4) if v else None
            for c, v in conf_by_class.items()},
        'frames_per_source': dict(per_source),
    }
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / 'pseudo_label_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')

    print('\ninstances per class (DRAFT, needs review):')
    for c in CLASSES:
        mean = summary['mean_confidence_per_class'][c]
        print(f'  {c:<11}{counts.get(c, 0):>7}   mean conf '
              f'{mean if mean is not None else "-"}')
    print(f'\nlabels : {labels_root}')
    print(f'meta   : {meta_root}')

    for rare in ('goalkeeper', 'referee', 'ball'):
        if counts.get(rare, 0) == 0:
            print(f'! the model produced no `{rare}` at all -- check the model id '
                  f'covers all four classes.')

    print('\nNEXT: these are DRAFTS. Review every goalkeeper / referee / ball box '
          'by hand (LABELING.md), then run:\n'
          '  python tools/validate_annotations.py --strict')


if __name__ == '__main__':
    main()
