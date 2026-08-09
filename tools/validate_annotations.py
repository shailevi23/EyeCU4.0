#!/usr/bin/env python
"""
Validate YOLO annotations before training.

Ultralytics silently skips images it cannot read and quietly ignores malformed
label lines, so a broken dataset trains to a poor score with no error. This
checks the things that fail that way:

  * images that will not decode
  * missing / empty label files
  * malformed label lines (wrong field count, non-numeric)
  * class ids outside 0..3
  * coordinates outside [0, 1] or with non-positive width/height
  * suspiciously tiny boxes (usually a mis-drawn annotation)
  * labels with no matching image

Works on either layout:
  data/frames + data/labels        (pre-split, mirrored subdirectories)
  data/dataset                     (Ultralytics images/<split>, labels/<split>)

Examples:
    python tools/validate_annotations.py --frames data/frames --labels data/labels
    python tools/validate_annotations.py --dataset data/dataset --strict
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASSES = ['player', 'goalkeeper', 'referee', 'ball']
IMG_EXTS = {'.jpg', '.jpeg', '.png'}
SPLITS = ('train', 'val', 'test')

# Below this relative area a box is almost certainly a mis-click rather than a
# real object -- though a distant ball is legitimately tiny, so this warns
# rather than fails.
TINY_AREA = 1e-6


def imread_unicode(path: Path):
    """cv2.imread fails on non-ASCII paths on Windows; read bytes instead."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def pairs_from_split_layout(dataset_root: Path):
    for split in SPLITS:
        img_dir = dataset_root / 'images' / split
        lbl_dir = dataset_root / 'labels' / split
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() in IMG_EXTS:
                yield img, lbl_dir / f'{img.stem}.txt', split
        # labels with no image
        if lbl_dir.is_dir():
            stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
            for lbl in sorted(lbl_dir.glob('*.txt')):
                if lbl.stem not in stems:
                    yield None, lbl, split


def pairs_from_mirror_layout(frames_root: Path, labels_root: Path):
    for img in sorted(p for p in frames_root.rglob('*') if p.suffix.lower() in IMG_EXTS):
        rel = img.relative_to(frames_root)
        yield img, (labels_root / rel).with_suffix('.txt'), rel.parts[0]
    if labels_root.is_dir():
        for lbl in sorted(labels_root.rglob('*.txt')):
            rel = lbl.relative_to(labels_root)
            if not any((frames_root / rel).with_suffix(ext).exists() for ext in IMG_EXTS):
                yield None, lbl, rel.parts[0] if len(rel.parts) > 1 else 'root'


def iou(a, b):
    """IoU of two YOLO boxes given as (cx, cy, w, h)."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix, iy = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def check_label(lbl: Path, dup_iou: float = 0.90):
    """Return (per-class Counter, list of error strings)."""
    counts, errors = Counter(), []
    seen_rows = set()
    boxes = []          # (class_id, cx, cy, w, h) for the duplicate scan
    try:
        text = lbl.read_text(encoding='utf-8')
    except OSError as e:
        return counts, [f'unreadable label ({e})']

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f'line {lineno}: expected 5 fields, got {len(parts)}')
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f'line {lineno}: non-numeric value')
            continue

        if not 0 <= cid < len(CLASSES):
            errors.append(f'line {lineno}: class id {cid} outside 0..{len(CLASSES) - 1}')
            continue
        counts[CLASSES[cid]] += 1

        if w <= 0 or h <= 0:
            errors.append(f'line {lineno}: non-positive box {w:.4f}x{h:.4f}')
        for name, v in (('cx', cx), ('cy', cy), ('w', w), ('h', h)):
            if not 0.0 <= v <= 1.0:
                errors.append(f'line {lineno}: {name}={v:.4f} outside [0,1] '
                              f'(labels must be normalised)')
        if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 or \
           cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
            errors.append(f'line {lineno}: box extends past the image edge')

        # Two boxes on one object is the failure this project exists to fix, so
        # it must not survive into the training data. Ultralytics silently drops
        # exact repeats; near-duplicates it keeps, and they teach the detector
        # that double-boxing is correct.
        if line in seen_rows:
            errors.append(f'line {lineno}: exact duplicate of an earlier box')
        seen_rows.add(line)
        boxes.append((cid, cx, cy, w, h))

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][0] != boxes[j][0]:
                continue                      # different classes may overlap
            overlap = iou(boxes[i][1:], boxes[j][1:])
            if overlap >= dup_iou:
                errors.append(
                    f'near-duplicate {CLASSES[boxes[i][0]]} boxes '
                    f'(IoU {overlap:.2f}) at #{i + 1} and #{j + 1}')
    return counts, errors


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', help='Split dataset root (images/<split>).')
    p.add_argument('--frames', default='data/frames')
    p.add_argument('--labels', default='data/labels')
    p.add_argument('--check-images', action='store_true', default=True,
                   help='Decode every image (default on).')
    p.add_argument('--no-check-images', dest='check_images', action='store_false')
    p.add_argument('--warn-tiny', action='store_true',
                   help='Also report suspiciously tiny boxes.')
    p.add_argument('--strict', action='store_true',
                   help='Exit non-zero on any error (use in CI).')
    args = p.parse_args()

    if args.dataset:
        root = Path(args.dataset)
        if not root.exists():
            sys.exit(f'Dataset not found: {root}')
        pairs = list(pairs_from_split_layout(root))
        where = str(root)
    else:
        frames_root, labels_root = Path(args.frames), Path(args.labels)
        if not frames_root.exists():
            sys.exit(f'Frames not found: {frames_root}')
        pairs = list(pairs_from_mirror_layout(frames_root, labels_root))
        where = f'{frames_root} + {labels_root}'

    if not pairs:
        sys.exit(f'Nothing to validate under {where}')

    errors, warnings = [], []
    counts = Counter()
    per_group = Counter()
    n_images = n_labelled = n_empty = 0

    for img, lbl, group in pairs:
        if img is None:
            errors.append(f'ORPHAN LABEL (no image): {lbl}')
            continue

        n_images += 1
        per_group[group] += 1

        if args.check_images and imread_unicode(img) is None:
            errors.append(f'CORRUPT IMAGE: {img}')
            continue

        if not lbl.exists():
            errors.append(f'MISSING LABEL: {img}')
            continue

        c, errs = check_label(lbl)
        for e in errs:
            errors.append(f'{lbl.name}: {e}')
        counts.update(c)
        if sum(c.values()) == 0 and not errs:
            n_empty += 1          # legitimate hard negative
        else:
            n_labelled += 1

    print(f'\nValidated {n_images} images under {where}')
    print(f'  labelled          : {n_labelled}')
    print(f'  empty (hard negs) : {n_empty}')
    print(f'  errors            : {len(errors)}')

    if counts:
        print('\ninstances per class:')
        total = sum(counts.values())
        for c in CLASSES:
            n = counts.get(c, 0)
            print(f'  {c:<11}{n:>8}{(n / total if total else 0):>8.1%}')
        for rare in ('goalkeeper', 'referee', 'ball'):
            if counts.get(rare, 0) < 100:
                warnings.append(f'only {counts.get(rare, 0)} `{rare}` instances; '
                                f'this class will train poorly')

    if errors:
        print(f'\n{len(errors)} error(s):')
        for e in errors[:40]:
            print(f'  ! {e}')
        if len(errors) > 40:
            print(f'  ... and {len(errors) - 40} more')

    if warnings:
        print('\nwarnings:')
        for w in warnings:
            print(f'  ~ {w}')

    if not errors:
        print('\nNo errors.')

    if args.strict and errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
