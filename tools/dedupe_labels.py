#!/usr/bin/env python
"""
Remove duplicate boxes from YOLO label files.

Two boxes on one object is the specific failure this project exists to fix, so
it must not survive into the training data. Ultralytics silently drops exact
repeated rows at load time but keeps near-duplicates, which actively teach the
detector that double-boxing is correct.

Removes:
  * exact repeated rows
  * same-class pairs overlapping at IoU >= --iou (default 0.90)

Never removes:
  * boxes of different classes, however much they overlap
  * genuinely overlapping players -- two adjacent players sit far below 0.90 IoU

When a near-duplicate pair is found the LARGER box is kept: an over-tight box
crops off part of the player, which is the more damaging error.

Examples:
    python tools/dedupe_labels.py --dry-run
    python tools/dedupe_labels.py
    python tools/dedupe_labels.py --iou 0.85 --dry-run
"""

import argparse
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

CLASSES = ['player', 'goalkeeper', 'referee', 'ball']


def iou(a, b):
    """IoU of two YOLO boxes given as (cx, cy, w, h)."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def dedupe(lines, iou_thresh, stats):
    """Return (kept_lines, removed_descriptions)."""
    parsed, kept_rows, removed = [], [], []

    seen = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line in seen:
            removed.append(('exact', line.split()[0]))
            stats['exact'] += 1
            stats[f'exact_{CLASSES[int(float(line.split()[0]))]}'] += 1
            continue
        seen.add(line)
        parts = line.split()
        parsed.append((int(float(parts[0])), tuple(float(v) for v in parts[1:]), line))

    # Largest first, so the box that survives a near-duplicate pair is the
    # more generous one.
    order = sorted(range(len(parsed)), key=lambda i: -(parsed[i][1][2] * parsed[i][1][3]))
    dropped = set()
    for pos, i in enumerate(order):
        if i in dropped:
            continue
        for j in order[pos + 1:]:
            if j in dropped or parsed[i][0] != parsed[j][0]:
                continue
            overlap = iou(parsed[i][1], parsed[j][1])
            if overlap >= iou_thresh:
                dropped.add(j)
                cls = CLASSES[parsed[j][0]]
                removed.append(('near', f'{cls} IoU {overlap:.2f}'))
                stats['near'] += 1
                stats[f'near_{cls}'] += 1

    for i, (_, _, line) in enumerate(parsed):
        if i not in dropped:
            kept_rows.append(line)
    return kept_rows, removed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labels', default='data/labels')
    p.add_argument('--iou', type=float, default=0.90,
                   help='Same-class IoU at or above which boxes are duplicates.')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--no-backup', dest='backup', action='store_false', default=True)
    args = p.parse_args()

    labels_root = Path(args.labels)
    if not labels_root.exists():
        sys.exit(f'Labels not found: {labels_root}')

    files = sorted(labels_root.rglob('*.txt'))
    if not files:
        sys.exit(f'No label files under {labels_root}')

    stats = Counter()
    changes = []
    for lbl in files:
        lines = lbl.read_text(encoding='utf-8').splitlines()
        kept, removed = dedupe(lines, args.iou, stats)
        if removed:
            changes.append((lbl, kept, removed))

    before = sum(1 for f in files
                 for l in f.read_text(encoding='utf-8').splitlines() if l.strip())
    total_removed = stats['exact'] + stats['near']

    print(f'scanned {len(files)} label files, {before} boxes')
    print(f'files with duplicates : {len(changes)}')
    print(f'exact duplicate rows  : {stats["exact"]}')
    print(f'near-duplicates (>={args.iou}): {stats["near"]}')
    print()
    for cls in CLASSES:
        e, n = stats.get(f'exact_{cls}', 0), stats.get(f'near_{cls}', 0)
        if e or n:
            print(f'  {cls:<11} exact={e}  near={n}')

    if changes:
        print('\naffected files:')
        for lbl, _, removed in changes[:25]:
            what = ', '.join(d for _, d in removed)
            print(f'  {lbl.relative_to(labels_root).as_posix()}: {what}')
        if len(changes) > 25:
            print(f'  ... and {len(changes) - 25} more')

    if args.dry_run:
        print('\n--dry-run: nothing written.')
        return
    if not changes:
        print('\nNothing to remove.')
        return

    if args.backup:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dest = Path('data/backups') / f'labels_before_dedupe_{stamp}'
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(labels_root, dest)
        print(f'\nbacked up -> {dest}')

    for lbl, kept, _ in changes:
        lbl.write_text('\n'.join(kept) + ('\n' if kept else ''), encoding='utf-8')

    after = sum(1 for f in files
                for l in f.read_text(encoding='utf-8').splitlines() if l.strip())
    print(f'\nremoved {total_removed} box(es): {before} -> {after}')
    print('\nNext: python tools/validate_annotations.py --strict')


if __name__ == '__main__':
    main()
