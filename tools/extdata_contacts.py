#!/usr/bin/env python
"""
Contact sheets: the part of the audit that has to be looked at, not computed.

Three modes, each answering a different question that statistics cannot:

    overview    what IS this footage -- broadcast, amateur, training, stills?
    classcrops  what does class '0' / '1' / '2' / '3' actually depict?
    ball        is the ball box on the ball, and is it the right size?

Sampling is deterministic (seeded, evenly spaced through the sorted file list)
so a sheet can be regenerated and pointed at. Crops are drawn from the ORIGINAL
stored pixels and scaled up for viewing; the box is drawn before the upscale so
a loose box stays visibly loose.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / 'experiments' / 'external_data_audit'
SHEETS = AUDIT / 'contact_sheets'
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# BGR, one per class index; deliberately high-contrast against grass
COLORS = [(60, 60, 255), (0, 220, 255), (80, 255, 80), (255, 120, 0),
          (255, 0, 255), (200, 200, 200)]


def imread(p):
    import cv2
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(p, a):
    import cv2
    ok, buf = cv2.imencode('.jpg', a, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f'encode failed for {p}')
    buf.tofile(str(p))


def rows(lp: Path):
    """Yield (class_index, x1, y1, x2, y2) in normalised coords.

    Handles both YOLO detection rows (5 fields) and YOLO segmentation rows
    (class + polygon), converting a polygon to its axis-aligned bounds. S1
    mixes the two in the same export.
    """
    if not lp.exists():
        return
    for l in lp.read_text(encoding='utf-8', errors='replace').splitlines():
        f = l.split()
        if len(f) < 5:
            continue
        ci = int(float(f[0]))
        v = [float(x) for x in f[1:]]
        if len(v) == 4:
            cx, cy, w, h = v
            yield ci, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        elif len(v) >= 6 and len(v) % 2 == 0:
            xs, ys = v[0::2], v[1::2]
            yield ci, min(xs), min(ys), max(xs), max(ys)


def all_images(root: Path):
    for split in ('train', 'valid', 'test'):
        d = root / split / 'images'
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                yield split, p, root / split / 'labels' / f'{p.stem}.txt'


def spaced(seq, n):
    """Evenly spaced sample -- shows the dataset's span, not its first page."""
    if len(seq) <= n:
        return list(seq)
    idx = np.linspace(0, len(seq) - 1, n).round().astype(int)
    return [seq[i] for i in dict.fromkeys(idx.tolist())]


def grid(tiles, cols, pad=4):
    import cv2
    if not tiles:
        return None
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    r = (len(tiles) + cols - 1) // cols
    out = np.full((r * (th + pad) + pad, cols * (tw + pad) + pad, 3), 32, np.uint8)
    for i, t in enumerate(tiles):
        y = pad + (i // cols) * (th + pad)
        x = pad + (i % cols) * (tw + pad)
        out[y:y + t.shape[0], x:x + t.shape[1]] = t
    return out


def label(img, text, scale=0.45):
    import cv2
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, text[:60], (3, 13), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def mode_overview(sid, root, names, n, cols):
    import cv2
    items = list(all_images(root))
    tiles = []
    for split, ip, lp in spaced(items, n):
        img = imread(ip)
        if img is None:
            continue
        h, w = img.shape[:2]
        for ci, x1, y1, x2, y2 in rows(lp):
            c = COLORS[ci % len(COLORS)]
            cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), c, 2)
        t = cv2.resize(img, (420, int(420 * h / w)))
        tiles.append(label(t, f'{split}/{ip.name[:34]}'))
    g = grid(tiles, cols)
    if g is not None:
        p = SHEETS / f'{sid}_overview.jpg'
        imwrite(p, g)
        print(f'  {p.name}  {len(tiles)} images')


def mode_classcrops(sid, root, names, n, cols):
    import cv2
    per = defaultdict(list)
    for split, ip, lp in all_images(root):
        for ci, x1, y1, x2, y2 in rows(lp):
            per[ci].append((ip, x1, y1, x2, y2))
    for ci in sorted(per):
        tiles = []
        for ip, x1, y1, x2, y2 in spaced(per[ci], n):
            img = imread(ip)
            if img is None:
                continue
            h, w = img.shape[:2]
            bx1, by1, bx2, by2 = x1 * w, y1 * h, x2 * w, y2 * h
            m = 0.6 * max(bx2 - bx1, by2 - by1) + 6
            cx1, cy1 = int(max(0, bx1 - m)), int(max(0, by1 - m))
            cx2, cy2 = int(min(w, bx2 + m)), int(min(h, by2 + m))
            crop = img[cy1:cy2, cx1:cx2].copy()
            if crop.size == 0:
                continue
            cv2.rectangle(crop, (int(bx1 - cx1), int(by1 - cy1)),
                          (int(bx2 - cx1), int(by2 - cy1)),
                          COLORS[ci % len(COLORS)], 1)
            s = 150 / max(crop.shape[0], crop.shape[1])
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            tiles.append(crop)
        g = grid(tiles, cols)
        if g is not None:
            nm = str(names[ci]) if ci < len(names) else f'idx{ci}'
            safe = ''.join(ch if ch.isalnum() else '_' for ch in nm)
            p = SHEETS / f'{sid}_class{ci}_{safe}.jpg'
            imwrite(p, g)
            print(f'  {p.name}  class {ci} "{nm}"  {len(per[ci])} instances, {len(tiles)} shown')


def mode_ball(sid, root, names, ball_idx, n, cols):
    """Ball crops at high zoom: the only way to see a 7 px box is to enlarge it."""
    import cv2
    inst = []
    for split, ip, lp in all_images(root):
        for ci, x1, y1, x2, y2 in rows(lp):
            if ci == ball_idx:
                inst.append((ip, x1, y1, x2, y2))
    tiles = []
    for ip, x1, y1, x2, y2 in spaced(inst, n):
        img = imread(ip)
        if img is None:
            continue
        h, w = img.shape[:2]
        bx1, by1, bx2, by2 = x1 * w, y1 * h, x2 * w, y2 * h
        bw, bh = bx2 - bx1, by2 - by1
        m = max(2.0 * max(bw, bh), 20)
        cx1, cy1 = int(max(0, bx1 - m)), int(max(0, by1 - m))
        cx2, cy2 = int(min(w, bx2 + m)), int(min(h, by2 + m))
        crop = img[cy1:cy2, cx1:cx2].copy()
        if crop.size == 0:
            continue
        s = 180 / max(crop.shape[0], crop.shape[1])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(crop, (int((bx1 - cx1) * s), int((by1 - cy1) * s)),
                      (int((bx2 - cx1) * s), int((by2 - cy1) * s)), (60, 60, 255), 1)
        tiles.append(label(crop, f'{bw:.0f}x{bh:.0f}px', 0.4))
    g = grid(tiles, cols)
    if g is not None:
        p = SHEETS / f'{sid}_ball.jpg'
        imwrite(p, g)
        print(f'  {p.name}  {len(inst)} ball instances, {len(tiles)} shown')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['overview', 'classcrops', 'ball', 'all'],
                    default='all')
    ap.add_argument('--sources', default='S1,S2,S3,S4,S5,S6')
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--cols', type=int, default=6)
    args = ap.parse_args()

    SHEETS.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    srcs = json.loads((AUDIT / 'raw' / 'SOURCES.json').read_text(encoding='utf-8'))['sources']
    ballmap = json.loads((AUDIT / 'reports' / 'class_map.json').read_text(encoding='utf-8')) \
        if (AUDIT / 'reports' / 'class_map.json').exists() else {}

    for sid in args.sources.split(','):
        src = srcs[sid]
        root = REPO / src['extracted_to']
        names = src['declared_classes']
        print(f'{sid}  {src["workspace"]}/{src["project"]}  classes={names}')
        if args.mode in ('overview', 'all'):
            mode_overview(sid, root, names, args.n, args.cols)
        if args.mode in ('classcrops', 'all'):
            mode_classcrops(sid, root, names, args.n, args.cols)
        if args.mode in ('ball', 'all'):
            bi = None
            m = ballmap.get(sid, {}).get('mapping', {})
            for k, v in m.items():
                if v == 'ball':
                    bi = int(k)
            if bi is None:
                for i, nm in enumerate(names):
                    if str(nm).lower() in ('ball', 'ballon', 'football'):
                        bi = i
            if bi is not None:
                mode_ball(sid, root, names, bi, args.n, args.cols)


if __name__ == '__main__':
    main()
