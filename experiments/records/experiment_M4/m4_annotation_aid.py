#!/usr/bin/env python
"""
M4 annotation aid -- pure image-processing helper (no model inference) that
renders a magnified, pixel-gridded version of a candidate frame so the
human annotator (the assistant) can read approximate box coordinates
directly off gridlines, and crop/zoom sub-regions for small objects
(chiefly the ball). This is scaffolding for the annotator's own eyes; it
proposes nothing and labels nothing itself.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path('experiments/records/experiment_M4/annotation_aid')


def grid(path, scale=2, step=40, out_name=None):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    big = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    for x in range(0, w + 1, step):
        cv2.line(big, (x * scale, 0), (x * scale, h * scale), (0, 255, 0), 1)
        cv2.putText(big, str(x), (x * scale + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for y in range(0, h + 1, step):
        cv2.line(big, (0, y * scale), (w * scale, y * scale), (0, 255, 0), 1)
        cv2.putText(big, str(y), (2, y * scale + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / (out_name or (Path(path).stem + f'_grid_s{scale}_g{step}.png'))
    cv2.imwrite(str(out), big)
    print(out, big.shape[1], big.shape[0])
    return out


def crop(path, x1, y1, x2, y2, scale=4, out_name=None):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    sub = img[y1:y2, x1:x2]
    big = cv2.resize(sub, ((x2 - x1) * scale, (y2 - y1) * scale), interpolation=cv2.INTER_CUBIC)
    step = 10
    for x in range(0, x2 - x1 + 1, step):
        cv2.line(big, (x * scale, 0), (x * scale, big.shape[0]), (0, 255, 0), 1)
        cv2.putText(big, str(x + x1), (x * scale + 1, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    for y in range(0, y2 - y1 + 1, step):
        cv2.line(big, (0, y * scale), (big.shape[1], y * scale), (0, 255, 0), 1)
        cv2.putText(big, str(y + y1), (1, y * scale + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / (out_name or f'{Path(path).stem}_crop_{x1}_{y1}_{x2}_{y2}.png')
    cv2.imwrite(str(out), big)
    print(out, big.shape[1], big.shape[0])
    return out


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'grid':
        grid(sys.argv[2], scale=int(sys.argv[3]) if len(sys.argv) > 3 else 2,
            step=int(sys.argv[4]) if len(sys.argv) > 4 else 40)
    elif cmd == 'crop':
        crop(sys.argv[2], *[int(a) for a in sys.argv[3:7]],
            scale=int(sys.argv[7]) if len(sys.argv) > 7 else 4)
