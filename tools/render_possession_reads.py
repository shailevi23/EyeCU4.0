#!/usr/bin/env python
"""
P1 -- render blind annotation images for POSSESSION_VAL_V1.

Draws ONLY: the frame, its index, VERIFIED GT human boxes and GT track ids.
It never loads the detector, the selector, the tracker or the assigner, so it
cannot leak a model prediction into the annotator's view. See
experiments/records/experiment_P1/ANNOTATION_PROTOCOL_V2.md.

Two panels per frame:
    top     full frame, upscaled, GT boxes + ids
    bottom  magnified crop, for finding a ball a few pixels across

The crop centre defaults to the centroid of the GT boxes on that frame -- a
GT-derived quantity, not a model output. --center overrides it once the
annotator has located the ball by eye in an earlier frame of the same window.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SEQ_DIR = Path('data/tracking_val_gt/sequences')
MOT_DIR = Path('data/tracking_val_gt/mot/EyeCU-val')
PV = Path('data/possession_val_v1')

FULL_SCALE = 1.7
CROP_W, CROP_H = 300, 170
CROP_SCALE = 3.6


def load_gt(seq):
    out = defaultdict(list)
    p = MOT_DIR / seq / 'gt' / 'gt.txt'
    for line in p.read_text(encoding='utf-8').splitlines():
        q = line.split(',')
        if len(q) >= 6:
            x, y, w, h = (float(v) for v in q[2:6])
            out[int(q[0])].append((int(q[1]), [x, y, x + w, y + h]))
    return out


def imread(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def draw_gt(img, boxes, scale, thickness=1, font=0.42):
    for tid, (x1, y1, x2, y2) in boxes:
        p1 = (int(x1 * scale), int(y1 * scale))
        p2 = (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(img, p1, p2, (0, 235, 0), thickness)
        cv2.putText(img, str(tid), (p1[0], max(11, p1[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), 3)
        cv2.putText(img, str(tid), (p1[0], max(11, p1[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (40, 255, 255), 1)
    return img


def render(seq, frame, centre=None):
    gt = load_gt(seq)[frame]
    src = imread(SEQ_DIR / seq / 'img1' / f'{frame:06d}.jpg')
    h, w = src.shape[:2]

    full = cv2.resize(src, None, fx=FULL_SCALE, fy=FULL_SCALE,
                      interpolation=cv2.INTER_CUBIC)
    draw_gt(full, gt, FULL_SCALE)
    cv2.rectangle(full, (0, 0), (250, 26), (0, 0, 0), -1)
    cv2.putText(full, f'f{frame}  FULL', (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    if centre is None:
        if gt:
            cx = int(np.mean([(b[0] + b[2]) / 2 for _, b in gt]))
            cy = int(np.mean([(b[1] + b[3]) / 2 for _, b in gt]))
        else:
            cx, cy = w // 2, h // 2
    else:
        cx, cy = centre
    x0 = max(0, min(int(cx) - CROP_W // 2, w - CROP_W))
    y0 = max(0, min(int(cy) - CROP_H // 2, h - CROP_H))
    crop = src[y0:y0 + CROP_H, x0:x0 + CROP_W].copy()
    crop = cv2.resize(crop, None, fx=CROP_SCALE, fy=CROP_SCALE,
                      interpolation=cv2.INTER_CUBIC)
    shifted = [(tid, [b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0]) for tid, b in gt]
    draw_gt(crop, shifted, CROP_SCALE, thickness=1, font=0.5)
    cv2.rectangle(crop, (0, 0), (430, 26), (0, 0, 0), -1)
    cv2.putText(crop, f'f{frame}  ZOOM x{CROP_SCALE} @({x0},{y0})', (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    tw = max(full.shape[1], crop.shape[1])
    def pad(a):
        if a.shape[1] == tw:
            return a
        return np.hstack([a, np.zeros((a.shape[0], tw - a.shape[1], 3), np.uint8)])
    return np.vstack([pad(full), np.zeros((6, tw, 3), np.uint8), pad(crop)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--window', help='window_id from the frozen list')
    ap.add_argument('--frames', help='comma list of 1-based frame numbers')
    ap.add_argument('--centers', default='',
                    help='frame=x,y pairs separated by ";" (crop centre override)')
    ap.add_argument('--out', default='data/possession_val_v1/p1_reads')
    args = ap.parse_args()

    frozen = json.loads((PV / 'POSSESSION_VAL_V1_FROZEN.json')
                        .read_text(encoding='utf-8'))
    wins = {w['window_id']: w for w in frozen['windows']}
    w = wins[args.window]
    frames = ([int(x) for x in args.frames.split(',')] if args.frames
              else w['frames_1based'])
    for f in frames:
        assert f in w['frames_1based'], f'{f} is not in the frozen window'

    centres = {}
    for part in filter(None, args.centers.split(';')):
        k, v = part.split('=')
        centres[int(k)] = tuple(int(z) for z in v.split(','))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in frames:
        img = render(w['sequence'], f, centres.get(f))
        name = f"{args.window}_f{f}.jpg"
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / name).write_bytes(buf.tobytes())
        print(out / name, img.shape)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
