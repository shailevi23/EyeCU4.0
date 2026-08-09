#!/usr/bin/env python
"""
Compare two detectors on the SAME validation instances, paired.

Why paired. Two models scored on one validation set are not independent
samples: they see identical images and identical ground-truth objects. Asking
whether model B's point estimate clears model A's independent confidence
interval throws away that pairing and is far too conservative — with 111 ball
instances it would demand a ~19-point jump before calling anything real. A
paired test asks a sharper question: *on the instances where the two models
disagree, which one wins?*

For each ground-truth object this reports one of four outcomes:

    both      both models found it
    A only    A found it, B missed it     ) the discordant pairs --
    B only    B found it, A missed it     ) these are what McNemar tests
    neither   both missed it

McNemar's test then asks whether the split between "A only" and "B only" is
lopsided enough to be more than chance. Concordant pairs (both/neither) carry
no information about which model is better and are excluded by construction.

Also reports localisation quality (mean IoU on jointly-found objects) and
inference speed, so a resolution change can be judged on cost as well.

Examples:
    python tools/compare_models.py --a runs/A/best.pt --b runs/B/best.pt
    python tools/compare_models.py --a A.pt --b B.pt --class-name ball --imgsz-a 960 --imgsz-b 1280
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = ['player', 'goalkeeper', 'referee', 'ball']
IMG_EXTS = {'.jpg', '.jpeg', '.png'}


def iou_matrix(gt, pred):
    """gt: (N,4) xyxy, pred: (M,4) xyxy -> (N,M) IoU."""
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros((len(gt), len(pred)))
    x1 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    y1 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    y2 = np.minimum(gt[:, None, 3], pred[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ga = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    pa = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    union = ga[:, None] + pa[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def load_gt(label_path: Path, w: int, h: int):
    """YOLO label file -> {class_id: (K,4) xyxy in pixels}."""
    out = defaultdict(list)
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cid = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        out[cid].append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                         (cx + bw / 2) * w, (cy + bh / 2) * h])
    return {k: np.array(v) for k, v in out.items()}


def match(gt_boxes, pred_boxes, thr):
    """
    Greedy highest-IoU matching, each prediction used once.
    Returns (hit_mask over gt, iou_of_match).
    """
    n = len(gt_boxes)
    hit = np.zeros(n, dtype=bool)
    best = np.zeros(n)
    if n == 0 or len(pred_boxes) == 0:
        return hit, best

    ious = iou_matrix(gt_boxes, pred_boxes)
    used = set()
    order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
    for g, p in order:
        if ious[g, p] < thr:
            break
        if hit[g] or p in used:
            continue
        hit[g] = True
        best[g] = ious[g, p]
        used.add(p)
    return hit, best


def mcnemar(a_only: int, b_only: int):
    """
    Exact two-sided binomial McNemar on the discordant pairs.
    Returns (p_value, note). No SciPy dependency.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0, 'no discordant pairs — the models behave identically here'
    from math import comb
    k = min(a_only, b_only)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p), ''


def predict_all(model_path, images, imgsz, conf, device=None):
    from ultralytics import YOLO
    model = YOLO(model_path)
    preds, total_ms = {}, 0.0
    for i, img in enumerate(images, 1):
        kw = dict(conf=conf, imgsz=imgsz, verbose=False)
        if device:
            kw['device'] = device
        t0 = time.time()
        r = model.predict(str(img), **kw)[0]
        total_ms += (time.time() - t0) * 1000
        by_cls = defaultdict(list)
        for box in r.boxes:
            by_cls[int(box.cls)].append(box.xyxy[0].tolist())
        preds[img.name] = {k: np.array(v) for k, v in by_cls.items()}
        if i % 50 == 0:
            print(f'    {i}/{len(images)}')
    return preds, total_ms / max(1, len(images))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', required=True, help='Model A weights (.pt)')
    p.add_argument('--b', required=True, help='Model B weights (.pt)')
    p.add_argument('--dataset', default='data/dataset_baseline')
    p.add_argument('--split', default='val',
                   help='NEVER use test for model selection.')
    p.add_argument('--imgsz-a', type=int, default=960)
    p.add_argument('--imgsz-b', type=int, default=1280)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.5,
                   help='IoU at which a ground-truth object counts as found.')
    p.add_argument('--class-name', help='Restrict output to one class.')
    p.add_argument('--out', default='data/model_comparison.json')
    args = p.parse_args()

    if args.split == 'test':
        sys.exit('Refusing to run on test. The test split is scored once, after '
                 'the model is chosen on val. Use --split val.')

    root = Path(args.dataset)
    img_dir, lbl_dir = root / 'images' / args.split, root / 'labels' / args.split
    if not img_dir.is_dir():
        sys.exit(f'Not found: {img_dir}')
    images = sorted(q for q in img_dir.iterdir() if q.suffix.lower() in IMG_EXTS)
    print(f'{len(images)} images in {args.split}\n')

    import cv2
    sizes = {}
    for img in images:
        arr = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)
        sizes[img.name] = (arr.shape[1], arr.shape[0])

    print(f'A: {args.a} @ {args.imgsz_a}px')
    preds_a, ms_a = predict_all(args.a, images, args.imgsz_a, args.conf)
    print(f'B: {args.b} @ {args.imgsz_b}px')
    preds_b, ms_b = predict_all(args.b, images, args.imgsz_b, args.conf)

    # Pair every ground-truth object across the two models.
    tally = {c: Counter_() for c in range(len(CLASSES))}
    iou_sum = {c: [0.0, 0.0, 0] for c in range(len(CLASSES))}  # A, B, n_both

    for img in images:
        w, h = sizes[img.name]
        gt = load_gt(lbl_dir / f'{img.stem}.txt', w, h)
        for cid, boxes in gt.items():
            hit_a, iou_a = match(boxes, preds_a[img.name].get(cid, np.empty((0, 4))), args.iou)
            hit_b, iou_b = match(boxes, preds_b[img.name].get(cid, np.empty((0, 4))), args.iou)
            for k in range(len(boxes)):
                if hit_a[k] and hit_b[k]:
                    tally[cid].both += 1
                    iou_sum[cid][0] += iou_a[k]
                    iou_sum[cid][1] += iou_b[k]
                    iou_sum[cid][2] += 1
                elif hit_a[k]:
                    tally[cid].a_only += 1
                elif hit_b[k]:
                    tally[cid].b_only += 1
                else:
                    tally[cid].neither += 1

    wanted = [CLASSES.index(args.class_name)] if args.class_name else range(len(CLASSES))
    report = {'model_a': args.a, 'model_b': args.b, 'split': args.split,
              'imgsz_a': args.imgsz_a, 'imgsz_b': args.imgsz_b,
              'iou_threshold': args.iou, 'conf': args.conf,
              'ms_per_image_a': round(ms_a, 2), 'ms_per_image_b': round(ms_b, 2),
              'classes': {}}

    print(f'\n{"class":<12}{"n":>5}{"both":>7}{"A only":>8}{"B only":>8}{"neither":>9}'
          f'{"recall A":>10}{"recall B":>10}{"p":>9}')
    print('-' * 78)
    for cid in wanted:
        t = tally[cid]
        n = t.both + t.a_only + t.b_only + t.neither
        if n == 0:
            continue
        ra = (t.both + t.a_only) / n
        rb = (t.both + t.b_only) / n
        pval, note = mcnemar(t.a_only, t.b_only)
        print(f'{CLASSES[cid]:<12}{n:>5}{t.both:>7}{t.a_only:>8}{t.b_only:>8}'
              f'{t.neither:>9}{ra:>10.3f}{rb:>10.3f}{pval:>9.4f}')
        if note:
            print(f'             {note}')
        s = iou_sum[cid]
        report['classes'][CLASSES[cid]] = {
            'n': n, 'both': t.both, 'a_only': t.a_only, 'b_only': t.b_only,
            'neither': t.neither, 'recall_a': round(ra, 4), 'recall_b': round(rb, 4),
            'mcnemar_p': round(pval, 6),
            'mean_iou_a_on_shared': round(s[0] / s[2], 4) if s[2] else None,
            'mean_iou_b_on_shared': round(s[1] / s[2], 4) if s[2] else None,
        }

    print(f'\nspeed: A {ms_a:.1f} ms/img ({1000 / ms_a:.1f} FPS)   '
          f'B {ms_b:.1f} ms/img ({1000 / ms_b:.1f} FPS)   '
          f'B is {ms_b / ms_a:.2f}x the cost')

    print('\nlocalisation on jointly-found objects (mean IoU):')
    for cid in wanted:
        s = iou_sum[cid]
        if s[2]:
            print(f'  {CLASSES[cid]:<12} A {s[0] / s[2]:.3f}   B {s[1] / s[2]:.3f}   '
                  f'(n={s[2]})')

    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nwritten: {args.out}')
    print('\nReading this: "A only" and "B only" are the discordant pairs — the '
          'only\nevidence about which model is better. p is a two-sided exact '
          'McNemar test\non those pairs. Concordant outcomes carry no signal '
          'and are excluded.')


class Counter_:
    __slots__ = ('both', 'a_only', 'b_only', 'neither')

    def __init__(self):
        self.both = self.a_only = self.b_only = self.neither = 0


if __name__ == '__main__':
    main()
