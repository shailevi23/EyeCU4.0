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
    # kind='stable' so exact IoU ties resolve deterministically; the default
    # quicksort leaves tie order unspecified and makes results irreproducible.
    order = np.dstack(np.unravel_index(
        np.argsort(-ious, axis=None, kind='stable'), ious.shape))[0]
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
    # Same, broken out by source match. Frames within a match are strongly
    # correlated -- adjacent samples of one clip -- so a pooled McNemar treats
    # non-independent trials as independent and its p-value is optimistic.
    # Per-match results show whether an effect is consistent or driven by one clip.
    per_match = defaultdict(lambda: {c: Counter_() for c in range(len(CLASSES))})
    per_match_iou = defaultdict(
        lambda: {c: [0.0, 0.0, 0] for c in range(len(CLASSES))})

    for img in images:
        w, h = sizes[img.name]
        src = img.stem.rsplit('_', 1)[0]      # <match>_<frame_index>
        gt = load_gt(lbl_dir / f'{img.stem}.txt', w, h)
        for cid, boxes in gt.items():
            hit_a, iou_a = match(boxes, preds_a[img.name].get(cid, np.empty((0, 4))), args.iou)
            hit_b, iou_b = match(boxes, preds_b[img.name].get(cid, np.empty((0, 4))), args.iou)
            for k in range(len(boxes)):
                if hit_a[k] and hit_b[k]:
                    tally[cid].both += 1
                    per_match[src][cid].both += 1
                    for store in (iou_sum[cid], per_match_iou[src][cid]):
                        store[0] += iou_a[k]
                        store[1] += iou_b[k]
                        store[2] += 1
                elif hit_a[k]:
                    tally[cid].a_only += 1
                    per_match[src][cid].a_only += 1
                elif hit_b[k]:
                    tally[cid].b_only += 1
                    per_match[src][cid].b_only += 1
                else:
                    tally[cid].neither += 1
                    per_match[src][cid].neither += 1

    wanted = ([CLASSES.index(args.class_name)] if args.class_name
              else list(range(len(CLASSES))))

    fps_a, fps_b = 1000 / ms_a, 1000 / ms_b
    report = {
        'model_a': args.a, 'model_b': args.b,
        'split': args.split,
        'thresholds': {
            'confidence': args.conf,
            'match_iou': args.iou,
            'note': 'Identical for A and B. Applied without any per-model '
                    'tuning, and never selected on the test split.',
        },
        'imgsz_a': args.imgsz_a, 'imgsz_b': args.imgsz_b,
        'speed': {
            'ms_per_image_a': round(ms_a, 2), 'ms_per_image_b': round(ms_b, 2),
            'fps_a': round(fps_a, 1), 'fps_b': round(fps_b, 1),
            'fps_ratio_b_over_a': round(fps_b / fps_a, 3),
            'cost_ratio_b_over_a': round(ms_b / ms_a, 3),
        },
        'classes': {}, 'per_match': {},
    }

    print()
    print(f'thresholds (identical for both models): '
          f'confidence {args.conf}, match IoU {args.iou}')
    print(f'imgsz: A {args.imgsz_a}px, B {args.imgsz_b}px')
    print()

    print('POOLED')
    head = (f'{"class":<12}{"n":>5}{"both":>6}{"A only":>7}{"B only":>7}'
            f'{"neither":>8}{"rec A":>8}{"rec B":>8}{"abs D":>8}{"rel D":>8}'
            f'{"IoU D":>8}{"p":>8}')
    print(head)
    print('-' * len(head))
    for cid in wanted:
        t = tally[cid]
        n = t.both + t.a_only + t.b_only + t.neither
        if n == 0:
            continue
        ra = (t.both + t.a_only) / n
        rb = (t.both + t.b_only) / n
        abs_d = rb - ra
        rel_d = (abs_d / ra) if ra else None
        si = iou_sum[cid]
        iou_a_m = si[0] / si[2] if si[2] else None
        iou_b_m = si[1] / si[2] if si[2] else None
        iou_d = (iou_b_m - iou_a_m) if si[2] else None
        pval, note = mcnemar(t.a_only, t.b_only)

        rel_s = f'{rel_d:+.1%}' if rel_d is not None else '-'
        iou_s = f'{iou_d:+.3f}' if iou_d is not None else '-'
        print(f'{CLASSES[cid]:<12}{n:>5}{t.both:>6}{t.a_only:>7}{t.b_only:>7}'
              f'{t.neither:>8}{ra:>8.3f}{rb:>8.3f}{abs_d:>+8.3f}'
              f'{rel_s:>8}{iou_s:>8}{pval:>8.4f}')
        if note:
            print(f'             {note}')

        report['classes'][CLASSES[cid]] = {
            'n': n, 'both': t.both, 'a_only': t.a_only, 'b_only': t.b_only,
            'neither': t.neither,
            'recall_a': round(ra, 4), 'recall_b': round(rb, 4),
            'recall_delta_absolute': round(abs_d, 4),
            'recall_delta_relative': (round(rel_d, 4) if rel_d is not None else None),
            'mean_iou_a_on_shared': round(iou_a_m, 4) if si[2] else None,
            'mean_iou_b_on_shared': round(iou_b_m, 4) if si[2] else None,
            'mean_iou_delta_on_shared': round(iou_d, 4) if si[2] else None,
            'n_shared_for_iou': si[2],
            'mcnemar_p_pooled': round(pval, 6),
        }

    print()
    print('PER VALIDATION MATCH')
    print('Frames within a match are correlated, so the pooled p above is')
    print('optimistic. A real effect should show up across several matches,')
    print('not rest on one.')
    for cid in wanted:
        rows = []
        for m in sorted(per_match):
            t = per_match[m][cid]
            if t.both + t.a_only + t.b_only + t.neither:
                rows.append((m, t))
        if not rows:
            continue
        print()
        print(f'  {CLASSES[cid]}')
        h2 = (f'    {"match":<34}{"n":>4}{"both":>6}{"A only":>7}{"B only":>7}'
              f'{"neither":>8}{"rec A":>7}{"rec B":>7}{"abs D":>8}{"IoU D":>8}')
        print(h2)
        print('    ' + '-' * (len(h2) - 4))
        entries = {}
        for m, t in rows:
            n = t.both + t.a_only + t.b_only + t.neither
            ra = (t.both + t.a_only) / n
            rb = (t.both + t.b_only) / n
            si = per_match_iou[m][cid]
            iou_d = (si[1] - si[0]) / si[2] if si[2] else None
            iou_s = f'{iou_d:+.3f}' if iou_d is not None else '-'
            print(f'    {m:<34}{n:>4}{t.both:>6}{t.a_only:>7}{t.b_only:>7}'
                  f'{t.neither:>8}{ra:>7.3f}{rb:>7.3f}{rb - ra:>+8.3f}{iou_s:>8}')
            entries[m] = {
                'n': n, 'both': t.both, 'a_only': t.a_only,
                'b_only': t.b_only, 'neither': t.neither,
                'recall_a': round(ra, 4), 'recall_b': round(rb, 4),
                'recall_delta_absolute': round(rb - ra, 4),
                'mean_iou_delta_on_shared': round(iou_d, 4) if si[2] else None,
            }
        wins = sum(1 for e in entries.values() if e['recall_delta_absolute'] > 0)
        losses = sum(1 for e in entries.values() if e['recall_delta_absolute'] < 0)
        ties = len(entries) - wins - losses
        print(f'    -> B better in {wins}, worse in {losses}, tied in {ties} '
              f'of {len(entries)} matches')
        report['per_match'][CLASSES[cid]] = {
            'matches': entries,
            'b_better_in': wins, 'b_worse_in': losses, 'tied': ties,
        }

    print()
    print(f'SPEED   A {ms_a:.1f} ms/img ({fps_a:.1f} FPS)   '
          f'B {ms_b:.1f} ms/img ({fps_b:.1f} FPS)')
    print(f'        FPS ratio B/A = {fps_b / fps_a:.3f}   '
          f'(B costs {ms_b / ms_a:.2f}x per image)')

    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print()
    print(f'written: {args.out}')
    print()
    print('Reading this. "A only" and "B only" are the discordant pairs and the')
    print('only evidence about which model is better; concordant outcomes carry')
    print('no signal. p is a two-sided exact McNemar on those pairs, POOLED, and')
    print('is optimistic because frames within a match are correlated -- weigh')
    print('per-match consistency at least as heavily. Judge adoption on effect')
    print('size and cost, not on p alone.')


class Counter_:
    __slots__ = ('both', 'a_only', 'b_only', 'neither')

    def __init__(self):
        self.both = self.a_only = self.b_only = self.neither = 0


if __name__ == '__main__':
    main()
