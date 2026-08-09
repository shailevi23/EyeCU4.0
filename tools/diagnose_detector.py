#!/usr/bin/env python
"""
Detector diagnostics on the frozen validation split.

Standard per-class precision/recall answers "how good is the detector". It does
not answer the question that actually decides what to fix next: when the system
gets a person wrong, did it fail to *find* them, or find them and label the role
wrong? Those two failures have completely different remedies -- more data and
better features for the first, temporal role smoothing for the second -- and
per-class recall cannot tell them apart, because a goalkeeper detected as a
player is scored simply as a goalkeeper miss.

So this reports, in one inference pass:

  per-class P/R          the familiar numbers, at the production threshold
  any-human recall       a GT player/goalkeeper/referee counts as found if ANY
                         human-role prediction matches it. The gap between this
                         and per-class recall is pure role confusion.
  role confusion matrix  what each missed/found human was actually called
  ball threshold grid    a predeclared sweep, to see whether usable ball
                         proposals already exist below the production threshold
  per-match ball P/R     because frames within a match are correlated

Everything is VAL-only by construction: --split test is refused. The threshold
grid is predeclared and run once; it is a measurement, not a tuning loop.

Example:
    python tools/diagnose_detector.py --model best_A_960.pt --imgsz 960
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_models import CLASSES, IMG_EXTS, iou_matrix, load_gt  # noqa: E402

HUMAN_CLASSES = ('player', 'goalkeeper', 'referee')
HUMAN_IDS = tuple(CLASSES.index(c) for c in HUMAN_CLASSES)
BALL_ID = CLASSES.index('ball')

# Predeclared. Fixed before looking at any result; not widened afterwards.
BALL_THRESHOLD_GRID = (0.05, 0.10, 0.15, 0.20, 0.25)


def match_indices(gt_boxes, pred_boxes, thr):
    """
    Greedy highest-IoU one-to-one matching.

    Returns an array over ground truth holding the matched prediction index,
    or -1 where nothing matched. Unlike compare_models.match this keeps the
    index, because the caller needs the matched prediction's *class* to tell a
    localisation failure from a role error.
    """
    n = len(gt_boxes)
    matched = np.full(n, -1, dtype=int)
    if n == 0 or len(pred_boxes) == 0:
        return matched

    ious = iou_matrix(np.asarray(gt_boxes), np.asarray(pred_boxes))
    used = set()
    # kind='stable' so exact IoU ties always resolve to the lower (gt, pred)
    # index pair. The default quicksort leaves tie order unspecified, which
    # makes the whole diagnostic non-reproducible run to run.
    order = np.dstack(np.unravel_index(
        np.argsort(-ious, axis=None, kind='stable'), ious.shape))[0]
    for g, p in order:
        if ious[g, p] < thr:
            break
        if matched[g] >= 0 or p in used:
            continue
        matched[g] = p
        used.add(p)
    return matched


def predict_all(model_path, images, imgsz, floor_conf, device=None):
    """One pass at a low floor; every threshold above it is then free."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    preds = {}
    for i, img in enumerate(images, 1):
        kw = dict(conf=floor_conf, imgsz=imgsz, verbose=False)
        if device:
            kw['device'] = device
        r = model.predict(str(img), **kw)[0]
        boxes, clss, confs = [], [], []
        for box in r.boxes:
            boxes.append(box.xyxy[0].tolist())
            clss.append(int(box.cls))
            confs.append(float(box.conf))
        preds[img.name] = (np.array(boxes).reshape(-1, 4),
                           np.array(clss, dtype=int),
                           np.array(confs))
        if i % 50 == 0:
            print(f'    {i}/{len(images)}')
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--dataset', default='data/dataset_baseline')
    ap.add_argument('--split', default='val',
                    help='NEVER use test for diagnostics or threshold choice.')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--conf', type=float, default=0.25,
                    help='Production confidence threshold.')
    ap.add_argument('--iou', type=float, default=0.5,
                    help='IoU at which a ground-truth object counts as found.')
    ap.add_argument('--out', default='detector_diagnostics.json')
    args = ap.parse_args()

    if args.split == 'test':
        sys.exit('Refusing: the frozen test split must not be used for '
                 'diagnostics, thresholds or any development decision.')

    root = Path(args.dataset)
    img_dir, lbl_dir = root / 'images' / args.split, root / 'labels' / args.split
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        sys.exit(f'No images in {img_dir}')
    print(f'{len(images)} images in {args.split}')

    from PIL import Image
    sizes = {p.name: Image.open(p).size for p in images}

    floor = min(BALL_THRESHOLD_GRID)
    print(f'\n{args.model} @ {args.imgsz}px (single pass at conf>={floor})')
    preds = predict_all(args.model, images, args.imgsz, floor)

    # ---------------------------------------------------------------- tallies
    tp = Counter()          # per class, at production conf
    fp = Counter()
    n_gt = Counter()
    any_human_found = 0
    n_human_gt = 0
    confusion = defaultdict(Counter)          # gt class -> predicted-as
    ball_grid = {t: {'tp': 0, 'fp': 0} for t in BALL_THRESHOLD_GRID}
    ball_per_match = defaultdict(lambda: {t: {'tp': 0, 'fp': 0, 'n': 0}
                                          for t in BALL_THRESHOLD_GRID})

    for img in images:
        w, h = sizes[img.name]
        gt = load_gt(lbl_dir / f'{img.stem}.txt', w, h)
        pboxes, pcls, pconf = preds[img.name]
        keep = pconf >= args.conf
        src = img.stem.rsplit('_', 1)[0]

        # --- standard per-class, at the production threshold
        for cid in range(len(CLASSES)):
            g = gt.get(cid, np.empty((0, 4)))
            n_gt[cid] += len(g)
            sel = keep & (pcls == cid)
            p = pboxes[sel]
            m = match_indices(g, p, args.iou)
            hits = int((m >= 0).sum())
            tp[cid] += hits
            fp[cid] += len(p) - hits

        # --- any-human recall and role confusion, class-agnostic over humans
        gt_h_boxes, gt_h_cls = [], []
        for cid in HUMAN_IDS:
            for b in gt.get(cid, np.empty((0, 4))):
                gt_h_boxes.append(b)
                gt_h_cls.append(cid)
        sel_h = keep & np.isin(pcls, HUMAN_IDS)
        ph_boxes, ph_cls = pboxes[sel_h], pcls[sel_h]

        m = match_indices(gt_h_boxes, ph_boxes, args.iou)
        n_human_gt += len(gt_h_boxes)
        any_human_found += int((m >= 0).sum())
        for k, pi in enumerate(m):
            got = CLASSES[ph_cls[pi]] if pi >= 0 else 'MISSED'
            confusion[CLASSES[gt_h_cls[k]]][got] += 1

        # --- ball threshold grid, pooled and per match
        gball = gt.get(BALL_ID, np.empty((0, 4)))
        ball_per_match[src][BALL_THRESHOLD_GRID[0]]  # touch so match appears
        for t in BALL_THRESHOLD_GRID:
            selb = (pconf >= t) & (pcls == BALL_ID)
            pb = pboxes[selb]
            mb = match_indices(gball, pb, args.iou)
            hits = int((mb >= 0).sum())
            ball_grid[t]['tp'] += hits
            ball_grid[t]['fp'] += len(pb) - hits
            ball_per_match[src][t]['tp'] += hits
            ball_per_match[src][t]['fp'] += len(pb) - hits
            ball_per_match[src][t]['n'] += len(gball)

    # ---------------------------------------------------------------- report
    def pr(t, f, n):
        prec = t / (t + f) if (t + f) else None
        rec = t / n if n else None
        return prec, rec

    report = {'model': args.model, 'split': args.split, 'imgsz': args.imgsz,
              'thresholds': {'confidence': args.conf, 'match_iou': args.iou},
              'n_images': len(images)}

    print(f'\nthresholds: confidence {args.conf}, match IoU {args.iou}')
    print('\nPER CLASS (production threshold)')
    head = f'{"class":<12}{"n":>6}{"TP":>6}{"FP":>6}{"prec":>8}{"recall":>8}'
    print(head); print('-' * len(head))
    report['per_class'] = {}
    for cid, name in enumerate(CLASSES):
        prec, rec = pr(tp[cid], fp[cid], n_gt[cid])
        print(f'{name:<12}{n_gt[cid]:>6}{tp[cid]:>6}{fp[cid]:>6}'
              f'{prec if prec is None else round(prec,3):>8}'
              f'{rec if rec is None else round(rec,3):>8}')
        report['per_class'][name] = {
            'n': n_gt[cid], 'tp': tp[cid], 'fp': fp[cid],
            'precision': None if prec is None else round(prec, 4),
            'recall': None if rec is None else round(rec, 4)}

    # --- any-human, as ONE mutually exclusive partition of the human GT.
    #
    # Every count below comes from the SAME class-agnostic one-to-one matching
    # that built the confusion matrix, so the three buckets sum to the GT total
    # by construction. Do not substitute the per-class TP sum for "correct
    # role": per-class matching runs a separate assignment per class, where a
    # GT competes only against predictions of its own class. That is a strictly
    # easier problem -- no cross-class prediction can steal a GT -- so it scores
    # higher, and subtracting it from a class-agnostic total mixes two
    # incompatible definitions and produces a wrong mis-role count.
    correct_role = sum(confusion[c][c] for c in HUMAN_CLASSES)
    wrong_role = any_human_found - correct_role
    missed = n_human_gt - any_human_found
    assert correct_role + wrong_role + missed == n_human_gt, 'partition must be exact'

    per_class_sum = sum(tp[c] for c in HUMAN_IDS)   # class-aware, for contrast only
    any_rec = any_human_found / n_human_gt if n_human_gt else 0.0
    strict_rec = correct_role / n_human_gt if n_human_gt else 0.0

    print('\nANY-HUMAN LOCALISATION  '
          '(single class-agnostic one-to-one matching; buckets are exclusive)')
    print(f'  human GT instances          {n_human_gt}')
    print(f'  correct role                {correct_role}  (recall {strict_rec:.3f})')
    print(f'  wrong human role            {wrong_role}')
    print(f'  missed entirely             {missed}')
    print(f'  located by ANY human role   {any_human_found}  (recall {any_rec:.3f})')
    print(f'  [class-aware TP sum         {per_class_sum}  -- a DIFFERENT matching '
          f'rule, shown only for contrast; never mix it with the buckets above]')
    report['any_human'] = {
        'matching': 'class-agnostic one-to-one over player/goalkeeper/referee',
        'n': n_human_gt,
        'correct_role': correct_role,
        'wrong_human_role': wrong_role,
        'missed_entirely': missed,
        'located_any_role': any_human_found,
        'recall_correct_role': round(strict_rec, 4),
        'recall_any_role': round(any_rec, 4),
        'class_aware_tp_sum_not_comparable': per_class_sum}

    print('\nROLE CONFUSION  (rows = ground truth, cols = what it was called)')
    cols = list(HUMAN_CLASSES) + ['MISSED']
    corner = 'gt \\ pred'
    h2 = f'{corner:<14}' + ''.join(f'{c:>12}' for c in cols)
    print(h2); print('-' * len(h2))
    report['role_confusion'] = {}
    for gtc in HUMAN_CLASSES:
        row = confusion[gtc]
        print(f'{gtc:<14}' + ''.join(f'{row.get(c,0):>12}' for c in cols))
        report['role_confusion'][gtc] = {c: row.get(c, 0) for c in cols}

    print(f'\nBALL THRESHOLD GRID  (predeclared {list(BALL_THRESHOLD_GRID)}, run once)')
    h3 = f'{"conf":>6}{"TP":>6}{"FP":>6}{"prec":>8}{"recall":>8}{"FP/img":>9}'
    print(h3); print('-' * len(h3))
    report['ball_threshold_grid'] = {}
    n_ball = n_gt[BALL_ID]
    for t in BALL_THRESHOLD_GRID:
        d = ball_grid[t]
        prec, rec = pr(d['tp'], d['fp'], n_ball)
        fpi = d['fp'] / len(images)
        print(f'{t:>6}{d["tp"]:>6}{d["fp"]:>6}'
              f'{0 if prec is None else prec:>8.3f}{0 if rec is None else rec:>8.3f}{fpi:>9.2f}')
        report['ball_threshold_grid'][str(t)] = {
            'tp': d['tp'], 'fp': d['fp'],
            'precision': None if prec is None else round(prec, 4),
            'recall': None if rec is None else round(rec, 4),
            'fp_per_image': round(fpi, 4)}

    print('\nBALL PER MATCH')
    h4 = f'{"match":<32}{"n":>5}' + ''.join(f'{("R@"+str(t)):>9}' for t in BALL_THRESHOLD_GRID)
    print(h4); print('-' * len(h4))
    report['ball_per_match'] = {}
    for m_ in sorted(ball_per_match):
        d = ball_per_match[m_]
        n_ = d[BALL_THRESHOLD_GRID[0]]['n']
        cells, entry = '', {}
        for t in BALL_THRESHOLD_GRID:
            r_ = d[t]['tp'] / n_ if n_ else 0.0
            cells += f'{r_:>9.3f}'
            entry[str(t)] = {'tp': d[t]['tp'], 'fp': d[t]['fp'],
                             'recall': round(r_, 4)}
        print(f'{m_:<32}{n_:>5}{cells}')
        report['ball_per_match'][m_] = {'n': n_, 'by_threshold': entry}

    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nwritten: {args.out}')
    print('\nReading this. The gap between any-human recall and correct-role')
    print('recall is role confusion, which temporal smoothing can address; the')
    print('"not located at all" count is a genuine detection failure, which it')
    print('cannot. Lowering the ball threshold trades precision for recall --')
    print('judge it on FP/img, since every false ball corrupts possession.')


if __name__ == '__main__':
    main()
