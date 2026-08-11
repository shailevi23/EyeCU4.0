#!/usr/bin/env python
"""
Estimate how many keremberke `player` boxes are really goalkeeper or referee.

The problem is class identity, not geometry: the boxes exist and are good. So
nothing here redraws anything. Each existing human box is matched to a frozen
EyeCU detector prediction by IoU, and the detector's class becomes a PROPOSAL --
POSSIBLE_GOALKEEPER / POSSIBLE_REFEREE / LIKELY_PLAYER -- never a label.

The detector is an annotation assist and is treated as one. It is the model
whose weakness on goalkeeper and referee is the reason this data was wanted, so
using it as authority would be circular. Two things guard against that:

  * every proposal keeps the detector's confidence and the IoU that produced it,
    so a human reviewer sees the evidence, not just the verdict
  * an INDEPENDENT signal -- kit colour distinctness within the same frame --
    is computed separately, because officials and keepers wear kits unlike
    either team, and that fact does not come from any model

Where the two signals disagree the box is AMBIGUOUS. That is a real answer, not
a failure: those are exactly the boxes a human must actually look at.

Frames are consecutive within four broadcast runs, so proposals are also grouped
temporally -- a referee in frame N is the same person in frame N+1, and reviewing
a run is far cheaper than reviewing its frames.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'EyeCU_external_data/huggingface/keremberke_football_object_detection'
OUT = REPO / 'experiments' / 'external_sources'
EYECU_CLASSES = ['player', 'goalkeeper', 'referee', 'ball']


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def kit_signature(img, box):
    """Median hue/sat/val of the torso region -- independent of any model."""
    import cv2
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = img.shape[:2]
    # torso: middle 60% horizontally, upper 20-55% vertically
    bx, by = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 + 0.2 * bx)); cx2 = min(w, int(x2 - 0.2 * bx))
    cy1 = max(0, int(y1 + 0.20 * by)); cy2 = min(h, int(y1 + 0.55 * by))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return np.median(hsv.reshape(-1, 3), axis=0)


def hue_dist(a, b):
    """Circular hue distance in OpenCV's 0-179 space."""
    d = abs(float(a) - float(b))
    return min(d, 180.0 - d)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--iou-match', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(REPO / 'best_A_960.pt'))
    names = model.names
    print(f'frozen detector classes: {names}')

    items = []
    for split in ('train', 'valid', 'test'):
        aj = list((SRC / 'extracted' / split).rglob('_annotations.coco.json'))
        if not aj:
            continue
        base = aj[0].parent
        a = json.loads(aj[0].read_text(encoding='utf-8'))
        cats = {c['id']: c['name'] for c in a['categories']}
        by = defaultdict(list)
        for x in a['annotations']:
            by[x['image_id']].append(x)
        for im in a['images']:
            items.append((split, base / im['file_name'], im, by.get(im['id'], []), cats))
    if args.limit:
        items = items[:args.limit]
    print(f'{len(items)} images')

    rows = []
    for n, (split, path, im, anns, cats) in enumerate(items, 1):
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        preds = []
        if r.boxes is not None and len(r.boxes):
            xy = r.boxes.xyxy.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            preds = [(xy[i], names[int(cl[i])], float(cf[i])) for i in range(len(cl))]

        humans = [x for x in anns if cats.get(x['category_id']) == 'player']
        sigs = []
        for x in humans:
            bx, by, bw, bh = x['bbox']
            sigs.append(kit_signature(img, (bx, by, bx + bw, by + bh)))
        # the frame's dominant kit hues: the two teams. Officials and keepers
        # are, by design, meant to look like neither.
        valid = [s for s in sigs if s is not None]
        dom = None
        if len(valid) >= 6:
            hues = np.array([s[0] for s in valid])
            # two coarse modes via a 12-bin circular histogram
            hist, edges = np.histogram(hues, bins=12, range=(0, 180))
            top = np.argsort(hist)[::-1][:2]
            dom = [(edges[i] + edges[i + 1]) / 2 for i in top if hist[i] > 0]

        for x, sig in zip(humans, sigs):
            bx, by, bw, bh = x['bbox']
            box = (bx, by, bx + bw, by + bh)
            best, bcls, bconf = 0.0, None, 0.0
            for pxy, pcls, pconf in preds:
                v = iou(box, pxy)
                if v > best:
                    best, bcls, bconf = v, pcls, pconf
            det_proposal = (bcls if best >= args.iou_match else None)
            odd_kit = None
            if sig is not None and dom:
                odd_kit = bool(min(hue_dist(sig[0], d) for d in dom) > 22
                               and sig[1] > 60)
            rows.append({
                'split': split, 'file': im['file_name'],
                'stem': re.sub(r'[._]rf[._][0-9a-f]{6,}.*', '', im['file_name']),
                'bbox': [round(float(v), 1) for v in x['bbox']],
                'box_h': round(float(bh), 1),
                'detector_class': det_proposal,
                'detector_conf': round(float(bconf), 3),
                'detector_iou': round(float(best), 3),
                'kit_is_odd_for_this_frame': odd_kit,
            })
        if n % 150 == 0:
            print(f'  {n}/{len(items)}  boxes so far {len(rows)}')

    # ---- fuse the two independent signals -----------------------------------
    for r in rows:
        d = r['detector_class']
        k = r['kit_is_odd_for_this_frame']
        if d in ('goalkeeper', 'referee') and k is True:
            r['triage'] = f'POSSIBLE_{d.upper()}'
            r['agreement'] = 'BOTH_SIGNALS'
        elif d in ('goalkeeper', 'referee') and k is not True:
            r['triage'] = f'POSSIBLE_{d.upper()}'
            r['agreement'] = 'DETECTOR_ONLY'
        elif k is True and d == 'player':
            r['triage'] = 'AMBIGUOUS'
            r['agreement'] = 'KIT_ONLY'
        elif d == 'player':
            r['triage'] = 'LIKELY_PLAYER'
            r['agreement'] = 'BOTH_SIGNALS' if k is False else 'DETECTOR_ONLY'
        else:
            r['triage'] = 'AMBIGUOUS'
            r['agreement'] = 'NO_DETECTOR_MATCH'

    c = Counter(r['triage'] for r in rows)
    ag = Counter(r['agreement'] for r in rows)
    print(f'\n{len(rows)} human boxes')
    for k, v in c.most_common():
        print(f'   {k:<24} {v:>7}  ({100*v/len(rows):.1f}%)')
    print(f'   agreement: {dict(ag)}')

    # ---- temporal grouping ---------------------------------------------------
    def run_of(stem):
        m = re.match(r'^(\d+)(_pp)?[_.]', stem)
        if not m:
            return 'other'
        i = int(m.group(1)); fam = 'pp' if m.group(2) else 'plain'
        if fam == 'plain':
            return 'plain_A' if i <= 6000 else 'plain_B'
        return 'pp_A' if i <= 60 else 'pp_B'
    runs = defaultdict(lambda: Counter())
    for r in rows:
        r['run'] = run_of(r['stem'])
        runs[r['run']][r['triage']] += 1
    print('\nby broadcast run:')
    for k, v in sorted(runs.items()):
        print(f'   {k:<10} {dict(v)}')

    rep = {
        'method': ('existing boxes are never redrawn; each is matched by IoU to a '
                   'frozen-detector prediction and independently scored for kit '
                   'distinctness. Proposals are candidate filters, not labels.'),
        'detector': 'best_A_960.pt (frozen, production A) -- ANNOTATION ASSIST ONLY',
        'detector_is_not_annotation_authority': True,
        'iou_match_threshold': args.iou_match, 'conf': args.conf,
        'total_human_boxes': len(rows),
        'triage_counts': dict(c), 'agreement_counts': dict(ag),
        'by_run': {k: dict(v) for k, v in runs.items()},
    }
    (OUT / 'reports').mkdir(parents=True, exist_ok=True)
    (OUT / 'reports' / 'keremberke_role_triage.json').write_text(
        json.dumps(rep, indent=1), encoding='utf-8')
    (OUT / 'reports' / 'keremberke_role_triage_boxes.json').write_text(
        json.dumps(rows, indent=0), encoding='utf-8')
    print('\nwrote reports/keremberke_role_triage.json')


if __name__ == '__main__':
    main()
