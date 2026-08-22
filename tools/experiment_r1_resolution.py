#!/usr/bin/env python
"""
R1 -- inference-resolution isolation. DIAGNOSTIC / EXPERIMENT ONLY.

Six cells: {best_A_960.pt, best_B_1280.pt} x {960, 1280, 1920} inference imgsz.
Nothing else changes -- same original images, same conf, same NMS/IoU arg, same
preprocessing path (LocalDetector -> Ultralytics predict), same device, same GT.
No SAHI, no TTA, no augment, no tiling, no temporal selector, no retraining.

One forward pass per image per cell. Everything is derived from that one pass:

  raw proposals      every ball box at confidence >= PROPOSAL_FLOOR (0.01)
  frozen accepted    ball boxes >= BALL_CANDIDATE_CONF, suppress_ball_duplicates,
                     then kept at >= BALL_ACCEPT_CONF

That reproduces the frozen path of tools/eval_temporal_val.py and the proposal
path of tools/diagnose_temporal_ball_proposals.py without paying for three
separate passes. suppress_ball_duplicates is confidence-descending and greedy, so
adding boxes strictly below 0.10 cannot change which >= 0.25 boxes survive; the
reproduction check below is what actually validates that claim.

VALIDATION ONLY. Sealed TEST is unreachable from this file -- there is no split
argument, and both benchmarks are VAL_ONLY by construction.

    python tools/experiment_r1_resolution.py --out-dir <dir>
"""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_models import iou_matrix                                  # noqa: E402
from trackers.detector import (BALL_ACCEPT_CONF, BALL_CANDIDATE_CONF,   # noqa: E402
                               BALL_DEDUPE_IOU, HUMAN_CLASSES,
                               LocalDetector, suppress_ball_duplicates)

MATCH_IOU = 0.5
PROPOSAL_FLOOR = 0.01
NMS_IOU_ARG = 0.5          # inert on YOLO26 end-to-end; pinned identically anyway
DEVICE = 'cpu'
VAL = Path('data/dataset_baseline')
TV = Path('data/temporal_val')
BALL_CLS = 3

CELLS = [('A', 'best_A_960.pt', 960), ('A', 'best_A_960.pt', 1280),
         ('A', 'best_A_960.pt', 1920), ('B', 'best_B_1280.pt', 960),
         ('B', 'best_B_1280.pt', 1280), ('B', 'best_B_1280.pt', 1920)]
REFERENCE_CELL = 'A@960'   # defines the contact failure set membership

# Size strata, on the native 640x360 benchmark geometry, GT max(w, h) in px.
STRATA = (('tiny', 0.0, 12.0), ('small', 12.0, 20.0), ('larger', 20.0, 1e9))


def stratum(px: float) -> str:
    for name, lo, hi in STRATA:
        if lo <= px < hi:
            return name
    return 'larger'


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_gt(path: Path, w: int, h: int, cls=None):
    """YOLO txt -> xyxy px. cls=None keeps every class."""
    out = []
    if not path.exists():
        return np.empty((0, 4))
    for line in path.read_text(encoding='utf-8').splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        if cls is not None and int(p[0]) != cls:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out).reshape(-1, 4)


def best_iou(gt_box, boxes) -> float:
    if not len(boxes):
        return 0.0
    m = iou_matrix(np.asarray(gt_box).reshape(1, 4),
                   np.array([b['bbox'] for b in boxes]))
    return float(m.max())


def frozen_accept(balls):
    """The production ball path: candidate floor, frozen dedupe, accept at 0.25."""
    pool = [d for d in balls if d['confidence'] >= BALL_CANDIDATE_CONF]
    kept = suppress_ball_duplicates(pool, BALL_DEDUPE_IOU)
    acc = [d for d in kept if d['confidence'] >= BALL_ACCEPT_CONF]
    acc.sort(key=lambda d: -d['confidence'])
    return acc


class ShapeProbe:
    """Records the ACTUAL tensor shape reaching the network."""

    def __init__(self, det: LocalDetector):
        self.shapes = defaultdict(int)
        self._h = det.model.model.register_forward_pre_hook(self._hook)

    def _hook(self, _mod, args):
        t = args[0]
        if hasattr(t, 'shape') and len(t.shape) == 4:
            self.shapes[tuple(int(v) for v in t.shape)] += 1

    def close(self):
        self._h.remove()


def run_cell(weights: str, imgsz: int):
    """One forward pass per image over both benchmarks. Returns raw records."""
    import cv2
    from PIL import Image

    det = LocalDetector(weights, confidence=PROPOSAL_FLOOR, iou=NMS_IOU_ARG,
                        imgsz=imgsz, device=DEVICE, ball_candidate_pool=False)
    probe = ShapeProbe(det)
    rec = {'val208': {}, 'temporal': {}, 'seconds': 0.0}
    t0 = time.time()

    def one(path: Path):
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        w, h = Image.open(path).size
        dets = det.detect(img)
        balls = [d for d in dets if d['class'] == 'ball']
        humans = [d for d in dets if d['class'] in HUMAN_CLASSES
                  and d['confidence'] >= 0.25]
        return (w, h), balls, humans

    for p in sorted((VAL / 'images' / 'val').iterdir()):
        (w, h), balls, humans = one(p)
        rec['val208'][p.name] = {
            'wh': [w, h],
            'gt_ball': read_gt(VAL / 'labels' / 'val' / f'{p.stem}.txt', w, h,
                               BALL_CLS).tolist(),
            'proposals': balls,
            'accepted': frozen_accept(balls),
            'humans': humans,
        }

    man = json.loads((TV / 'manifest.json').read_text(encoding='utf-8'))
    for f in man['frames']:
        p = TV / 'images' / f['file']
        (w, h), balls, humans = one(p)
        rec['temporal'][f['file']] = {
            'wh': [w, h],
            'gt_ball': read_gt(TV / 'labels' / f'{Path(f["file"]).stem}.txt',
                               w, h).tolist(),   # single-class benchmark
            'proposals': balls,
            'accepted': frozen_accept(balls),
            'humans': humans,
        }

    rec['seconds'] = round(time.time() - t0, 1)
    rec['input_shapes'] = {'x'.join(map(str, k)): v
                           for k, v in sorted(probe.shapes.items())}
    probe.close()
    rec['n_images'] = len(rec['val208']) + len(rec['temporal'])
    return rec


# ------------------------------------------------------------------ metrics

def ap_ball(records):
    """mAP50 / mAP50-95 for the ball class, via Ultralytics' own ap_per_class."""
    from ultralytics.utils.metrics import ap_per_class

    iouv = np.linspace(0.5, 0.95, 10)
    tp, conf = [], []
    n_gt = 0
    for r in records.values():
        gt = np.array(r['gt_ball']).reshape(-1, 4)
        n_gt += len(gt)
        preds = sorted(r['proposals'], key=lambda d: -d['confidence'])
        if not preds:
            continue
        pb = np.array([d['bbox'] for d in preds])
        m = iou_matrix(gt, pb) if len(gt) else np.zeros((0, len(pb)))
        hit = np.zeros((len(preds), len(iouv)), dtype=bool)
        for k, thr in enumerate(iouv):
            taken = set()
            for j in range(len(preds)):          # confidence-descending
                if not len(gt):
                    break
                order = np.argsort(-m[:, j])
                for g in order:
                    if m[g, j] < thr:
                        break
                    if g in taken:
                        continue
                    taken.add(g)
                    hit[j, k] = True
                    break
        tp.append(hit)
        conf.extend(d['confidence'] for d in preds)

    if not tp or not n_gt:
        return {'mAP50': 0.0, 'mAP50_95': 0.0, 'n_gt': n_gt}
    tp = np.concatenate(tp, 0)
    conf = np.array(conf)
    order = np.argsort(-conf)
    res = ap_per_class(tp[order], conf[order], np.zeros(len(conf)),
                       np.zeros(n_gt), plot=False, names={0: 'ball'})
    ap = res[5] if len(res) > 5 else res[-1]
    ap = np.asarray(ap).reshape(1, -1)
    return {'mAP50': round(float(ap[0, 0]), 4),
            'mAP50_95': round(float(ap[0].mean()), 4), 'n_gt': n_gt}


def pr_fp(records, keep=lambda k, r: True):
    """Precision / recall / FP-per-image at the frozen 0.25 operating point."""
    tp = fp = n_gt = n_img = 0
    for k, r in records.items():
        if not keep(k, r):
            continue
        n_img += 1
        gt = np.array(r['gt_ball']).reshape(-1, 4)
        n_gt += len(gt)
        acc = r['accepted']
        h = sum(1 for d in acc if best_iou_gt(gt, d) >= MATCH_IOU)
        h = min(h, len(gt))
        tp += h
        fp += len(acc) - h
    return {'images': n_img, 'gt': n_gt, 'tp': tp, 'fp': fp,
            'precision': round(tp / (tp + fp), 4) if tp + fp else None,
            'recall': round(tp / n_gt, 4) if n_gt else None,
            'fp_per_image': round(fp / n_img, 4) if n_img else None}


def best_iou_gt(gt, det) -> float:
    if not len(gt):
        return 0.0
    return float(iou_matrix(gt, np.array(det['bbox']).reshape(1, 4)).max())


def strata_recall(records):
    out = {n: {'gt': 0, 'tp': 0} for n, _, _ in STRATA}
    for r in records.values():
        gt = np.array(r['gt_ball']).reshape(-1, 4)
        for g in gt:
            s = stratum(float(max(g[2] - g[0], g[3] - g[1])))
            out[s]['gt'] += 1
            out[s]['tp'] += best_iou(g, r['accepted']) >= MATCH_IOU
    for s in out:
        out[s]['recall'] = (round(out[s]['tp'] / out[s]['gt'], 4)
                            if out[s]['gt'] else None)
    return out


def gt_objects(records, man_index):
    """Flat list of temporal GT ball objects with stable ids."""
    objs = []
    for fname, r in records.items():
        for i, g in enumerate(np.array(r['gt_ball']).reshape(-1, 4)):
            objs.append({'id': f'{fname}#{i}', 'file': fname, 'gt_index': i,
                         'bbox': g.tolist(),
                         'max_wh_px': round(float(max(g[2] - g[0], g[3] - g[1])), 2),
                         'window': man_index[fname]['window_key'],
                         'order': man_index[fname]['order_in_window']})
    return objs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    man = json.loads((TV / 'manifest.json').read_text(encoding='utf-8'))
    man_index = {f['file']: {'window_key': f"{f['match']} w{f['window']}",
                             'order_in_window': f['order_in_window'],
                             'shot_type': f['shot_type']}
                 for f in man['frames']}

    report = {'EXPERIMENT': 'R1 inference-resolution isolation',
              'DIAGNOSTIC_ONLY': True,
              'device': DEVICE,
              'thresholds': {'candidate': BALL_CANDIDATE_CONF,
                             'accept': BALL_ACCEPT_CONF,
                             'dedupe_iou': BALL_DEDUPE_IOU,
                             'match_iou': MATCH_IOU,
                             'proposal_floor': PROPOSAL_FLOOR,
                             'predict_iou_arg': NMS_IOU_ARG},
              'checkpoints': {}, 'cells': {}}

    for tag, wts, _ in CELLS:
        if tag not in report['checkpoints']:
            report['checkpoints'][tag] = {'path': str(Path(wts).resolve()),
                                          'sha256': sha256(Path(wts))}

    raw_cells = {}
    for tag, wts, imgsz in CELLS:
        name = f'{tag}@{imgsz}'
        print(f'\n===== {name} =====', flush=True)
        raw_cells[name] = run_cell(wts, imgsz)
        print(f'  {name}: {raw_cells[name]["n_images"]} images in '
              f'{raw_cells[name]["seconds"]}s  shapes '
              f'{raw_cells[name]["input_shapes"]}', flush=True)

    # ---- contact failure set, membership fixed by the reference cell
    ref = raw_cells[REFERENCE_CELL]
    objs = gt_objects(ref['temporal'], man_index)
    contact = []
    for o in objs:
        r = ref['temporal'][o['file']]
        if best_iou(o['bbox'], r['accepted']) >= MATCH_IOU:
            continue                       # detected by production baseline
        if best_iou(o['bbox'], r['proposals']) >= MATCH_IOU:
            continue                       # thresholding, not blindness
        gtb = np.asarray(o['bbox']).reshape(1, 4)
        hb = r['humans']
        ov = (float(iou_matrix(gtb, np.array([d['bbox'] for d in hb])).max())
              if hb else 0.0)
        # PLAYER/GK overlap: the GT ball intersects an accepted person box.
        inter = False
        for d in hb:
            if d['class'] not in ('player', 'goalkeeper'):
                continue
            b = d['bbox']
            if (min(gtb[0, 2], b[2]) > max(gtb[0, 0], b[0])
                    and min(gtb[0, 3], b[3]) > max(gtb[0, 1], b[1])):
                inter = True
                break
        contact.append({**o, 'human_overlap': inter, 'best_human_iou': round(ov, 4)})

    # events = contiguous runs of member frames inside one window
    by_win = defaultdict(list)
    for c in contact:
        by_win[c['window']].append(c)
    events = []
    for wkey, members in sorted(by_win.items()):
        members.sort(key=lambda c: c['order'])
        cur = [members[0]]
        for m in members[1:]:
            if m['order'] - cur[-1]['order'] <= 1:
                cur.append(m)
            else:
                events.append(cur); cur = [m]
        events.append(cur)
    for i, ev in enumerate(events):
        for m in ev:
            m['event'] = i

    report['contact_set'] = {
        'definition': ('temporal-benchmark GT balls that the production frozen '
                       f'path misses at {BALL_ACCEPT_CONF} AND for which no raw '
                       f'proposal exists at >= {PROPOSAL_FLOOR} (IoU {MATCH_IOU}), '
                       f'membership fixed by {REFERENCE_CELL}'),
        'reference_cell': REFERENCE_CELL,
        'n_members': len(contact),
        'n_human_overlap': sum(c['human_overlap'] for c in contact),
        'n_events': len(events),
        'members': contact}

    # ---- per-cell metrics
    for name, rc in raw_cells.items():
        v, t = rc['val208'], rc['temporal']
        cell = {'input_shapes': rc['input_shapes'], 'seconds': rc['seconds'],
                'val208': {**pr_fp(v), **ap_ball(v)},
                'val208_youth': pr_fp(v, lambda k, r: k.startswith('youth_premier_league')),
                'val208_non_youth': pr_fp(v, lambda k, r: not k.startswith('youth_premier_league')),
                'val208_strata': strata_recall(v),
                'temporal': pr_fp(t),
                'temporal_strata': strata_recall(t)}

        # temporal proposal-less population under THIS cell
        miss = noprop = 0
        for o in gt_objects(t, man_index):
            r = t[o['file']]
            if best_iou(o['bbox'], r['accepted']) >= MATCH_IOU:
                continue
            miss += 1
            noprop += best_iou(o['bbox'], r['proposals']) < MATCH_IOU
        cell['temporal_missed'] = miss
        cell['temporal_proposal_less'] = noprop

        # contact set recovery
        det_ids, ev_hit = [], set()
        for c in contact:
            if best_iou(c['bbox'], t[c['file']]['accepted']) >= MATCH_IOU:
                det_ids.append(c['id'])
                ev_hit.add(c['event'])
        cell['contact'] = {
            'detected': len(det_ids),
            'of': len(contact),
            'overlap_detected': sum(1 for c in contact
                                    if c['human_overlap'] and c['id'] in det_ids),
            'overlap_of': sum(c['human_overlap'] for c in contact),
            'events_touched': len(ev_hit),
            'events_of': len(events),
            'detected_ids': det_ids}
        report['cells'][name] = cell

    (out / 'R1_RESULTS.json').write_text(json.dumps(report, indent=1),
                                        encoding='utf-8')
    print(f'\nwritten: {out / "R1_RESULTS.json"}')

    # ---- console summary
    print('\n' + '=' * 96)
    print(f'{"cell":<9}{"P":>8}{"R":>8}{"mAP50":>9}{"mAP5095":>9}{"FP/img":>8}'
          f'{"tRec":>8}{"tPrec":>8}{"FP/fr":>8}{"contact":>9}{"ovl":>7}{"evt":>6}')
    print('-' * 96)
    for name in [f'{t}@{i}' for t, _, i in CELLS]:
        c = report['cells'][name]
        v, tp_ = c['val208'], c['temporal']
        k = c['contact']
        print(f'{name:<9}{v["precision"]:>8.4f}{v["recall"]:>8.4f}'
              f'{v["mAP50"]:>9.4f}{v["mAP50_95"]:>9.4f}{v["fp_per_image"]:>8.4f}'
              f'{tp_["recall"]:>8.4f}{tp_["precision"]:>8.4f}'
              f'{tp_["fp_per_image"]:>8.4f}'
              f'{str(k["detected"])+"/"+str(k["of"]):>9}'
              f'{str(k["overlap_detected"])+"/"+str(k["overlap_of"]):>7}'
              f'{str(k["events_touched"])+"/"+str(k["events_of"]):>6}')
    print('=' * 96)
    return 0


if __name__ == '__main__':
    sys.exit(main())
