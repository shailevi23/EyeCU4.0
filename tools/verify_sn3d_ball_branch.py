#!/usr/bin/env python
"""
D2 contract regression for the SN3D ball branch.

Verifies that selecting ball_detector_backend='sn3d' produces exactly the
SN3D_BASE detector behaviour measured in S1B/S1C/S1D, leaves the human branch
bit-identical, keeps the ball out of human association, and satisfies the
canonical detection schema.

Runs the ADAPTER (not a bespoke script) over the already-frozen 104 temporal
frames and 208 static images and compares against the cached S1D BASE records.

No training, no annotation, no image inspection, no threshold changes, no
temporal/possession logic. VALIDATION ONLY.

    python tools/verify_sn3d_ball_branch.py --ball <yolo-sn-ball.pt> --out <json>
"""

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_models import iou_matrix                                  # noqa: E402
from trackers.detector import (BALL_ACCEPT_CONF, CLASS_IDS,            # noqa: E402
                               CLASSES, HUMAN_CLASSES, TwoBranchDetector,
                               create_detector)

_s = importlib.util.spec_from_file_location(
    's1', Path(__file__).resolve().parent / 'experiment_s1_ball_specialists.py')
S1 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(S1)

TV = S1.TV
VAL_IMG = Path('data/dataset_baseline/images/val')
VAL_LAB = Path('data/dataset_baseline/labels/val')
SUBGROUP = Path('EyeCU_S1_results/s1c/S1C_STATIC_OVERLAP_SUBGROUP.json')
S1D = Path('EyeCU_S1_results/s1d/S1D_RESULTS.json')
HUMAN_MODEL = 'best_A_960.pt'     # production human branch, unchanged
HUMAN_IMGSZ = 960
D4 = S1.OFFICIAL_DIST_THRESHOLD
IOU_T = 0.50
BALL_CLS = 3


def load_boxes(path, w, h, cls=None):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding='utf-8').splitlines():
        q = line.split()
        if len(q) != 5:
            continue
        if cls is not None and int(q[0]) != cls:
            continue
        cx, cy, bw, bh = (float(v) for v in q[1:5])
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def match(gt, preds, mode, thr):
    pairs = []
    for gi, g in enumerate(gt):
        gc = ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)
        for pi, d in enumerate(preds):
            b = d['bbox']
            if mode == 'centre':
                v = float(np.hypot((b[0] + b[2]) / 2 - gc[0],
                                   (b[1] + b[3]) / 2 - gc[1]))
                if v < thr:
                    pairs.append((v, gi, pi))
            else:
                v = float(iou_matrix(np.array(g).reshape(1, 4),
                                     np.array(b).reshape(1, 4))[0, 0])
                if v >= thr:
                    pairs.append((-v, gi, pi))
    pairs.sort()
    ug, up, res = set(), set(), set()
    for _v, gi, pi in pairs:
        if gi in ug or pi in up:
            continue
        ug.add(gi)
        up.add(pi)
        res.add(gi)
    return res


def human_key(d):
    """Canonical, order-independent comparison key for a human detection."""
    return (d['class'], tuple(round(v, 6) for v in d['bbox']),
            round(float(d['confidence']), 6), d.get('state'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ball', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import cv2
    rep = {'CHECK': 'D2 SN3D ball-branch contract regression', 'checks': {}}

    # ---- build both modes through the production factory
    eyecu = create_detector(model_path=HUMAN_MODEL, confidence=BALL_ACCEPT_CONF,
                            imgsz=HUMAN_IMGSZ, ball_candidate_pool=True)
    sn3d = create_detector(model_path=HUMAN_MODEL, confidence=BALL_ACCEPT_CONF,
                           imgsz=HUMAN_IMGSZ, ball_candidate_pool=True,
                           ball_detector_backend='sn3d',
                           ball_model_path=args.ball)
    rep['adapter_type'] = type(sn3d).__name__
    rep['is_two_branch'] = isinstance(sn3d, TwoBranchDetector)

    man = json.loads((TV / 'manifest.json').read_text(encoding='utf-8'))
    tframes = man['frames']
    tpaths = [TV / 'images' / f['file'] for f in tframes]
    spaths = sorted(VAL_IMG.iterdir())

    # ---- run both modes once over both populations
    def run(det, paths):
        out, t0 = {}, time.time()
        for p in paths:
            im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            out[p.name] = det.detect(im)
        return out, (time.time() - t0) * 1000 / len(paths)

    det_e_t, ms_e_t = run(eyecu, tpaths)
    det_s_t, ms_s_t = run(sn3d, tpaths)
    det_e_s, ms_e_s = run(eyecu, spaths)
    det_s_s, ms_s_s = run(sn3d, spaths)
    rep['runtime_ms_per_frame'] = {
        'eyecu_temporal': round(ms_e_t, 1), 'sn3d_temporal': round(ms_s_t, 1),
        'eyecu_static': round(ms_e_s, 1), 'sn3d_static': round(ms_s_s, 1)}

    # ---- B. human regression
    diffs, counts = [], {c: {'eyecu': 0, 'sn3d': 0} for c in HUMAN_CLASSES}
    for det_e, det_s in ((det_e_t, det_s_t), (det_e_s, det_s_s)):
        for name in det_e:
            he = sorted((human_key(d) for d in det_e[name]
                         if d['class'] in HUMAN_CLASSES))
            hs = sorted((human_key(d) for d in det_s[name]
                         if d['class'] in HUMAN_CLASSES))
            for d in det_e[name]:
                if d['class'] in HUMAN_CLASSES:
                    counts[d['class']]['eyecu'] += 1
            for d in det_s[name]:
                if d['class'] in HUMAN_CLASSES:
                    counts[d['class']]['sn3d'] += 1
            if he != hs:
                diffs.append(name)
    rep['checks']['human_regression'] = {
        'counts': counts, 'frames_with_any_difference': len(diffs),
        'differing_frames': diffs[:10],
        'identical': not diffs}

    # ---- C. ball never enters CBIoU (the exact production filter)
    ball_reaching_association = 0
    for det_s in (det_s_t, det_s_s):
        for name, dets in det_s.items():
            for d in dets:
                cls = d.get('class')
                if cls not in CLASS_IDS:
                    continue
                if cls not in HUMAN_CLASSES:
                    continue          # football_tracker.py:373-378 drops it here
                if cls == 'ball':
                    ball_reaching_association += 1
    rep['checks']['cbiou_separation'] = {
        'filter': "football_tracker.py:373-378  if class_name not in HUMAN_CLASSES: continue",
        'ball_boxes_passing_filter': ball_reaching_association,
        'ball_enters_cbiou': ball_reaching_association > 0}

    # ---- D. ball contract validation
    bad, multi = [], {0: 0, 1: 0, 'gt1': 0}
    n_ball = 0
    for det_s, paths in ((det_s_t, tpaths), (det_s_s, spaths)):
        for p in paths:
            acc = [d for d in det_s[p.name]
                   if d['class'] == 'ball' and d['confidence'] >= BALL_ACCEPT_CONF]
            im_h, im_w = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8),
                                      cv2.IMREAD_COLOR).shape[:2]
            for d in acc:
                n_ball += 1
                x1, y1, x2, y2 = d['bbox']
                why = []
                if not all(math.isfinite(v) for v in d['bbox']):
                    why.append('non-finite bbox')
                if not x2 > x1:
                    why.append('x2 <= x1')
                if not y2 > y1:
                    why.append('y2 <= y1')
                if not (-1 <= x1 and -1 <= y1 and x2 <= im_w + 1 and y2 <= im_h + 1):
                    why.append('outside frame space')
                if not (math.isfinite(d['confidence']) and 0.0 <= d['confidence'] <= 1.0):
                    why.append('confidence out of range')
                if d['class'] != 'ball' or CLASS_IDS.get(d['class']) != BALL_CLS:
                    why.append('class not canonical ball')
                if why:
                    bad.append({'file': p.name, 'reasons': why})
            k = 0 if not acc else (1 if len(acc) == 1 else 'gt1')
            multi[k] += 1
    rep['checks']['ball_contract'] = {
        'accepted_ball_detections': n_ball,
        'schema': "{'bbox':[x1,y1,x2,y2], 'class':'ball', 'confidence':float} (+ 'state' with pool on)",
        'canonical_class_id': BALL_CLS, 'classes': CLASSES,
        'invalid_records': len(bad), 'examples': bad[:5],
        'frames_with_0_balls': multi[0], 'frames_with_1_ball': multi[1],
        'frames_with_more_than_1_ball': multi['gt1'],
        'valid': not bad}

    # ---- A. SN3D detection regression vs cached S1D BASE
    r1 = json.loads(S1.R1.read_text(encoding='utf-8'))
    hard = r1['contact_set']['members']
    sub = json.loads(SUBGROUP.read_text(encoding='utf-8'))
    smem = {m['id']: m for m in sub['members']}
    s1d = json.loads(S1D.read_text(encoding='utf-8'))

    tgt_c, tgt_b = {}, {}
    for f in tframes:
        im = cv2.imdecode(np.fromfile(str(TV / 'images' / f['file']), dtype=np.uint8),
                          cv2.IMREAD_COLOR)
        h_, w_ = im.shape[:2]
        b = load_boxes(TV / 'labels' / f'{Path(f["file"]).stem}.txt', w_, h_)
        tgt_b[f['file']] = b
        tgt_c[f['file']] = [((x[0] + x[2]) / 2, (x[1] + x[3]) / 2) for x in b]

    pred = {}
    for f in tframes:
        acc = [d for d in det_s_t[f['file']]
               if d['class'] == 'ball' and d['confidence'] >= BALL_ACCEPT_CONF]
        best = max(acc, key=lambda d: d['confidence']) if acc else None
        pred[f['file']] = (None if best is None else {
            'xy': ((best['bbox'][0] + best['bbox'][2]) / 2,
                   (best['bbox'][1] + best['bbox'][3]) / 2),
            'score': best['confidence']})
    tsc = S1.score(pred, tframes, tgt_c, hard, D4)
    tiou = 0
    for f in tframes:
        g = tgt_b[f['file']]
        if g:
            acc = [d for d in det_s_t[f['file']]
                   if d['class'] == 'ball' and d['confidence'] >= BALL_ACCEPT_CONF]
            tiou += len(match(g, acc, 'iou', IOU_T))

    hits4, hitsi = set(), set()
    for p in spaths:
        im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        h_, w_ = im.shape[:2]
        g = load_boxes(VAL_LAB / f'{p.stem}.txt', w_, h_, BALL_CLS)
        if not g:
            continue
        acc = [d for d in det_s_s[p.name]
               if d['class'] == 'ball' and d['confidence'] >= BALL_ACCEPT_CONF]
        for gi in match(g, acc, 'centre', D4):
            hits4.add(f'{p.stem}#{gi}')
        for gi in match(g, acc, 'iou', IOU_T):
            hitsi.add(f'{p.stem}#{gi}')

    G = {'overall': list(smem),
         'youth': [i for i in smem if smem[i]['youth']],
         'non_youth': [i for i in smem if not smem[i]['youth']],
         'overlap': [i for i in smem if smem[i]['overlap_either']],
         'non_overlap': [i for i in smem if not smem[i]['overlap_either']]}

    exp_t = s1d['temporal']['BASE']['centre_4px']
    exp_ti = s1d['temporal']['BASE']['iou50_matched']
    exp_s = s1d['static']['BASE']

    got = {'temporal': {'recall_tp': tsc['tp'], 'hard': tsc['hard_recovered'],
                        'overlap': tsc['overlap_recovered'],
                        'events': tsc['events_touched'],
                        'empty_fired': tsc['empty_fired_frames'],
                        'iou50': tiou,
                        'hard_files': sorted(tsc['hard_recovered_files'])},
           'static': {g: {'centre_4px': len(set(ids) & hits4),
                          'iou50': len(set(ids) & hitsi)}
                      for g, ids in G.items()}}
    exp = {'temporal': {'recall_tp': exp_t['tp'], 'hard': exp_t['hard_recovered'],
                        'overlap': exp_t['overlap_recovered'],
                        'events': exp_t['events_touched'],
                        'empty_fired': exp_t['empty_fired_frames'],
                        'iou50': exp_ti,
                        'hard_files': sorted(exp_t['hard_recovered_files'])},
           'static': {g: {'centre_4px': exp_s[g]['centre_4px'],
                          'iou50': exp_s[g]['iou50']} for g in G}}
    rep['checks']['sn3d_regression'] = {
        'expected_source': str(S1D), 'expected': exp, 'got': got,
        'matches': got == exp}

    passed = (rep['checks']['sn3d_regression']['matches']
              and rep['checks']['human_regression']['identical']
              and not rep['checks']['cbiou_separation']['ball_enters_cbiou']
              and rep['checks']['ball_contract']['valid']
              and rep['is_two_branch'])
    rep['PASS'] = bool(passed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps({k: (v if k != 'checks' else
                          {n: (c.get('identical', c.get('matches',
                               c.get('valid', c.get('ball_enters_cbiou')))))
                           for n, c in v.items()})
                      for k, v in rep.items() if k != 'runtime_ms_per_frame'},
                     indent=1))
    print('regression matches:', rep['checks']['sn3d_regression']['matches'])
    print('PASS:', rep['PASS'])
    print('written:', args.out)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
