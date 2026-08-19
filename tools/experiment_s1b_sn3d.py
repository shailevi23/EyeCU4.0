#!/usr/bin/env python
"""
S1B -- official SoccerNet-v3D yolo-sn-ball.pt against EyeCU B, apples to apples.

ONE model run, on the frozen 104-frame S1 temporal population, scored by the
identical common metric and the identical single-best selection policy used for
A@960 / B@1280 / B@1920 in S1. EyeCU baselines are READ from the saved S1 JSON
and never rerun.

Raw detector only: no SAHI, no TTA, no augment, no tiling, no tracking, no
BallTemporalSelector, no interpolation, no temporal fusion.

Selection policy, byte-identical to the S1 YOLO baselines:
    predict at floor BALL_CANDIDATE_CONF -> keep ball >= BALL_CANDIDATE_CONF
    -> suppress_ball_duplicates(BALL_DEDUPE_IOU) -> keep >= BALL_ACCEPT_CONF
    -> single best by confidence -> centre = bbox centre

VALIDATION ONLY. Sealed TEST is unreachable from this file.

    python tools/experiment_s1b_sn3d.py --model <yolo-sn-ball.pt> --out <json>
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_models import iou_matrix                                 # noqa: E402
from trackers.detector import (BALL_ACCEPT_CONF, BALL_CANDIDATE_CONF,  # noqa: E402
                               BALL_DEDUPE_IOU, LocalDetector,
                               suppress_ball_duplicates)

_spec = importlib.util.spec_from_file_location(
    's1', Path(__file__).resolve().parent / 'experiment_s1_ball_specialists.py')
S1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S1)

TV = S1.TV
S1_JSON = Path('EyeCU_S1_results/s1_final/S1_RESULTS.json')
IOU_MATCH = 0.50
INFER_IMGSZ = 1280
NMS_IOU = 0.5


def gt_boxes(stem: str, w: int, h: int):
    p = TV / 'labels' / f'{stem}.txt'
    if not p.exists():
        return np.empty((0, 4))
    out = []
    for line in p.read_text(encoding='utf-8').splitlines():
        q = line.split()
        if len(q) == 5:
            cx, cy, bw, bh = (float(v) for v in q[1:5])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out).reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import cv2

    mp = Path(args.model)
    man = json.loads((TV / 'manifest.json').read_text(encoding='utf-8'))
    frames_meta = man['frames']

    # frozen population, loaded verbatim
    s1 = json.loads(S1_JSON.read_text(encoding='utf-8'))
    hard = s1['hard_set'] if isinstance(s1.get('hard_set'), list) else None
    r1 = json.loads(S1.R1.read_text(encoding='utf-8'))
    hard = r1['contact_set']['members']

    gt_c, gt_b = {}, {}
    for f in frames_meta:
        img = cv2.imdecode(np.fromfile(str(TV / 'images' / f['file']),
                                       dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        gt_c[f['file']] = S1.load_gt_centre(Path(f['file']).stem, w, h)
        gt_b[f['file']] = gt_boxes(Path(f['file']).stem, w, h)

    det = LocalDetector(str(mp), confidence=BALL_CANDIDATE_CONF, iou=NMS_IOU,
                        imgsz=INFER_IMGSZ, device='cpu', ball_candidate_pool=False)

    single, accepted_all = {}, {}
    t0 = time.time()
    for f in frames_meta:
        img = cv2.imdecode(np.fromfile(str(TV / 'images' / f['file']),
                                       dtype=np.uint8), cv2.IMREAD_COLOR)
        balls = [d for d in det.detect(img) if d['class'] == 'ball']
        kept = suppress_ball_duplicates(
            [b for b in balls if b['confidence'] >= BALL_CANDIDATE_CONF],
            BALL_DEDUPE_IOU)
        acc = [b for b in kept if b['confidence'] >= BALL_ACCEPT_CONF]
        accepted_all[f['file']] = acc
        best = max(acc, key=lambda d: d['confidence']) if acc else None
        single[f['file']] = (None if best is None else {
            'xy': ((best['bbox'][0] + best['bbox'][2]) / 2,
                   (best['bbox'][1] + best['bbox'][3]) / 2),
            'score': best['confidence']})
    ms = (time.time() - t0) * 1000 / len(frames_meta)

    report = {
        'EXPERIMENT': 'S1B official SoccerNet-v3D yolo-sn-ball.pt',
        'DIAGNOSTIC_ONLY': True, 'device': 'cpu',
        'checkpoint': {'file': mp.name, 'abspath': str(mp.resolve()),
                       'sha256': hashlib.sha256(mp.read_bytes()).hexdigest(),
                       'bytes': mp.stat().st_size},
        'inference': {'imgsz': INFER_IMGSZ, 'conf_accept': BALL_ACCEPT_CONF,
                      'conf_floor': BALL_CANDIDATE_CONF,
                      'predict_iou_arg': NMS_IOU,
                      'dedupe_iou': BALL_DEDUPE_IOU,
                      'selection': 'single best accepted ball by confidence',
                      'ms_per_frame_cpu': round(ms, 1)},
        'population': {'targets': len(frames_meta),
                       'gt_ball_frames': sum(1 for f in frames_meta if gt_c[f['file']]),
                       'empty_frames': sum(1 for f in frames_meta if not gt_c[f['file']]),
                       'hard': len(hard),
                       'overlap': sum(bool(m['human_overlap']) for m in hard),
                       'events': len({m['event'] for m in hard})},
        'primary_4px': S1.score(single, frames_meta, gt_c, hard,
                                S1.OFFICIAL_DIST_THRESHOLD),
        'strict_1p333px': S1.score(single, frames_meta, gt_c, hard,
                                   S1.NORMALISED_DIST_THRESHOLD),
    }

    # ---- secondary: ANY accepted detection within 4 px, and extra predictions
    any_hit = extra = 0
    any_hit_files = set()
    for f in frames_meta:
        g = gt_c[f['file']]
        acc = accepted_all[f['file']]
        if not g:
            continue
        extra += max(0, len(acc) - 1)
        for d in acc:
            cx = (d['bbox'][0] + d['bbox'][2]) / 2
            cy = (d['bbox'][1] + d['bbox'][3]) / 2
            if min(float(np.hypot(cx - c[0], cy - c[1])) for c in g) < S1.OFFICIAL_DIST_THRESHOLD:
                any_hit += 1
                any_hit_files.add(f['file'])
                break
    n_pos = report['population']['gt_ball_frames']
    ev = defaultdict(int)
    for m in hard:
        if m['file'] in any_hit_files:
            ev[m['event']] += 1
    report['multi_detection_diagnostic'] = {
        'single_best_tp': report['primary_4px']['tp'],
        'any_detection_tp': any_hit,
        'any_detection_recall': round(any_hit / n_pos, 4) if n_pos else None,
        'extra_accepted_predictions_on_positive_frames': extra,
        'any_detection_hard': sum(1 for m in hard if m['file'] in any_hit_files),
        'any_detection_overlap': sum(1 for m in hard if m['human_overlap']
                                     and m['file'] in any_hit_files),
        'any_detection_events': len(ev)}

    # ---- historical reproduction audit: conf>=0.25, IoU>=0.50
    iou_tp = 0
    for f in frames_meta:
        g = gt_b[f['file']]
        acc = accepted_all[f['file']]
        if not len(g) or not acc:
            continue
        pb = np.array([d['bbox'] for d in acc])
        m = iou_matrix(g, pb)
        iou_tp += int((m.max(axis=1) >= IOU_MATCH).sum())
    report['historical_iou_audit'] = {
        'criterion': f'conf >= {BALL_ACCEPT_CONF}, IoU >= {IOU_MATCH}',
        'matched': iou_tp, 'of': n_pos,
        'recall': round(iou_tp / n_pos, 4) if n_pos else None,
        'historical_reference': '57/77 = 0.7403'}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding='utf-8')

    p = report['primary_4px']
    s = report['strict_1p333px']
    print(f"SN3D@1280 4px  : {p['tp']}/{p['gt_frames']} R {p['recall']} "
          f"P {p['precision']} FP1 {p['fp1']} FP2 {p['fp2']} "
          f"empty {p['empty_fired_frames']}/{p['empty_frames']} "
          f"hard {p['hard_recovered']}/{p['hard_of']} "
          f"ovl {p['overlap_recovered']}/{p['overlap_of']} "
          f"evt {p['events_touched']}/{p['events_of']} e2+ {p['events_with_2plus']}")
    print(f"SN3D@1280 1.333: {s['tp']}/{s['gt_frames']} R {s['recall']} "
          f"hard {s['hard_recovered']}/{s['hard_of']} "
          f"ovl {s['overlap_recovered']}/{s['overlap_of']} "
          f"evt {s['events_touched']}/{s['events_of']}")
    print(f"any-det: {report['multi_detection_diagnostic']}")
    print(f"IoU audit: {report['historical_iou_audit']}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
