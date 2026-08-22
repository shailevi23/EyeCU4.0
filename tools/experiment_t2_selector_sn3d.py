#!/usr/bin/env python
"""
T2 -- re-baseline the existing BallTemporalSelector under the SN3D ball branch.

DIAGNOSTIC ONLY. Measures the selector exactly as it currently ships: no
parameter is changed, nothing is tuned, and the selector code is imported rather
than reimplemented. The only thing different from the historical selector
baseline is where the raw candidates come from -- the production SN3D ball
branch instead of the EyeCU ball detector.

Detections come from the REAL production path
(create_detector(ball_detector_backend='sn3d')), not a separate experimental
reimplementation, so what is measured is what ships.

Scored on the frozen 104-frame temporal benchmark under BOTH matching rules:
  centre < 4 px   the rule the SN3D detector gates used (S1B/S1C/S1D)
  IoU >= 0.50     the rule the historical selector artifact used, so the old and
                  new selector numbers are compared like for like

No possession, no team control, no CBIoU, no tracking, no annotation, no image
inspection. VALIDATION ONLY.

    python tools/experiment_t2_selector_sn3d.py --out <json>
"""

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_models import iou_matrix                                  # noqa: E402
from trackers.ball_temporal import (INTERPOLATED, OBSERVED, RECOVERED,  # noqa: E402
                                    UNKNOWN, BallTemporalSelector,
                                    FrameInput, detect_cuts)
from trackers.detector import (BALL_ACCEPT_CONF, SN3D_BALL_SHA256,      # noqa: E402
                               create_detector, resolve_sn3d_ball_path,
                               verify_sn3d_ball_checkpoint)

_s = importlib.util.spec_from_file_location(
    's1', Path(__file__).resolve().parent / 'experiment_s1_ball_specialists.py')
S1 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(S1)

TV = S1.TV
D4 = S1.OFFICIAL_DIST_THRESHOLD
IOU_T = 0.50
HUMAN_MODEL = 'best_A_960.pt'
HUMAN_IMGSZ = 960


def gt_boxes(stem, w, h):
    p = TV / 'labels' / f'{stem}.txt'
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding='utf-8').splitlines():
        q = line.split()
        if len(q) == 5:
            cx, cy, bw, bh = (float(v) for v in q[1:5])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def hit_centre(gt, box):
    if not gt or box is None:
        return False
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(float(np.hypot(cx - (g[0] + g[2]) / 2,
                              cy - (g[1] + g[3]) / 2)) < D4 for g in gt)


def hit_iou(gt, box):
    if not gt or box is None:
        return False
    return float(iou_matrix(np.array(gt).reshape(-1, 4),
                            np.array(box).reshape(1, 4)).max()) >= IOU_T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import cv2

    ball_path = resolve_sn3d_ball_path()
    sha = verify_sn3d_ball_checkpoint(ball_path)
    assert sha == SN3D_BALL_SHA256

    det = create_detector(model_path=HUMAN_MODEL, confidence=BALL_ACCEPT_CONF,
                          imgsz=HUMAN_IMGSZ, ball_candidate_pool=True,
                          ball_detector_backend='sn3d')

    man = json.loads((TV / 'manifest.json').read_text(encoding='utf-8'))
    frames_meta = man['frames']
    by_window = defaultdict(list)
    for f in frames_meta:
        by_window[(f['match'], f['window'])].append(f)
    for v in by_window.values():
        v.sort(key=lambda f: f['order_in_window'])

    # ---- one inference pass through the production branch
    cand, gts, thumbs = {}, {}, {}
    for key, fl in sorted(by_window.items()):
        for f in fl:
            p = TV / 'images' / f['file']
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            cand[f['file']] = [d for d in det.detect(img) if d['class'] == 'ball']
            gts[f['file']] = gt_boxes(Path(f['file']).stem, w, h)
            thumbs[f['file']] = cv2.cvtColor(cv2.resize(img, (64, 36)),
                                             cv2.COLOR_BGR2GRAY)
        print(f'  {key[0]} w{key[1]}: {len(fl)} frames', flush=True)

    # ---- raw detector arm (accepted only), same accept threshold
    raw = {}
    for f in frames_meta:
        acc = [c for c in cand[f['file']] if c['confidence'] >= BALL_ACCEPT_CONF]
        best = max(acc, key=lambda d: d['confidence']) if acc else None
        raw[f['file']] = None if best is None else list(best['bbox'])

    # ---- selector, exactly as it ships
    states, outputs = {}, {}
    cuts_total = 0
    for key, fl in sorted(by_window.items()):
        cuts = detect_cuts([thumbs[f['file']] for f in fl])
        cuts_total += sum(cuts)
        w = cv2.imdecode(np.fromfile(str(TV / 'images' / fl[0]['file']),
                                     dtype=np.uint8), cv2.IMREAD_COLOR).shape[1]
        sel = BallTemporalSelector(frame_width=w)
        fin = []
        for f in fl:
            dt = 1.0 / f['effective_fps']
            fin.append(FrameInput(candidates=[dict(c) for c in cand[f['file']]],
                                  timestamp=f['order_in_window'] * dt, dt=dt,
                                  cut=cuts[len(fin)]))
        for f, o in zip(fl, sel.run(fin)):
            states[f['file']] = o.state
            outputs[f['file']] = o.bbox

    # ---- frozen hard population
    r1 = json.loads(S1.R1.read_text(encoding='utf-8'))
    hard = r1['contact_set']['members']
    hard_files = {m['file'] for m in hard}
    ovl_files = {m['file'] for m in hard if m['human_overlap']}
    ev_of = {m['file']: m['event'] for m in hard}

    def arm(pick, hitfn):
        obs = rec = interp = cov = 0
        halluc = 0
        hits = set()
        for f in frames_meta:
            g = gts[f['file']]
            box = pick(f['file'])
            ok = hitfn(g, box)
            if g:
                if ok:
                    hits.add(f['file'])
                    cov += 1
                    s = states.get(f['file'])
                    obs += (s == OBSERVED)
                    rec += (s == RECOVERED)
                    interp += (s == INTERPOLATED)
            else:
                halluc += box is not None
        ev = {ev_of[x] for x in hits & hard_files}
        return {'total': len(hits), 'observed': obs, 'recovered': rec,
                'interpolated': interp,
                'hard': len(hits & hard_files), 'overlap': len(hits & ovl_files),
                'events': len(ev), 'empty_fired': halluc, 'hits': hits}

    raw_c = arm(lambda k: raw[k], hit_centre)
    sel_c = arm(lambda k: outputs[k], hit_centre)
    raw_i = arm(lambda k: raw[k], hit_iou)
    sel_i = arm(lambda k: outputs[k], hit_iou)

    def preserve(r, s):
        return {'raw_hits': len(r['hits']),
                'preserved': len(r['hits'] & s['hits']),
                'lost_by_selector': sorted(r['hits'] - s['hits']),
                'new_recoveries': sorted(s['hits'] - r['hits']),
                'hard_raw': len(r['hits'] & hard_files),
                'hard_preserved': len((r['hits'] & s['hits']) & hard_files),
                'overlap_raw': len(r['hits'] & ovl_files),
                'overlap_preserved': len((r['hits'] & s['hits']) & ovl_files)}

    report = {
        'EXPERIMENT': 'T2 BallTemporalSelector re-baseline under SN3D',
        'DIAGNOSTIC_ONLY': True, 'parameters_modified': 'NONE',
        'ball_branch': {'backend': 'sn3d', 'path': str(ball_path),
                        'sha256': sha, 'imgsz': 1280,
                        'accept_conf': BALL_ACCEPT_CONF,
                        'detector_type': type(det).__name__},
        'population': {'frames': len(frames_meta),
                       'gt_ball': sum(1 for f in frames_meta if gts[f['file']]),
                       'empty': sum(1 for f in frames_meta if not gts[f['file']]),
                       'hard': len(hard_files), 'overlap': len(ovl_files),
                       'events': len(set(ev_of.values()))},
        'camera_cuts_detected': cuts_total,
        'states': dict(Counter(states.values())),
        'centre_4px': {'raw': {k: v for k, v in raw_c.items() if k != 'hits'},
                       'selector': {k: v for k, v in sel_c.items() if k != 'hits'},
                       'preservation': preserve(raw_c, sel_c)},
        'iou50': {'raw': {k: v for k, v in raw_i.items() if k != 'hits'},
                  'selector': {k: v for k, v in sel_i.items() if k != 'hits'},
                  'preservation': preserve(raw_i, sel_i)},
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding='utf-8')

    for name, blk in (('centre<4px', report['centre_4px']),
                      ('IoU>=0.50', report['iou50'])):
        r, s, p = blk['raw'], blk['selector'], blk['preservation']
        print(f"\n{name}")
        print(f"  raw      total {r['total']}/77 hard {r['hard']}/24 "
              f"ovl {r['overlap']}/20 evt {r['events']}/15 empty {r['empty_fired']}/27")
        print(f"  selector total {s['total']}/77 (obs {s['observed']} rec {s['recovered']} "
              f"interp {s['interpolated']}) hard {s['hard']}/24 ovl {s['overlap']}/20 "
              f"evt {s['events']}/15 empty {s['empty_fired']}/27")
        print(f"  preserved {p['preserved']}/{p['raw_hits']} "
              f"(hard {p['hard_preserved']}/{p['hard_raw']}, "
              f"ovl {p['overlap_preserved']}/{p['overlap_raw']}) "
              f"lost {len(p['lost_by_selector'])} new {len(p['new_recoveries'])}")
    print(f"\nstates: {report['states']}")
    print(f"written: {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
