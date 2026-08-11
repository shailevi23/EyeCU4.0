#!/usr/bin/env python
"""
Prove the integrated CBIoU path reproduces the isolated T2 reference exactly.

This is the integration gate. The vendored copy runs in the PRODUCTION
environment -- different supervision version, same numpy -- so nothing about
equivalence is safe to assume. Every frame is compared: identity, box, row
count, ordering after canonical normalisation, and the frame span of every
track.

Discrepancies are reported, never smoothed. If output differs, the correct
response is to stop and explain it, not to adjust anything until it matches.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

HUMAN = {'player', 'goalkeeper', 'referee'}
MIN_CONF = 0.01          # exactly the T2 / LIBRARY_DEFAULTS supply rule


def run_integrated(candidates: Path, frames: Path, fps: float):
    """The production path: vendored CBIoU at exact library defaults."""
    import cv2
    import supervision as sv
    from rf_trackers import CBIoUTracker

    tracker = CBIoUTracker(frame_rate=float(fps))
    rows = [json.loads(l) for l in candidates.read_text(encoding='utf-8').splitlines()
            if l.strip()]
    out = []
    for rec in rows:
        f = rec['frame'] + 1          # frozen store is 0-based; MOT is 1-based
        dets = [d for d in rec['detections']
                if d['class'] in HUMAN and d['confidence'] > MIN_CONF]
        xyxy = (np.array([d['bbox'] for d in dets], dtype=np.float32) if dets
                else np.empty((0, 4), dtype=np.float32))
        conf = (np.array([d['confidence'] for d in dets], dtype=np.float32) if dets
                else np.empty((0,), dtype=np.float32))
        sd = sv.Detections(xyxy=xyxy, confidence=conf,
                           class_id=np.zeros(len(conf), dtype=int))
        img = cv2.imdecode(np.fromfile(str(frames / f'{f:06d}.jpg'), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        res = tracker.update(sd, frame=img)
        if res is None or not len(res) or res.tracker_id is None:
            continue
        for i in range(len(res)):
            tid = res.tracker_id[i]
            if tid is None or (isinstance(tid, float) and np.isnan(tid)) or int(tid) < 0:
                continue
            tid = int(tid) + 1        # 0-based raw -> positive public id
            x1, y1, x2, y2 = (float(v) for v in res.xyxy[i])
            c = float(res.confidence[i]) if res.confidence is not None else 1.0
            out.append(f'{f},{int(tid)},{x1:.2f},{y1:.2f},'
                       f'{x2-x1:.2f},{y2-y1:.2f},{c:.4f},1,1')
    return out


def parse(lines):
    rows = []
    for l in lines:
        if not l.strip():
            continue
        p = l.split(',')
        rows.append((int(p[0]), int(p[1]), float(p[2]), float(p[3]),
                     float(p[4]), float(p[5]), float(p[6])))
    return rows


def compare(ref, got):
    """Canonical normalisation: sort by (frame, id). Order is not semantic."""
    r, g = sorted(parse(ref)), sorted(parse(got))
    issues = []
    if len(r) != len(g):
        issues.append(f'row count {len(g)} vs reference {len(r)}')
    rk = {(x[0], x[1]) for x in r}
    gk = {(x[0], x[1]) for x in g}
    if rk != gk:
        issues.append(f'{len(rk - gk)} (frame,id) only in reference, '
                      f'{len(gk - rk)} only in integrated')
    worst = 0.0
    byk = {(x[0], x[1]): x for x in r}
    for x in g:
        y = byk.get((x[0], x[1]))
        if y is None:
            continue
        worst = max(worst, max(abs(a - b) for a, b in zip(x[2:6], y[2:6])))
    if worst > 0:
        issues.append(f'max box coordinate difference {worst}')

    def life(rows):
        d = defaultdict(list)
        for x in rows:
            d[x[1]].append(x[0])
        return {i: (min(v), max(v), len(v)) for i, v in d.items()}
    if life(r) != life(g):
        lr, lg = life(r), life(g)
        diff = {i: (lr.get(i), lg.get(i)) for i in set(lr) | set(lg)
                if lr.get(i) != lg.get(i)}
        issues.append(f'track lifecycle differs for {len(diff)} ids: '
                      f'{dict(list(diff.items())[:3])}')
    byte_identical = list(ref) == list(got)
    return issues, {'reference_rows': len(r), 'integrated_rows': len(g),
                    'max_box_delta': worst,
                    'identical_after_normalisation': not issues,
                    'byte_identical_unsorted': byte_identical}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', default='experiments/tracking_v2/t2/outputs/EyeCU-val/CBIoUTracker/data')
    ap.add_argument('--out', default='experiments/tracking_v2/integration')
    args = ap.parse_args()

    gt = REPO / 'data' / 'tracking_val_gt'
    v1 = REPO / 'data' / 'tracking_val_v1'
    man = json.loads((gt / 'manifest.json').read_text(encoding='utf-8'))
    seqs = sorted(man['sequences'], key=lambda s: s['sequence'])
    out = REPO / args.out
    report = {'reference': args.reference, 'sequences': {}}

    print(f'{"sequence":<32}{"ref rows":>10}{"int rows":>10}{"max delta":>11}  verdict')
    all_ok = True
    for s in seqs:
        seq = s['sequence']
        ref = (REPO / args.reference / f'{seq}.txt').read_text(encoding='utf-8').splitlines()
        got = run_integrated(v1 / 'candidates' / f'{seq}.jsonl',
                             gt / 'sequences' / seq / 'img1', s['native_fps'])
        dst = out / 'outputs' / 'EyeCU-val' / 'CBIoU_INTEGRATED' / 'data' / f'{seq}.txt'
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text('\n'.join(got) + ('\n' if got else ''), encoding='utf-8')
        issues, stats = compare(ref, got)
        all_ok &= not issues
        report['sequences'][seq] = {'issues': issues, **stats}
        print(f'  {seq[:30]:<32}{stats["reference_rows"]:>10}'
              f'{stats["integrated_rows"]:>10}{stats["max_box_delta"]:>11.4f}  '
              f'{"EXACT" if not issues else "DIFFERS"}')
        for i in issues:
            print(f'      {i}')

    report['exact_equivalence'] = all_ok
    out.mkdir(parents=True, exist_ok=True)
    (out / 'equivalence.json').write_text(json.dumps(report, indent=1),
                                          encoding='utf-8')
    print(f'\nEXACT OUTPUT EQUIVALENCE: {"PASS" if all_ok else "FAIL"}')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
