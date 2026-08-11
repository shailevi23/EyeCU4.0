#!/usr/bin/env python
"""
Real integrated end-to-end runtime: legacy vs CBIoU, through FootballTracker.

Not composed. Both arms run the actual production object in one process --
the same detector, the same detection-preparation code, the same ball path,
the same track bookkeeping -- and differ only in which association backend
FootballTracker was constructed with. That is the comparison the criterion
asks for.

Frames are decoded once and shared, so decode cost is measured separately and
is identical for both arms rather than being attributed to either. Warmup is
discarded, three repeats are measured, and arm order is rotated between them so
a machine drifting during the run cannot be mistaken for a slow tracker.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

REPEATS = 3
ARMS = ['legacy', 'cbiou']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments/tracking_v2/integration/runtime.json')
    args = ap.parse_args()

    import cv2
    from trackers.detector import create_detector
    from trackers.football_tracker import FootballTracker

    gt = REPO / 'data' / 'tracking_val_gt'
    man = json.loads((gt / 'manifest.json').read_text(encoding='utf-8'))
    seqs = sorted(man['sequences'], key=lambda s: s['sequence'])

    decode = {}
    frames = {}
    for s in seqs:
        seq, n = s['sequence'], s['frame_count']
        d = gt / 'sequences' / seq / 'img1'
        t0 = time.perf_counter()
        imgs = [cv2.imdecode(np.fromfile(str(d / f'{i:06d}.jpg'), dtype=np.uint8),
                             cv2.IMREAD_COLOR) for i in range(1, n + 1)]
        decode[seq] = 1000 * (time.perf_counter() - t0) / n
        frames[seq] = imgs
    print(f'decoded {sum(len(v) for v in frames.values())} frames once; '
          f'both arms see the same arrays')

    detector = create_detector(model_path=str(REPO / 'best_A_960.pt'),
                               confidence=0.25, imgsz=960)

    def one(arm, s):
        """
        The real production entry point, end to end.

        get_object_tracks runs detection and association together, exactly as
        the pipeline calls it. Detection time is measured inside the same run
        by wrapping the detector, so the split is real rather than inferred and
        the total is genuine wall time, not a sum of separate runs.
        """
        seq = s['sequence']
        # detect_objects_in_frames passes a frame_id, so the detector's own
        # cache would serve the SECOND arm for free and make the comparison
        # meaningless. Every measured run starts cold.
        detector.clear_cache()
        ft = FootballTracker(detector=detector, persist_cache=False,
                             tracker_backend=arm,
                             frame_rate=float(s['native_fps']))
        det_t = 0.0
        real_detect = detector.detect

        def timed_detect(img, *a, **k):
            nonlocal det_t
            t0 = time.perf_counter()
            r = real_detect(img, *a, **k)
            det_t += time.perf_counter() - t0
            return r

        detector.detect = timed_detect
        try:
            t0 = time.perf_counter()
            ft.get_object_tracks(frames[seq], read_from_cache=False)
            total = time.perf_counter() - t0
        finally:
            detector.detect = real_detect
        n = len(frames[seq])
        return 1000 * det_t / n, 1000 * (total - det_t) / n

    print('warmup (discarded)')
    for arm in ARMS:
        one(arm, seqs[0])

    samples = {a: {s['sequence']: [] for s in seqs} for a in ARMS}
    for rep in range(REPEATS):
        rot = ARMS[rep % len(ARMS):] + ARMS[:rep % len(ARMS)]
        print(f'repeat {rep + 1}  arm order {rot}')
        for arm in rot:
            for s in seqs:
                samples[arm][s['sequence']].append(one(arm, s))

    med = statistics.median
    decode_mean = sum(decode.values()) / len(decode)
    out = {'design': {'repeats': REPEATS, 'warmup': 1,
                      'order': 'arm order rotated between repeats',
                      'frames': 'decoded once, shared by both arms',
                      'composed': False,
                      'note': ('both arms are the same FootballTracker object '
                               'differing only in tracker_backend'),
                      'detector_cache': ('cleared before every measured run; '
                                         'production passes a frame_id, so a '
                                         'warm cache would have served the '
                                         'second arm for free')},
           'decode_ms_per_frame': round(decode_mean, 3),
           'arms': {}}
    print(f'\n{"arm":<10}{"decode":>9}{"detector":>10}{"tracker":>9}'
          f'{"total":>9}{"FPS":>8}{"vs legacy":>11}')
    totals = {}
    for arm in ARMS:
        det_ms = sum(med([x[0] for x in samples[arm][s['sequence']]])
                     for s in seqs) / len(seqs)
        trk_ms = sum(med([x[1] for x in samples[arm][s['sequence']]])
                     for s in seqs) / len(seqs)
        total = det_ms + trk_ms
        totals[arm] = total
        out['arms'][arm] = {
            'detector_ms_per_frame': round(det_ms, 3),
            'tracker_ms_per_frame': round(trk_ms, 3),
            'total_ms_per_frame': round(total, 3),
            'total_with_decode_ms_per_frame': round(total + decode_mean, 3),
            'effective_fps': round(1000 / total, 2),
            'per_sequence': {s['sequence']: {
                'detector': round(med([x[0] for x in samples[arm][s['sequence']]]), 3),
                'tracker': round(med([x[1] for x in samples[arm][s['sequence']]]), 3)}
                for s in seqs},
        }
    for arm in ARMS:
        pct = 100 * (totals[arm] - totals['legacy']) / totals['legacy']
        out['arms'][arm]['total_pct_vs_legacy'] = round(pct, 3)
        a = out['arms'][arm]
        print(f'  {arm:<10}{decode_mean:>9.2f}{a["detector_ms_per_frame"]:>10.2f}'
              f'{a["tracker_ms_per_frame"]:>9.2f}{a["total_ms_per_frame"]:>9.2f}'
              f'{a["effective_fps"]:>8.2f}{pct:>10.2f}%')
    reg = out['arms']['cbiou']['total_pct_vs_legacy']
    out['gate'] = {'requirement': 'total end-to-end regression <= +10%',
                   'observed_pct': reg, 'pass': reg <= 10.0}
    print(f'\nRUNTIME GATE (<= +10%): {"PASS" if reg <= 10.0 else "FAIL"}  '
          f'observed {reg:+.2f}%')
    Path(REPO / args.out).write_text(json.dumps(out, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
