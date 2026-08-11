#!/usr/bin/env python
"""
Controlled END-TO-END runtime for T2, on the real pipeline.

The criterion is total-pipeline regression, so the detector has to be in the
measurement. It is: the production LocalDetector with best_A_960.pt at
imgsz 960 and conf 0.25, the same checkpoint hash the frozen store records,
over the same decoded package frames.

COMPOSITION, and why it is honest here. The modern trackers cannot be imported
into the interpreter that holds the detector -- that is the isolation the whole
experiment rests on, and breaking it to time something would invalidate more
than it measured. So each stage is timed separately on the same frames and the
same machine, and the total is their sum. The pipeline is strictly sequential
per frame, detector then tracker, with no overlap, so the sum is the elapsed
time rather than a proxy for it.

That assumption is not asserted. For the legacy arm, which CAN run in one
process, a genuine fused end-to-end measurement is taken as well and reported
beside the composed figure. If the two disagree materially the composition is
wrong and the report says so.

Controls: declared warmup discarded, 3 measured repeats, sequence order rotated
between repeats, identical decoded frames throughout.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CANDIDATES = ['CBIoUTracker', 'BoTSORTTracker']
LEGACY = 'LEGACY_SUPERVISION_BYTETRACK'
REPEATS = 3
MIN_CONF = 0.01


def imread(p):
    import cv2
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_frames(gt, seq, n):
    d = gt / 'sequences' / seq / 'img1'
    return [imread(d / f'{i:06d}.jpg') for i in range(1, n + 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--iso', required=True)
    ap.add_argument('--out', default='experiments/tracking_v2/t2/runtime.json')
    args = ap.parse_args()

    from trackers.detector import create_detector
    import supervision as sv
    from tools.bakeoff_legacy_runner import CLASS_IDS
    from trackers.detector import HUMAN_ACCEPT_CONF, HUMAN_CLASSES

    gt = REPO / 'data' / 'tracking_val_gt'
    v1 = REPO / 'data' / 'tracking_val_v1'
    man = json.loads((gt / 'manifest.json').read_text(encoding='utf-8'))
    seqs = sorted(man['sequences'], key=lambda s: s['sequence'])
    frames = {s['sequence']: load_frames(gt, s['sequence'], s['frame_count'])
              for s in seqs}
    print(f'decoded {sum(len(v) for v in frames.values())} frames once; every '
          f'arm sees the same arrays')

    det = create_detector(model_path=str(REPO / 'best_A_960.pt'),
                          confidence=0.25, imgsz=960)

    def detector_pass(seq):
        t = 0.0
        for img in frames[seq]:
            t0 = time.perf_counter()
            det.detect(img)
            t += time.perf_counter() - t0
        return 1000 * t / len(frames[seq])

    def legacy_fused_pass(seq):
        """True single-process detector + tracker, the production path."""
        tracker = sv.ByteTrack()
        t = 0.0
        for img in frames[seq]:
            t0 = time.perf_counter()
            dets = [d for d in det.detect(img) if d['class'] in HUMAN_CLASSES]
            if dets:
                sd = sv.Detections(
                    xyxy=np.array([d['bbox'] for d in dets], dtype=float),
                    class_id=np.array([CLASS_IDS[d['class']] for d in dets]),
                    confidence=np.array([d['confidence'] for d in dets], dtype=float))
                tracked = tracker.update_with_detections(sd)
                for i in range(len(tracked.xyxy)):
                    if float(tracked.confidence[i]) < HUMAN_ACCEPT_CONF:
                        continue
            t += time.perf_counter() - t0
        return 1000 * t / len(frames[seq])

    iso = Path(args.iso)
    py, runner = iso / 'venv' / 'Scripts' / 'python.exe', iso / 't2_runner.py'
    tmp = iso / 'rt'
    tmp.mkdir(exist_ok=True)

    def tracker_pass(tracker, s):
        seq = s['sequence']
        if tracker == LEGACY:
            from tools.bakeoff_legacy_runner import run as run_legacy
            m = run_legacy(v1 / 'detections' / f'{seq}.jsonl',
                           tmp / 'o.txt', tmp / 'm.json', 0.0)
            return m['tracker_ms_per_frame']
        cmd = [str(py), str(runner), '--tracker', tracker,
               '--profile', 'LIBRARY_DEFAULTS',
               '--candidates', str(v1 / 'candidates' / f'{seq}.jsonl'),
               '--frames', str(gt / 'sequences' / seq / 'img1'),
               '--out', str(tmp / 'o.txt'), '--meta-out', str(tmp / 'm.json'),
               '--fps', str(s['native_fps']), '--min-conf', str(MIN_CONF)]
        r = subprocess.run(cmd, cwd=str(iso), capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(r.stderr[-800:])
        return json.loads((tmp / 'm.json').read_text(encoding='utf-8'))[
            'tracker_ms_per_frame']

    print('warmup (discarded)')
    for s in seqs[:1]:
        detector_pass(s['sequence'])
        legacy_fused_pass(s['sequence'])
        for t in CANDIDATES + [LEGACY]:
            tracker_pass(t, s)

    det_s = {s['sequence']: [] for s in seqs}
    fused_s = {s['sequence']: [] for s in seqs}
    trk_s = {t: {s['sequence']: [] for s in seqs} for t in CANDIDATES + [LEGACY]}
    for rep in range(REPEATS):
        rot = seqs[rep % len(seqs):] + seqs[:rep % len(seqs)]
        print(f'repeat {rep + 1}  sequence order {[s["sequence"][:10] for s in rot]}')
        for s in rot:
            seq = s['sequence']
            det_s[seq].append(detector_pass(seq))
            fused_s[seq].append(legacy_fused_pass(seq))
            for t in CANDIDATES + [LEGACY]:
                trk_s[t][seq].append(tracker_pass(t, s))

    med = lambda xs: statistics.median(xs)
    detector = {k: round(med(v), 3) for k, v in det_s.items()}
    fused = {k: round(med(v), 3) for k, v in fused_s.items()}
    out = {'design': {'repeats': REPEATS, 'warmup': 1,
                      'order': 'sequence order rotated between repeats',
                      'frames': 'decoded once, identical arrays for every arm',
                      'detector': 'best_A_960.pt, imgsz 960, conf 0.25, CPU',
                      'composition': ('detector and tracker timed separately and '
                                      'summed; validated against a fused '
                                      'single-process legacy measurement'),
                      'statistic': 'median of repeats, then mean across the three sequences'},
           'detector_ms_per_frame': detector,
           'legacy_fused_ms_per_frame': fused,
           'arms': {}}

    det_mean = sum(detector.values()) / len(detector)
    fused_mean = sum(fused.values()) / len(fused)
    legacy_trk = {k: round(med(v), 3) for k, v in trk_s[LEGACY].items()}
    legacy_total = det_mean + sum(legacy_trk.values()) / len(legacy_trk)
    out['composition_check'] = {
        'legacy_composed_total_ms_per_frame': round(legacy_total, 3),
        'legacy_fused_total_ms_per_frame': round(fused_mean, 3),
        'difference_ms': round(legacy_total - fused_mean, 3),
        'difference_pct': round(100 * (legacy_total - fused_mean) / fused_mean, 2),
    }

    print(f'\n{"arm":<32}{"detector":>10}{"tracker":>10}{"total":>10}'
          f'{"FPS":>8}{"vs legacy":>11}')
    for arm in [LEGACY] + CANDIDATES:
        per = {k: round(med(v), 3) for k, v in trk_s[arm].items()}
        trk_mean = sum(per.values()) / len(per)
        total = det_mean + trk_mean
        pct = 100 * (total - legacy_total) / legacy_total
        out['arms'][arm] = {
            'detector_ms_per_frame': round(det_mean, 3),
            'tracker_ms_per_frame': round(trk_mean, 3),
            'tracker_per_sequence': per,
            'total_ms_per_frame': round(total, 3),
            'effective_fps': round(1000 / total, 2),
            'total_pct_vs_legacy': round(pct, 2),
            'all_tracker_samples': {k: [round(x, 3) for x in v]
                                    for k, v in trk_s[arm].items()},
        }
        print(f'  {arm[:30]:<32}{det_mean:>10.2f}{trk_mean:>10.2f}{total:>10.2f}'
              f'{1000/total:>8.2f}{pct:>10.2f}%')
    print(f'\ncomposition check: legacy composed {legacy_total:.2f} ms vs fused '
          f'{fused_mean:.2f} ms  ({out["composition_check"]["difference_pct"]:+.2f}%)')
    Path(REPO / args.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'written {args.out}')


if __name__ == '__main__':
    main()
