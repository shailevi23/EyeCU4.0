#!/usr/bin/env python
"""
Paired, frame-interleaved end-to-end runtime: legacy vs CBIoU.

The previous attempt was inconclusive because the detector -- identical work in
both arms -- differed by 102 ms/frame between them. Rotating whole arms spread
the machine drift but could not cancel it, because the drift was as large as the
measurement.

Here the two arms are paired IN TIME at frame granularity. For every frame both
arms run back to back, and which one goes first alternates by frame parity:

    odd frames   legacy, then cbiou
    even frames  cbiou, then legacy

A slow moment now lands on both arms within milliseconds of each other, and the
alternation stops either arm systematically owning the first-mover position.

Each arm keeps its own tracker state and its own detector, and every detector
call is real: caches are cleared before each call, calls are counted, and the
two arms' detections are compared for equivalence. If a systematic arm-specific
detector bias survives all of that, the run is declared INVALID rather than
explained away.
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

ARMS = ['legacy', 'cbiou']
REPEATS = 3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments/tracking_v2/integration/paired_runtime.json')
    ap.add_argument('--max-frames', type=int, default=0,
                    help='bounded sanity mode: cap frames per sequence')
    ap.add_argument('--repeats', type=int, default=REPEATS)
    args = ap.parse_args()

    import cv2
    from trackers.detector import create_detector
    from trackers.football_tracker import FootballTracker

    gt = REPO / 'data' / 'tracking_val_gt'
    man = json.loads((gt / 'manifest.json').read_text(encoding='utf-8'))
    seqs = sorted(man['sequences'], key=lambda s: s['sequence'])
    frames = {}
    for s in seqs:
        d = gt / 'sequences' / s['sequence'] / 'img1'
        frames[s['sequence']] = [
            cv2.imdecode(np.fromfile(str(d / f'{i:06d}.jpg'), dtype=np.uint8),
                         cv2.IMREAD_COLOR)
            for i in range(1, (args.max_frames or s['frame_count']) + 1)]
    print(f'decoded {sum(len(v) for v in frames.values())} frames once')

    # one detector per arm: a shared one would let arm A warm a cache for arm B
    det = {a: create_detector(model_path=str(REPO / 'best_A_960.pt'),
                              confidence=0.25, imgsz=960) for a in ARMS}

    def run_sequence(s, record=True):
        seq = s['sequence']; n = len(frames[s['sequence']])
        ft = {a: FootballTracker(detector=det[a], persist_cache=False,
                                 tracker_backend=a,
                                 frame_rate=float(s['native_fps']))
              for a in ARMS}
        stats = {a: {'det': [], 'trk': [], 'tot': [], 'calls': 0, 'hits': 0}
                 for a in ARMS}
        mismatched = 0
        for i, img in enumerate(frames[seq]):
            order = ARMS if (i + 1) % 2 == 1 else ARMS[::-1]   # 1-based parity
            dets_seen = {}
            for arm in order:
                d_obj = det[arm]
                # A hit is a call SERVED from cache. The cache being non-empty
                # from the previous frame is not that: it is cleared here before
                # the call, so every invocation recomputes. Counting staleness
                # as a hit is what wrongly invalidated the first paired run.
                d_obj.clear_cache()          # every call is real inference
                if d_obj.detections_cache:
                    stats[arm]['hits'] += 1  # unreachable unless clearing fails
                box = {'t': 0.0, 'dets': None}
                real = d_obj.detect

                def timed(im, fid=None, _r=real, _b=box, *a, **k):
                    t = time.perf_counter()
                    r = _r(im, fid, *a, **k)
                    _b['t'] += time.perf_counter() - t
                    _b['dets'] = r
                    return r

                d_obj.detect = timed
                try:
                    t0 = time.perf_counter()
                    # the real production entry point; self.tracker state
                    # persists across calls, so frame N depends only on this
                    # arm's frames 1..N-1
                    ft[arm].get_object_tracks([img], read_from_cache=False)
                    t2 = time.perf_counter()
                finally:
                    d_obj.detect = real
                stats[arm]['calls'] += 1
                dets_seen[arm] = box['dets'] or []
                stats[arm]['det'].append(1000 * box['t'])
                stats[arm]['trk'].append(1000 * ((t2 - t0) - box['t']))
                stats[arm]['tot'].append(1000 * (t2 - t0))
            a, b = dets_seen['legacy'], dets_seen['cbiou']
            if len(a) != len(b) or any(
                    x['class'] != y['class'] or
                    abs(x['confidence'] - y['confidence']) > 1e-6 or
                    max(abs(p - q) for p, q in zip(x['bbox'], y['bbox'])) > 1e-6
                    for x, y in zip(a, b)):
                mismatched += 1
        if not record:
            return None
        out = {'frames': n, 'detector_mismatched_frames': mismatched}
        for arm in ARMS:
            st = stats[arm]
            out[arm] = {'detector_ms': round(statistics.mean(st['det']), 3),
                        'tracker_ms': round(statistics.mean(st['trk']), 4),
                        'total_ms': round(statistics.mean(st['tot']), 3),
                        'detector_calls': st['calls'], 'cache_hits': st['hits']}
        out['paired_detector_delta_ms'] = [
            round(c - l, 4) for l, c in zip(stats['legacy']['det'], stats['cbiou']['det'])]
        out['paired_total_delta_ms'] = [
            round(c - l, 4) for l, c in zip(stats['legacy']['tot'], stats['cbiou']['tot'])]
        return out

    print('warmup (discarded)')
    run_sequence(seqs[0], record=False)

    raw = []
    for rep in range(args.repeats):
        rot = seqs[rep % len(seqs):] + seqs[:rep % len(seqs)]
        print(f'repeat {rep + 1}  sequence order {[s["sequence"][:10] for s in rot]}')
        for s in rot:
            r = run_sequence(s)
            r['repeat'] = rep + 1
            r['sequence'] = s['sequence']
            raw.append(r)
            print(f"    {s['sequence'][:26]:<28}"
                  f"legacy {r['legacy']['total_ms']:7.2f}  "
                  f"cbiou {r['cbiou']['total_ms']:7.2f}  "
                  f"paired delta {r['cbiou']['total_ms'] - r['legacy']['total_ms']:+6.2f} ms  "
                  f"det mismatch {r['detector_mismatched_frames']}")

    calls_ok = all(r[a]['detector_calls'] == r['frames'] for r in raw for a in ARMS)
    hits_ok = all(r[a]['cache_hits'] == 0 for r in raw for a in ARMS)
    dets_ok = all(r['detector_mismatched_frames'] == 0 for r in raw)
    all_det_delta = [d for r in raw for d in r['paired_detector_delta_ms']]
    det_bias = statistics.mean(all_det_delta)
    det_med = statistics.median(all_det_delta)
    legacy_det = statistics.mean([r['legacy']['detector_ms'] for r in raw])
    bias_pct = 100 * det_bias / legacy_det
    bias_ok = abs(bias_pct) <= 2.0

    tot = {a: statistics.mean([r[a]['total_ms'] for r in raw]) for a in ARMS}
    dm = {a: statistics.mean([r[a]['detector_ms'] for r in raw]) for a in ARMS}
    tm = {a: statistics.mean([r[a]['tracker_ms'] for r in raw]) for a in ARMS}
    paired = statistics.mean([d for r in raw for d in r['paired_total_delta_ms']])
    pct = 100 * paired / (tot['legacy'])

    valid = calls_ok and hits_ok and dets_ok and bias_ok
    common = 100 * (tm['cbiou'] - tm['legacy']) / (legacy_det + tm['legacy'])

    out = {
        'method': 'paired frame-interleaved, arm-first alternating by frame parity',
        'repeats': args.repeats, 'warmup': 1,
        'sequence_order': 'rotated across repeats',
        'independent_detector_per_arm': True,
        'independent_tracker_state_per_arm': True,
        'validity': {
            'detector_calls_match_frames': calls_ok,
            'zero_cache_hits': hits_ok,
            'detector_outputs_equivalent': dets_ok,
            'paired_detector_bias_ms': round(det_bias, 4),
            'paired_detector_bias_median_ms': round(det_med, 4),
            'paired_detector_bias_pct': round(bias_pct, 3),
            'bias_tolerance_pct': 2.0,
            'valid': valid,
        },
        'summary': {a: {'detector_ms_per_frame': round(dm[a], 3),
                        'tracker_ms_per_frame': round(tm[a], 4),
                        'total_ms_per_frame': round(tot[a], 3),
                        'effective_fps': round(1000 / tot[a], 2)} for a in ARMS},
        'paired_total_delta_ms_per_frame': round(paired, 4),
        'paired_pct_vs_legacy': round(pct, 3),
        'common_detector_sanity_check': {
            'label': 'COMMON-DETECTOR SANITY CHECK -- diagnostic only',
            'common_detector_ms': round(legacy_det, 3),
            'legacy_total_ms': round(legacy_det + tm['legacy'], 3),
            'cbiou_total_ms': round(legacy_det + tm['cbiou'], 3),
            'pct': round(common, 3),
            'note': 'does not replace the paired result unless that is INVALID',
        },
        'gate': {'requirement': '<= +10% end-to-end', 'observed_pct': round(pct, 3),
                 'pass': (pct <= 10.0) if valid else None,
                 'status': ('PASS' if valid and pct <= 10.0 else
                            'FAIL' if valid else 'INVALID')},
        'raw': raw,
    }
    Path(REPO / args.out).write_text(json.dumps(out, indent=1), encoding='utf-8')

    print(f'\nvalidity: calls {calls_ok}  hits0 {hits_ok}  dets_equal {dets_ok}  '
          f'detector bias {det_bias:+.2f} ms ({bias_pct:+.2f}%)  -> '
          f'{"VALID" if valid else "INVALID"}')
    print(f'\n{"arm":<8}{"detector":>10}{"tracker":>10}{"total":>10}{"FPS":>8}')
    for a in ARMS:
        print(f'  {a:<8}{dm[a]:>10.2f}{tm[a]:>10.3f}{tot[a]:>10.2f}'
              f'{1000/tot[a]:>8.2f}')
    print(f'\npaired delta {paired:+.3f} ms/frame  ({pct:+.3f}%)')
    print(f'COMMON-DETECTOR SANITY CHECK (diagnostic): {common:+.3f}%')
    print(f'\nRUNTIME GATE: {out["gate"]["status"]}')


if __name__ == '__main__':
    main()
