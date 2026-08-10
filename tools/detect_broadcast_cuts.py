#!/usr/bin/env python
"""
Find hard broadcast cuts in a video, from pixels only.

CONTENT RULE, declared before any candidate window is inspected. No tracker
output, no identity, no detection metric of any kind is involved -- a cut is a
property of the footage, and the moment tracker behaviour informs which window
we benchmark on, the benchmark is measuring its own selection.

Two signals, both scale-free, computed between consecutive decoded frames:

    mad     mean absolute difference over the greyscale frame
    hcorr   correlation of 32-bin greyscale histograms

A pan or a tackle moves pixels but keeps the palette, so `mad` rises while
`hcorr` stays near 1. A cut to a different camera or a replay changes both.
Requiring both is what separates a cut from fast motion; either alone would
flag every camera whip in a football broadcast.

A frame pair is a CUT when

    mad >= MAD_CUT  AND  hcorr <= HCORR_CUT

The thresholds are calibrated against a cut the human annotator already
confirmed by eye, then applied unchanged everywhere else. That is calibrating
an instrument against a known event, not tuning until the answer looks right --
and the calibration is reported alongside every result so the reader can see
the margin between a real cut and ordinary motion.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MAD_CUT = 18.0        # mean abs greyscale difference between adjacent frames
HCORR_CUT = 0.80      # histogram correlation; 1.0 is identical distributions
BINS = 32


def frame_signals(prev, cur):
    a = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    mad = float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))
    ha = cv2.calcHist([a], [0], None, [BINS], [0, 256])
    hb = cv2.calcHist([b], [0], None, [BINS], [0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    hcorr = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    return mad, hcorr


def scan_video(video: str, start: int, count: int):
    """Signals for each adjacent pair inside [start, start+count-1]."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f'cannot open {video}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out, prev, idx = [], None, start
    while idx < start + count:
        ok, fr = cap.read()
        if not ok:
            break
        if prev is not None:
            mad, hcorr = frame_signals(prev, fr)
            out.append({'from_frame': idx - 1, 'to_frame': idx,
                        'mad': round(mad, 3), 'hcorr': round(hcorr, 4)})
        prev, idx = fr, idx + 1
    cap.release()
    return out


def cuts_in(signals, mad_cut=MAD_CUT, hcorr_cut=HCORR_CUT):
    return [s for s in signals
            if s['mad'] >= mad_cut and s['hcorr'] <= hcorr_cut]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=300)
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    sig = scan_video(args.video, args.start, args.count)
    cuts = cuts_in(sig)
    mads = np.array([s['mad'] for s in sig])
    hs = np.array([s['hcorr'] for s in sig])
    print(f'{len(sig)} adjacent pairs from frame {args.start}')
    print(f'  mad    median {np.median(mads):.2f}  p95 {np.percentile(mads, 95):.2f}'
          f'  max {mads.max():.2f}')
    print(f'  hcorr  median {np.median(hs):.4f}  p5 {np.percentile(hs, 5):.4f}'
          f'  min {hs.min():.4f}')
    print(f'  thresholds: mad >= {MAD_CUT}  AND  hcorr <= {HCORR_CUT}')
    print(f'\nCUTS: {len(cuts)}')
    for c in cuts:
        print(f'   between source frames {c["from_frame"]} and {c["to_frame"]}'
              f'   mad {c["mad"]:.2f}  hcorr {c["hcorr"]:.4f}')
    if not cuts:
        worst = max(sig, key=lambda s: s['mad'])
        print(f'   (closest pair: {worst["from_frame"]}->{worst["to_frame"]} '
              f'mad {worst["mad"]:.2f} hcorr {worst["hcorr"]:.4f})')
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {'video': args.video, 'start': args.start, 'count': args.count,
             'thresholds': {'mad_cut': MAD_CUT, 'hcorr_cut': HCORR_CUT},
             'cuts': cuts, 'signals': sig}, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
