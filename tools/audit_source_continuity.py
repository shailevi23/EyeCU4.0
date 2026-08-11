#!/usr/bin/env python
"""
Audit whole source videos for broadcast structure. Pixels only.

The question is whether a video is a continuous broadcast or a highlights/recap
cut, because a benchmark window that spans an edit cannot carry identity across
it -- and no amount of careful annotation fixes that.

No tracker output, no identity, no detection metric. Selecting benchmark
footage on tracker behaviour would make the benchmark a measurement of its own
selection.

TWO SIGNALS, both between frame t and frame t+LAG:

    mad     mean absolute greyscale difference
    hcorr   correlation of 32-bin greyscale histograms

LAG = 8 rather than 1 deliberately. Modern football broadcasts join shots with
a DISSOLVE, spreading the change over several frames so that no adjacent pair
looks catastrophic -- a lag-1 hard-cut rule sails straight past one. Over a
third of a second, ordinary play barely changes the palette while a scene
change replaces it.

    a frame is DISCONTINUOUS when hcorr(t, t+LAG) <= HCORR_MIN

Calibrated against a transition the human annotator confirmed by eye and the
broadcast clock corroborates (Austin src 393->394, clock 29:0x -> 30:35): the
dissolve sits at hcorr 0.80-0.85 while steady play three frames later is at
0.995. The threshold is then applied UNCHANGED to every source, so the
comparison between sources is the evidence, not any one absolute number.

Reported per source: how many discontinuities, and -- the number that decides
eligibility -- the longest continuous run.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LAG = 8
HCORR_MIN = 0.90
BINS = 32
NEED_FRAMES = 300


def hist(gray):
    h = cv2.calcHist([gray], [0], None, [BINS], [0, 256])
    cv2.normalize(h, h)
    return h


def scan(video: str, max_frames=None):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f'cannot open {video}')
    ring, out, idx = [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ring.append((idx, g, hist(g)))
        if len(ring) > LAG:
            i0, g0, h0 = ring.pop(0)
            out.append({
                'frame': i0,
                'mad': round(float(np.mean(np.abs(
                    g0.astype(np.int16) - ring[-1][1].astype(np.int16)))), 3),
                'hcorr': round(float(cv2.compareHist(
                    h0, ring[-1][2], cv2.HISTCMP_CORREL)), 4),
            })
        idx += 1
    cap.release()
    return out, idx


def runs_of_continuity(sig, total, hcorr_min=HCORR_MIN):
    """Maximal stretches containing no discontinuous frame."""
    bad = sorted(s['frame'] for s in sig if s['hcorr'] <= hcorr_min)
    runs, start = [], 0
    for b in bad:
        # the transition itself spans [b, b+LAG]; the run ends where it begins
        if b > start:
            runs.append((start, b - 1))
        start = max(start, b + LAG + 1)
    if start < total:
        runs.append((start, total - 1))
    return [(a, z, z - a + 1) for a, z in runs if z >= a], bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default='data/tracking_val_gt/manifest.json')
    ap.add_argument('--videos', nargs='*', default=None)
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    if args.videos:
        vids = [(Path(v).stem, v) for v in args.videos]
    else:
        man = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
        vids = [(s['match'], s['source_video']) for s in man['sequences']]

    report = []
    for name, path in vids:
        sig, total = scan(path)
        runs, bad = runs_of_continuity(sig, total)
        hs = np.array([s['hcorr'] for s in sig])
        longest = max((r[2] for r in runs), default=0)
        eligible = [r for r in runs if r[2] >= NEED_FRAMES]
        report.append({
            'match': name, 'video': path, 'frames': total,
            'discontinuities': len(bad),
            'per_1000_frames': round(1000 * len(bad) / max(1, total), 2),
            'hcorr_median': round(float(np.median(hs)), 4),
            'hcorr_p05': round(float(np.percentile(hs, 5)), 4),
            'longest_continuous_run': longest,
            'runs_ge_300': len(eligible),
            'first_runs': runs[:6],
            'discontinuity_frames': bad[:40],
        })
        print(f'\n=== {name}   {total} frames')
        print(f'    discontinuities {len(bad)}  '
              f'({1000*len(bad)/max(1,total):.2f} per 1000 frames)')
        print(f'    hcorr(lag {LAG})  median {np.median(hs):.4f}  '
              f'p5 {np.percentile(hs, 5):.4f}')
        print(f'    longest continuous run {longest} frames   '
              f'runs >= {NEED_FRAMES}: {len(eligible)}')
        if bad:
            print(f'    first discontinuities at {bad[:12]}')

    print(f'\n{"match":<32}{"frames":>8}{"cuts":>7}{"per1k":>8}'
          f'{"longest":>9}{"runs>=300":>11}')
    for r in report:
        print(f'  {r["match"][:30]:<32}{r["frames"]:>8}{r["discontinuities"]:>7}'
              f'{r["per_1000_frames"]:>8}{r["longest_continuous_run"]:>9}'
              f'{r["runs_ge_300"]:>11}')

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {'lag': LAG, 'hcorr_min': HCORR_MIN, 'need_frames': NEED_FRAMES,
             'sources': report}, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
