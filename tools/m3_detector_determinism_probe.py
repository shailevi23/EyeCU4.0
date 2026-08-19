#!/usr/bin/env python
"""
M3 -- second half of the CBIoU-nondeterminism isolation. The CBIoU-only probe
(tools/m3_cbiou_determinism_probe.py) found the tracker byte-identical given
byte-identical input, across all 204 frames of youth_premier_league_1133.
So: does the DETECTOR itself produce different raw output across two fresh
runs of the same frames? This is the remaining candidate source.

READ-ONLY diagnostic. Does not change any detector threshold or weight.
"""

import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackers.detector import BALL_ACCEPT_CONF                          # noqa: E402
from trackers.football_tracker import FootballTracker                   # noqa: E402

SEQ = Path('data/tracking_val_gt/sequences/youth_premier_league_1133/img1')


def run_detector_once(imgs):
    tracker = FootballTracker(model_path='best_A_960.pt', imgsz=960,
                              confidence=BALL_ACCEPT_CONF, persist_cache=False,
                              ball_candidate_pool=True, ball_detector_backend='sn3d')
    tracker.detector.clear_cache()
    dets = tracker.detect_objects_in_frames(imgs)
    # round for stable JSON/text comparison of true float noise vs formatting
    out = []
    for frame in dets:
        out.append(sorted(
            [{'class': d.get('class'), 'bbox': [round(float(v), 6) for v in d['bbox']],
              'confidence': round(float(d.get('confidence', 0.0)), 6)}
             for d in frame],
            key=lambda d: (d['class'], d['bbox'])))
    return out


def main():
    frame_ids = list(range(1, 205))
    imgs = [cv2.imdecode(np.fromfile(str(SEQ / f'{f:06d}.jpg'), dtype=np.uint8),
                         cv2.IMREAD_COLOR) for f in frame_ids]

    print('detector pass 1...', flush=True)
    run_a = run_detector_once(imgs)
    print('detector pass 2...', flush=True)
    run_b = run_detector_once(imgs)

    diffs = []
    for i, (fa, fb) in enumerate(zip(run_a, run_b)):
        if fa != fb:
            diffs.append({'frame_idx': i, 'n_dets_a': len(fa), 'n_dets_b': len(fb),
                          'run_a': fa, 'run_b': fb})

    result = {'sequence': 'youth_premier_league_1133', 'n_frames': len(imgs),
              'n_frames_differing': len(diffs), 'record_identical': len(diffs) == 0,
              'first_3_diffs': diffs[:3]}
    out_path = Path('experiments/records/experiment_M3/detector_determinism_probe_result.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'first_3_diffs'}, indent=1))
    if diffs:
        print('FIRST DIFF:', json.dumps(diffs[0], indent=1, default=str)[:3000])
    print('written:', out_path)


if __name__ == '__main__':
    raise SystemExit(main())
