#!/usr/bin/env python
"""
DIAGNOSTIC ONLY -- do the balls the frozen path misses have a raw proposal at all?

Predeclared as layer C of the B2 experiment spec: "of 17 misses across those two
windows, 12 had no usable proposal even at confidence 0.01. Re-run identically
and report how many of those 12 now receive a usable proposal."

The question separates two very different failures. A ball the detector ranks
below the production threshold is a thresholding problem; a ball for which no
box exists at any confidence is detector blindness. Only the second is what a
finer feature stride is supposed to fix, so B2 has to be judged against it.

Two detectors, one model, per run:

  FROZEN PATH      LocalDetector(confidence=BALL_ACCEPT_CONF,
                                 ball_candidate_pool=True)
                   -- byte-identical construction to tools/eval_temporal_val.py.
                   A GT ball is MISSED when no accepted ball detection
                   (confidence >= BALL_ACCEPT_CONF, after the production dedupe)
                   reaches IoU >= MATCH_IOU with it.

  PROPOSAL PATH    LocalDetector(confidence=<floor>, ball_candidate_pool=False)
                   -- the pool is OFF, so no duplicate suppression runs and the
                   boxes are raw. The floor reaches BELOW the production
                   candidate threshold; that is the entire point, and it applies
                   to this diagnostic alone.

This tool NEVER writes a production threshold, never calls the temporal
selector, and never touches tools/eval_temporal_val.py. Its numbers carry no
success criterion and cannot be used to adopt or reject anything.

VALIDATION ONLY. The temporal benchmark is VAL_ONLY by construction and there is
no split argument to misuse -- sealed TEST is unreachable from this file.

Example:
    python tools/diagnose_temporal_ball_proposals.py --model best_A_960.pt
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_models import iou_matrix                                # noqa: E402
from trackers.detector import (BALL_ACCEPT_CONF, BALL_CANDIDATE_CONF,  # noqa: E402
                               LocalDetector)

# Both frozen, and deliberately imported rather than restated: the frozen
# benchmark's geometry rule and the production accept threshold must not be able
# to drift apart from this diagnostic.
MATCH_IOU = 0.5
TV = Path('data/temporal_val')

# The two windows the P2 hypothesis is about, per the B2 spec. women_1 w1 is the
# small-ball regime; youth_premier_league w1 is documented as NOT primarily a
# resolution problem and is reported beside it, never pooled with it.
HISTORICAL_WINDOWS = (('women_1', 1), ('youth_premier_league', 1))
HISTORICAL_MISSES = 17
HISTORICAL_NO_PROPOSAL = 12


def load_gt_ball(path: Path, w: int, h: int) -> np.ndarray:
    """Benchmark labels are single-class (0 = ball). Same reader as the frozen eval."""
    if not path.exists():
        return np.empty((0, 4))
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        p = line.split()
        if len(p) == 5:
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out).reshape(-1, 4)


def best_iou(gt_box: np.ndarray, boxes) -> float:
    if not boxes:
        return 0.0
    m = iou_matrix(gt_box.reshape(1, 4), np.array([b['bbox'] for b in boxes]))
    return float(m.max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--proposal-floor', type=float, default=0.01,
                    help='Confidence floor for the DIAGNOSTIC proposal pass only. '
                         'Never applied to the frozen path. Default 0.01, the '
                         'value the historical diagnostic used.')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if not 0.0 < args.proposal_floor <= BALL_CANDIDATE_CONF:
        sys.exit(f'--proposal-floor must be in (0, {BALL_CANDIDATE_CONF}]; this is a '
                 'below-production diagnostic, not a threshold change.')

    import cv2
    from PIL import Image

    man = json.loads((TV / 'manifest.json').read_text(encoding='utf-8'))
    frames = man['frames']

    frozen = LocalDetector(args.model, confidence=BALL_ACCEPT_CONF,
                           imgsz=args.imgsz, ball_candidate_pool=True)
    raw = LocalDetector(args.model, confidence=args.proposal_floor,
                        imgsz=args.imgsz, ball_candidate_pool=False)

    per_window = defaultdict(lambda: {'gt': 0, 'missed': 0,
                                      'missed_with_proposal': 0,
                                      'missed_without_proposal': 0})
    objects = []

    for f in frames:
        p = TV / 'images' / f['file']
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        w, h = Image.open(p).size
        gt = load_gt_ball(TV / 'labels' / f'{Path(f["file"]).stem}.txt', w, h)
        if not len(gt):
            continue

        accepted = [d for d in frozen.detect(img)
                    if d['class'] == 'ball' and d['confidence'] >= BALL_ACCEPT_CONF]
        proposals = [d for d in raw.detect(img) if d['class'] == 'ball']

        key = f"{f['match']} w{f['window']}"
        for i, g in enumerate(gt):
            per_window[key]['gt'] += 1
            if best_iou(g, accepted) >= MATCH_IOU:
                continue                       # found by the frozen path
            per_window[key]['missed'] += 1
            iou_raw = best_iou(g, proposals)
            has = iou_raw >= MATCH_IOU
            per_window[key]['missed_with_proposal'] += has
            per_window[key]['missed_without_proposal'] += (not has)
            objects.append({
                'file': f['file'], 'window': key, 'gt_index': i,
                'gt_bbox_xyxy': [round(v, 2) for v in g.tolist()],
                'gt_max_wh_px': round(float(max(g[2] - g[0], g[3] - g[1])), 2),
                'shot_type': f['shot_type'],
                'best_iou_raw_proposal': round(iou_raw, 4),
                'has_usable_proposal': bool(has),
                'n_raw_proposals_in_frame': len(proposals),
            })
        print(f'  {f["file"]}: gt {len(gt)}  accepted {len(accepted)}  '
              f'raw@{args.proposal_floor} {len(proposals)}', flush=True)

    hist_keys = [f'{m} w{w}' for m, w in HISTORICAL_WINDOWS]
    hist = {'missed': sum(per_window[k]['missed'] for k in hist_keys),
            'without_proposal': sum(per_window[k]['missed_without_proposal']
                                    for k in hist_keys),
            'with_proposal': sum(per_window[k]['missed_with_proposal']
                                 for k in hist_keys)}
    reproduced = (hist['missed'] == HISTORICAL_MISSES
                  and hist['without_proposal'] == HISTORICAL_NO_PROPOSAL)

    total = {'missed': sum(v['missed'] for v in per_window.values()),
             'without_proposal': sum(v['missed_without_proposal']
                                     for v in per_window.values()),
             'with_proposal': sum(v['missed_with_proposal']
                                  for v in per_window.values())}

    report = {
        'DIAGNOSTIC_ONLY': True,
        'note': 'Carries no success criterion. Cannot adopt or reject anything.',
        'model': args.model,
        'imgsz': args.imgsz,
        'proposal_floor': args.proposal_floor,
        'frozen_thresholds': {'ball_accept_conf': BALL_ACCEPT_CONF,
                              'ball_candidate_conf': BALL_CANDIDATE_CONF,
                              'match_iou': MATCH_IOU},
        'benchmark': 'data/temporal_val (VAL_ONLY)',
        'per_window': {k: dict(v) for k, v in sorted(per_window.items())},
        'historical_windows': {
            'windows': hist_keys, **hist,
            'historical_missed': HISTORICAL_MISSES,
            'historical_without_proposal': HISTORICAL_NO_PROPOSAL,
            'population_reproduced': reproduced},
        'all_windows': total,
        'missed_objects': objects,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding='utf-8')

    print('\n' + '=' * 78)
    print(f'DIAGNOSTIC ONLY -- raw ball proposals at conf >= {args.proposal_floor}, '
          f'IoU >= {MATCH_IOU}')
    print(f'model: {args.model}')
    print('=' * 78)
    hdr = f'  {"window":<34}{"GT":>5}{"missed":>8}{"has prop":>10}{"no prop":>9}'
    print(hdr); print('  ' + '-' * (len(hdr) - 2))
    for k, v in sorted(per_window.items()):
        print(f'  {k:<34}{v["gt"]:>5}{v["missed"]:>8}'
              f'{v["missed_with_proposal"]:>10}{v["missed_without_proposal"]:>9}')
    print(f'\n  historical windows {hist_keys}:')
    print(f'    missed {hist["missed"]} (historical {HISTORICAL_MISSES}), '
          f'no usable proposal {hist["without_proposal"]} '
          f'(historical {HISTORICAL_NO_PROPOSAL})')
    print(f'    HISTORICAL POPULATION REPRODUCED: {"YES" if reproduced else "NO"}')
    print(f'\n  all windows: missed {total["missed"]}, '
          f'with proposal {total["with_proposal"]}, '
          f'without {total["without_proposal"]}')
    print(f'\n  report -> {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
