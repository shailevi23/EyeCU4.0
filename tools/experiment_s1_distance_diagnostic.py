#!/usr/bin/env python
"""
S1 validity diagnostic -- are the specialists' predictions NEAR the ball or not?

DIAGNOSTIC ONLY. Carries no success criterion and cannot promote or reject
anything. It answers exactly one question: when a WASB Soccer specialist fires on
a frame that has a GT ball, how far from the ball does it land?

This separates two very different explanations for a 0.0 official recall:

  genuine non-detection      predicted centres are far from the ball (tens to
                             hundreds of px) -- the model does not see it
  tolerance/scale artefact   predicted centres cluster just outside the 4 px
                             official tolerance -- the model sees it and the
                             metric or the coordinate mapping is at fault

It also reports the raw heatmap maximum, which shows whether the official 0.5
accept threshold is the binding constraint.

Runs on the 77 GT-ball target frames only. No thresholds are changed, no
predictions are re-decoded differently, no images are written or inspected.

    python tools/experiment_s1_distance_diagnostic.py --wasb-root <dir> \
        --weights-dir <dir> --out <json>
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_spec = importlib.util.spec_from_file_location(
    's1', Path(__file__).resolve().parent / 'experiment_s1_ball_specialists.py')
S1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S1)

TV = S1.TV


def pct(a, q):
    return None if not len(a) else round(float(np.percentile(a, q)), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wasb-root', required=True)
    ap.add_argument('--weights-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import cv2
    import torch

    wasb_root, weights_dir = Path(args.wasb_root), Path(args.weights_dir)
    man = json.loads((TV / 'manifest.json').read_text(encoding='utf-8'))

    # GT-ball target frames only
    targets, gts = [], {}
    for f in man['frames']:
        img = cv2.imdecode(np.fromfile(str(TV / 'images' / f['file']),
                                       dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        c = S1.load_gt_centre(Path(f['file']).stem, w, h)
        if c:
            targets.append(f)
            gts[f['file']] = c

    src = S1.SourceFrames()
    report = {'DIAGNOSTIC_ONLY': True,
              'note': ('Carries no success criterion. Explains a 0.0 official '
                       'recall; does not change it.'),
              'gt_ball_frames': len(targets),
              'official_dist_threshold_px': S1.OFFICIAL_DIST_THRESHOLD,
              'normalised_dist_threshold_px': round(S1.NORMALISED_DIST_THRESHOLD, 4),
              'official_score_threshold': S1.OFFICIAL_SCORE_THRESHOLD,
              'models': {}}

    for name, spec in S1.CANDIDATES.items():
        ck = weights_dir / spec['ckpt']
        if not ck.exists():
            report['models'][name] = {'status': 'no weights'}
            continue
        model, pp, _cfg, _p, nparam = S1.build_wasb(name, spec, wasb_root,
                                                    weights_dir)
        F, O = spec['model']['frames_in'], spec['model']['frames_out']
        pos = S1.PRIMARY_POSITION if (F == 3 and O == 3) else 0

        dists, hmmax, fired, nofire = [], [], 0, 0
        for f in targets:
            fi = f['source_frame_index']
            imgs = [src.get(f['source_video'], fi - pos + k) for k in range(F)]
            if any(i is None for i in imgs):
                continue

            # raw heatmap maximum, before the official accept threshold
            with torch.no_grad():
                hm = S1.wasb_raw_heatmap(model, spec, imgs, pos)
            hmmax.append(float(hm))

            res = S1.wasb_predict(model, pp, spec, imgs)
            best = S1.intra_frame_peak(res.get(pos, []))
            if best is None:
                nofire += 1
                continue
            fired += 1
            d = min(float(np.hypot(best['xy'][0] - c[0], best['xy'][1] - c[1]))
                    for c in gts[f['file']])
            dists.append(d)

        d = np.array(dists)
        report['models'][name] = {
            'status': 'ok', 'params': nparam, 'position': f'pos{pos}',
            'gt_frames_evaluated': len(hmmax),
            'frames_fired': fired, 'frames_silent': nofire,
            'dist_min': pct(d, 0), 'dist_p25': pct(d, 25), 'dist_median': pct(d, 50),
            'dist_p75': pct(d, 75), 'dist_max': pct(d, 100),
            'n_within_4px': int((d < 4).sum()) if len(d) else 0,
            'n_within_1p33px': int((d < S1.NORMALISED_DIST_THRESHOLD).sum()) if len(d) else 0,
            'n_within_10px': int((d < 10).sum()) if len(d) else 0,
            'n_within_25px': int((d < 25).sum()) if len(d) else 0,
            'n_within_50px': int((d < 50).sum()) if len(d) else 0,
            'heatmap_max_median': round(float(np.median(hmmax)), 4) if hmmax else None,
            'heatmap_max_p90': pct(np.array(hmmax), 90),
            'heatmap_max_max': round(float(np.max(hmmax)), 4) if hmmax else None,
        }
        r = report['models'][name]
        print(f'{name}: fired {fired}/{len(hmmax)}  dist med {r["dist_median"]} '
              f'min {r["dist_min"]}  <4px {r["n_within_4px"]} <10 {r["n_within_10px"]} '
              f'<25 {r["n_within_25px"]} <50 {r["n_within_50px"]}  '
              f'hm_max med {r["heatmap_max_median"]} max {r["heatmap_max_max"]}',
              flush=True)

    src.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'\nwritten: {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
