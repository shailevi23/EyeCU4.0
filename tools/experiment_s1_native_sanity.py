#!/usr/bin/env python
"""
S1-SANITY -- do the official WASB Soccer checkpoints localize their OWN domain?

TINY CONTROL. Not a benchmark. It asks one question: on a handful of native
ISSIA-CNR Soccer frames, with the official pipeline and the official metric, does
each released checkpoint put the ball in the right place at all?

If yes, S1's near-zero EyeCU recall is domain-transfer collapse.
If no, the S1 setup is suspect and its architecture conclusions do not hold.

Same five checkpoints as S1, same code path, same official pieces:
preprocessing, frames_in/out, postprocessor, threshold 0.5,
tracker=intra_frame_peak, inverse coordinate transform, dist < 4 px in original
source-frame coordinates. Nothing EyeCU-specific. No thresholds tuned. No
annotation. The only bypass is the framework's hard CUDA assert and its
hydra/omegaconf CLI layer, exactly as in S1.

Ground truth is the official soccer_annos XML; frame indices are 0-based
sequential decode order, matching runners/extract_frame.py (cnt starts at 0).
Only frames the official loader would treat as a visible, in-play ball are used:
outside=0, used_in_game=1, occluded=0 (datasets/soccer.py load_xml).

    python tools/experiment_s1_native_sanity.py --wasb-root <dir> \
        --weights-dir <dir> --video <ID-5.avi> --anno <ID-5.xml> --out <json>
"""

import argparse
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_spec = importlib.util.spec_from_file_location(
    's1', Path(__file__).resolve().parent / 'experiment_s1_ball_specialists.py')
S1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S1)

MAX_TARGETS = 10       # hard cost cap: 10 native frames TOTAL, shared by all models


def parse_official_gt(xml_path: Path):
    """Official load_xml semantics: keep outside=0 and used_in_game=1;
    visibility is occluded=0."""
    root = ET.parse(xml_path).getroot()
    gt = {}
    for track in root.iter('track'):
        for pts in track.iter('points'):
            fid = int(pts.attrib['frame'])
            outside = pts.attrib['outside'] == '1'
            visible = pts.attrib['occluded'] == '0'
            used = any(c.text == '1' for c in pts
                       if c.attrib.get('name') == 'used_in_game')
            if outside or not used:
                continue
            x, y = (float(v) for v in pts.attrib['points'].split(','))
            gt[fid] = {'xy': (x, y), 'visible': visible}
    return gt


def pick_contiguous(gt, n):
    """First run of n consecutive frame ids that are all visible in-play balls."""
    fids = sorted(f for f, v in gt.items() if v['visible'])
    run = []
    for f in fids:
        if run and f == run[-1] + 1:
            run.append(f)
        else:
            run = [f]
        if len(run) == n:
            return run
    return fids[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wasb-root', required=True)
    ap.add_argument('--weights-dir', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--anno', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import cv2

    gt = parse_official_gt(Path(args.anno))
    targets = pick_contiguous(gt, MAX_TARGETS)
    if len(targets) > MAX_TARGETS:
        targets = targets[:MAX_TARGETS]
    lo, hi = min(targets) - 1, max(targets) + 1

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f'cannot open {args.video}')
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sequential decode from 0, exactly as extract_frame_soccer counts.
    frames, idx = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if lo <= idx <= hi:
            frames[idx] = fr
        if idx > hi:
            break
        idx += 1
    cap.release()

    report = {
        'CONTROL_ONLY': True,
        'note': ('Tiny native-domain sanity control. Not a benchmark, no '
                 'statistical claim, carries no promotion criterion.'),
        'sample': {
            'source': ('official ISSIA-CNR Soccer, WASB-SBDT setup_soccer.sh; '
                       'GT from official soccer_annos.zip'),
            'video': Path(args.video).name,
            'video_role': 'ID-5 is an official soccer TEST video (configs/dataset/soccer.yaml)',
            'native_resolution': f'{native_w}x{native_h}',
            'video_frames': n_frames,
            'target_frame_ids': targets,
            'n_targets': len(targets),
            'gt_rule': 'outside=0, used_in_game=1, occluded=0 (official load_xml)',
            'frame_index_base': '0-based sequential decode (extract_frame_soccer cnt=0)',
            'large_download_required': 'NO -- 86 KB annotations + one 43 MB zip '
                                       'member pulled by HTTP range; the 292 MB '
                                       'Sequences.zip was never downloaded',
        },
        'official_rule': {'score_threshold': S1.OFFICIAL_SCORE_THRESHOLD,
                          'dist_threshold_px': S1.OFFICIAL_DIST_THRESHOLD,
                          'coordinate_system': f'original {native_w}x{native_h} frame',
                          'selection': 'tracker=intra_frame_peak'},
        'models': {},
    }

    wasb_root, weights_dir = Path(args.wasb_root), Path(args.weights_dir)
    for name, spec in S1.CANDIDATES.items():
        ck = weights_dir / spec['ckpt']
        if not ck.exists():
            report['models'][name] = {'sanity': 'NOT TESTED -- no weights'}
            continue
        model, pp, _cfg, _p, nparam = S1.build_wasb(name, spec, wasb_root,
                                                    weights_dir)
        F, O = spec['model']['frames_in'], spec['model']['frames_out']
        pos = S1.PRIMARY_POSITION if (F == 3 and O == 3) else 0

        dists, fired, skipped = [], 0, 0
        per_frame = []
        for t in targets:
            idxs = [t - pos + k for k in range(F)]
            imgs = [frames.get(i) for i in idxs]
            if any(i is None for i in imgs):
                skipped += 1
                continue
            res = S1.wasb_predict(model, pp, spec, imgs)
            best = S1.intra_frame_peak(res.get(pos, []))
            if best is None:
                per_frame.append({'frame': t, 'predicted': False})
                continue
            fired += 1
            g = gt[t]['xy']
            d = float(np.hypot(best['xy'][0] - g[0], best['xy'][1] - g[1]))
            dists.append(d)
            per_frame.append({'frame': t, 'predicted': True,
                              'pred_xy': [round(v, 2) for v in best['xy']],
                              'gt_xy': [round(v, 2) for v in g],
                              'dist_px': round(d, 2),
                              'tp': d < S1.OFFICIAL_DIST_THRESHOLD})

        d = np.array(dists)
        tp = int((d < S1.OFFICIAL_DIST_THRESHOLD).sum()) if len(d) else 0
        n = len(targets) - skipped
        passed = tp >= 1
        report['models'][name] = {
            'sanity': 'PASS' if passed else 'FAIL',
            'params': nparam, 'position': f'pos{pos}',
            'gt_positives': n, 'predictions_emitted': fired,
            'tp_within_4px': tp,
            'recall': round(tp / n, 4) if n else None,
            'dist_median_px': round(float(np.median(d)), 2) if len(d) else None,
            'dist_min_px': round(float(d.min()), 2) if len(d) else None,
            'dist_max_px': round(float(d.max()), 2) if len(d) else None,
            'frames_skipped_no_context': skipped,
            'per_frame': per_frame}
        r = report['models'][name]
        print(f'{name:<16} {r["sanity"]:<5} GT {n} pred {fired} TP<4px {tp} '
              f'recall {r["recall"]} med {r["dist_median_px"]} '
              f'min {r["dist_min_px"]}', flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'\nwritten: {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
