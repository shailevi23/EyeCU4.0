#!/usr/bin/env python
"""
S1C -- does SN3D's overlap advantage generalize to the independent static VAL?

Positive-arm scoring ONLY. The frozen 208-image VAL carries 111 verified source
ball GT, which is NOT exhaustive under EyeCU's binding
ALL_VISIBLE_PHYSICAL_FOOTBALLS ontology. So unmatched predictions are never
counted as false positives, and no precision, mAP or FP/image is computed here.

The player/GK overlap subgroup is built from GROUND-TRUTH boxes only -- no model
predictions, no IoU threshold, no pixel radius, no visual judgement -- and is
frozen to disk with its SHA256 BEFORE any detector runs.

Raw detectors only: no SAHI, no TTA, no augment, no tiling, no selector, no
interpolation, no tracking.

VALIDATION ONLY. Sealed TEST is unreachable from this file.

    python tools/experiment_s1c_static_overlap.py --sn3d <ckpt> --out-dir <dir>
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_models import iou_matrix                                 # noqa: E402
from trackers.detector import (BALL_ACCEPT_CONF, BALL_CANDIDATE_CONF,  # noqa: E402
                               BALL_DEDUPE_IOU, LocalDetector,
                               suppress_ball_duplicates)

_spec = importlib.util.spec_from_file_location(
    's1', Path(__file__).resolve().parent / 'experiment_s1_ball_specialists.py')
S1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S1)

VAL = Path('data/dataset_baseline')
IMG = VAL / 'images' / 'val'
LAB = VAL / 'labels' / 'val'
# Verified from data/dataset_baseline/football.yaml, not assumed.
CLS = {'player': 0, 'goalkeeper': 1, 'referee': 2, 'ball': 3}
DIST_4PX = S1.OFFICIAL_DIST_THRESHOLD
IOU_MATCH = 0.50
SN3D_SHA = 'e8c1a900300893c34bf36c964c5854ed93603470e04a4a8eba73f70e4eea148b'
MODELS = [('SN3D@1280', None, 1280), ('B@1280', 'best_B_1280.pt', 1280),
          ('B@1920', 'best_B_1280.pt', 1920)]


def read_labels(stem, w, h):
    """Returns {class_id: [xyxy, ...]} for one image."""
    out = {}
    p = LAB / f'{stem}.txt'
    if not p.exists():
        return out
    for line in p.read_text(encoding='utf-8').splitlines():
        q = line.split()
        if len(q) != 5:
            continue
        c = int(q[0])
        cx, cy, bw, bh = (float(v) for v in q[1:5])
        out.setdefault(c, []).append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                      (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def intersects(a, b):
    """Positive intersection area. No threshold."""
    return (min(a[2], b[2]) > max(a[0], b[0])
            and min(a[3], b[3]) > max(a[1], b[1]))


def inside(pt, b):
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def match_greedy(gt, preds, key):
    """
    Deterministic one-to-one matching maximizing valid matches.
    key(gt_box, pred) -> cost, lower is better; only pairs passing `ok` match.
    Returns set of matched GT indices.
    """
    pairs = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(preds):
            v = key(g, p)
            if v is not None:
                pairs.append((v, gi, pi))
    pairs.sort()
    used_g, used_p, matched = set(), set(), set()
    for _v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched.add(gi)
    return matched


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sn3d', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    import cv2
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sn = Path(args.sn3d)
    h = hashlib.sha256(sn.read_bytes()).hexdigest()
    if h != SN3D_SHA:
        sys.exit(f'SN3D checkpoint hash mismatch: {h} != {SN3D_SHA}. STOP.')

    # ---------------------------------------------- benchmark fact check
    images = sorted(IMG.iterdir())
    balls, humans, sizes = {}, {}, {}
    for p in images:
        im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        hh, ww = im.shape[:2]
        sizes[p.stem] = (ww, hh)
        lab = read_labels(p.stem, ww, hh)
        balls[p.stem] = lab.get(CLS['ball'], [])
        humans[p.stem] = {'player': lab.get(CLS['player'], []),
                          'goalkeeper': lab.get(CLS['goalkeeper'], [])}

    n_ball = sum(len(v) for v in balls.values())
    facts = {
        'images': len(images), 'ball_gt': n_ball,
        'positive_images': sum(1 for v in balls.values() if v),
        'multi_ball_images': sum(1 for v in balls.values() if len(v) > 1),
        'youth_images': sum(1 for p in images if p.stem.startswith('youth_premier_league')),
        'youth_ball_gt': sum(len(balls[p.stem]) for p in images
                             if p.stem.startswith('youth_premier_league')),
        'non_youth_images': sum(1 for p in images
                                if not p.stem.startswith('youth_premier_league')),
        'non_youth_ball_gt': sum(len(balls[p.stem]) for p in images
                                 if not p.stem.startswith('youth_premier_league')),
        'human_gt_player_boxes': sum(len(humans[p.stem]['player']) for p in images),
        'human_gt_goalkeeper_boxes': sum(len(humans[p.stem]['goalkeeper']) for p in images),
        'class_ids_source': 'data/dataset_baseline/football.yaml',
        'class_ids': CLS,
        'gt_provenance': ('hand-corrected annotation, docs/results/RESULTS.md '
                          '"659 frames corrected by hand ... (451 train, 208 val)"'),
        'exhaustive_under_ALL_VISIBLE': False,
    }

    # ---------------------------------------------- freeze overlap subgroup
    objs = []
    for p in images:
        stem = p.stem
        youth = stem.startswith('youth_premier_league')
        for i, b in enumerate(balls[stem]):
            ov_p = any(intersects(b, q) for q in humans[stem]['player'])
            ov_g = any(intersects(b, q) for q in humans[stem]['goalkeeper'])
            ctr = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            in_h = any(inside(ctr, q) for q in
                       humans[stem]['player'] + humans[stem]['goalkeeper'])
            objs.append({'id': f'{stem}#{i}', 'stem': stem, 'gt_index': i,
                         'bbox': [round(v, 2) for v in b],
                         'centre': [round(v, 2) for v in ctr],
                         'max_wh_px': round(max(b[2] - b[0], b[3] - b[1]), 2),
                         'youth': youth,
                         'overlap_player': ov_p, 'overlap_goalkeeper': ov_g,
                         'overlap_either': ov_p or ov_g,
                         'centre_inside_human': in_h})

    subgroup = {
        'FROZEN_BEFORE_INFERENCE': True,
        'definition': ('STATIC_PLAYER_GK_OVERLAP = ball GT bbox has positive '
                       'intersection area with >=1 GT player or goalkeeper bbox. '
                       'GT boxes only. No IoU threshold, no pixel radius, no '
                       'model predictions, no referee boxes, no visual judgement.'),
        'secondary_definition': ('BALL_CENTER_INSIDE_HUMAN = centre of ball GT '
                                 'bbox lies inside any GT player/GK bbox'),
        'benchmark_facts': facts,
        'total_ball_gt': len(objs),
        'overlap_either': sum(o['overlap_either'] for o in objs),
        'overlap_player': sum(o['overlap_player'] for o in objs),
        'overlap_goalkeeper': sum(o['overlap_goalkeeper'] for o in objs),
        'non_overlap': sum(not o['overlap_either'] for o in objs),
        'centre_inside_human': sum(o['centre_inside_human'] for o in objs),
        'members': objs}
    sp = out / 'S1C_STATIC_OVERLAP_SUBGROUP.json'
    sp.write_text(json.dumps(subgroup, indent=1), encoding='utf-8')
    sub_sha = hashlib.sha256(sp.read_bytes()).hexdigest()
    print(f'frozen subgroup -> {sp}  sha256 {sub_sha}', flush=True)
    print(f'  ball GT {len(objs)}  overlap {subgroup["overlap_either"]} '
          f'(player {subgroup["overlap_player"]}, gk {subgroup["overlap_goalkeeper"]}) '
          f'non-overlap {subgroup["non_overlap"]}  '
          f'centre-inside {subgroup["centre_inside_human"]}', flush=True)

    # ---------------------------------------------- inference
    results = {}
    for tag, w, imgsz in MODELS:
        weights = str(sn) if tag.startswith('SN3D') else w
        det = LocalDetector(weights, confidence=BALL_CANDIDATE_CONF, iou=0.5,
                            imgsz=imgsz, device='cpu', ball_candidate_pool=False)
        acc_by = {}
        t0 = time.time()
        for p in images:
            im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            bs = [d for d in det.detect(im) if d['class'] == 'ball']
            kept = suppress_ball_duplicates(
                [b for b in bs if b['confidence'] >= BALL_CANDIDATE_CONF],
                BALL_DEDUPE_IOU)
            acc_by[p.stem] = [b for b in kept if b['confidence'] >= BALL_ACCEPT_CONF]
        ms = (time.time() - t0) * 1000 / len(images)

        hit4, hitiou = set(), set()
        for p in images:
            stem = p.stem
            gt = balls[stem]
            if not gt:
                continue
            preds = acc_by[stem]
            if not preds:
                continue
            m4 = match_greedy(
                gt, preds,
                lambda g, d: (lambda dd: dd if dd < DIST_4PX else None)(
                    float(np.hypot((d['bbox'][0] + d['bbox'][2]) / 2 - (g[0] + g[2]) / 2,
                                   (d['bbox'][1] + d['bbox'][3]) / 2 - (g[1] + g[3]) / 2))))
            mi = match_greedy(
                gt, preds,
                lambda g, d: (lambda v: -v if v >= IOU_MATCH else None)(
                    float(iou_matrix(np.array(g).reshape(1, 4),
                                     np.array(d['bbox']).reshape(1, 4))[0, 0])))
            for gi in m4:
                hit4.add(f'{stem}#{gi}')
            for gi in mi:
                hitiou.add(f'{stem}#{gi}')
        results[tag] = {'hit_4px': hit4, 'hit_iou': hitiou, 'ms': round(ms, 1),
                        'accepted_total': sum(len(v) for v in acc_by.values())}
        print(f'{tag}: 4px {len(hit4)}/{len(objs)}  IoU50 {len(hitiou)}/{len(objs)} '
              f'  {ms:.0f} ms/img', flush=True)

    # ---------------------------------------------- report
    def group(pred):
        return [o for o in objs if pred(o)]

    GROUPS = {
        'overall': group(lambda o: True),
        'youth': group(lambda o: o['youth']),
        'non_youth': group(lambda o: not o['youth']),
        'player_gk_overlap': group(lambda o: o['overlap_either']),
        'non_overlap': group(lambda o: not o['overlap_either']),
        'centre_inside_human': group(lambda o: o['centre_inside_human']),
    }

    report = {'EXPERIMENT': 'S1C static generalization + human-overlap validation',
              'DIAGNOSTIC_ONLY': True, 'device': 'cpu',
              'scoring': ('POSITIVE-ARM ONLY. Unmatched predictions are NOT false '
                          'positives; no precision, mAP or FP/image is reported.'),
              'sn3d_sha256': h, 'subgroup_artifact_sha256': sub_sha,
              'benchmark_facts': facts,
              'subgroup_counts': {k: len(v) for k, v in GROUPS.items()},
              'runtime_ms_per_image': {t: results[t]['ms'] for t, _, _ in MODELS},
              'groups': {}, 'paired': {}}

    for gname, members in GROUPS.items():
        ids = {o['id'] for o in members}
        report['groups'][gname] = {'gt': len(members)}
        for tag, _, _ in MODELS:
            for metric, key in (('4px', 'hit_4px'), ('iou50', 'hit_iou')):
                m = len(ids & results[tag][key])
                report['groups'][gname][f'{tag}_{metric}'] = {
                    'matched': m, 'missed': len(members) - m,
                    'recall': round(m / len(members), 4) if members else None}

    for ref in ('B@1280', 'B@1920'):
        report['paired'][f'SN3D@1280_vs_{ref}'] = {}
        for gname, members in GROUPS.items():
            ids = {o['id'] for o in members}
            a = ids & results['SN3D@1280']['hit_4px']
            b = ids & results[ref]['hit_4px']
            report['paired'][f'SN3D@1280_vs_{ref}'][gname] = {
                'n': len(members), 'both': len(a & b),
                'sn3d_only': len(a - b), 'b_only': len(b - a),
                'neither': len(ids - a - b)}

    (out / 'S1C_RESULTS.json').write_text(json.dumps(report, indent=1),
                                          encoding='utf-8')
    print(f'\nwritten: {out / "S1C_RESULTS.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
