#!/usr/bin/env python
"""
S1D -- contract-aware SN3D BASE vs OPT gate.

Runs SN3D_BASE and SN3D_OPT over both frozen populations, scores them under the
same primary centre criterion used in S1/S1B/S1C, adds bbox-geometry
decomposition, and caches per-object hit records so no future gate has to pay for
these predictions again.

BASE is rerun ONLY because the saved S1B/S1C artifacts stored aggregates rather
than per-object hits, and the paired BASE-vs-OPT counts and the geometry
decomposition are impossible without them. EyeCU B@1280/B@1920 are NOT rerun --
their figures are read from the saved artifacts.

Static scoring is POSITIVE-ARM ONLY: the 111-ball GT is not exhaustive under
ALL_VISIBLE_PHYSICAL_FOOTBALLS, so unmatched predictions are never false
positives and no precision, mAP or FP/image is computed from it.

Raw detectors only: no SAHI, no TTA, no augment, no tiling, no selector, no
interpolation, no tracking.

VALIDATION ONLY. Sealed TEST is unreachable from this file.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_models import iou_matrix                                 # noqa: E402
from trackers.detector import (BALL_ACCEPT_CONF, BALL_CANDIDATE_CONF,  # noqa: E402
                               BALL_DEDUPE_IOU, LocalDetector,
                               suppress_ball_duplicates)

_s = importlib.util.spec_from_file_location(
    's1', Path(__file__).resolve().parent / 'experiment_s1_ball_specialists.py')
S1 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(S1)

TV = S1.TV
VAL_IMG = Path('data/dataset_baseline/images/val')
VAL_LAB = Path('data/dataset_baseline/labels/val')
SUBGROUP = Path('EyeCU_S1_results/s1c/S1C_STATIC_OVERLAP_SUBGROUP.json')
BASE_SHA = 'e8c1a900300893c34bf36c964c5854ed93603470e04a4a8eba73f70e4eea148b'
OPT_SHA = 'a3082fb435a8501ae17cfe4ac78e66ca7041205e115feda344d34d1693064f36'
D4 = S1.OFFICIAL_DIST_THRESHOLD
D133 = S1.NORMALISED_DIST_THRESHOLD
IOU_T = 0.50
IMGSZ = 1280
BALL_CLS = 3


def q(a, p):
    return None if not len(a) else round(float(np.percentile(a, p)), 4)


def stats(a):
    a = np.asarray(a, dtype=float)
    return {'n': int(len(a)), 'median': q(a, 50), 'p25': q(a, 25), 'p75': q(a, 75)}


def detect_all(weights, paths, imgsz):
    det = LocalDetector(str(weights), confidence=BALL_CANDIDATE_CONF, iou=0.5,
                        imgsz=imgsz, device='cpu', ball_candidate_pool=False)
    import cv2
    out = {}
    t0 = time.time()
    for p in paths:
        im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        bs = [d for d in det.detect(im) if d['class'] == 'ball']
        kept = suppress_ball_duplicates(
            [b for b in bs if b['confidence'] >= BALL_CANDIDATE_CONF], BALL_DEDUPE_IOU)
        out[p.name] = [b for b in kept if b['confidence'] >= BALL_ACCEPT_CONF]
    return out, (time.time() - t0) * 1000 / max(len(paths), 1)


def match(gt_boxes, preds, mode, thr):
    """Deterministic one-to-one matching. Returns {gt_index: pred_dict}."""
    pairs = []
    for gi, g in enumerate(gt_boxes):
        gc = ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)
        for pi, d in enumerate(preds):
            b = d['bbox']
            if mode == 'centre':
                v = float(np.hypot((b[0] + b[2]) / 2 - gc[0], (b[1] + b[3]) / 2 - gc[1]))
                if v < thr:
                    pairs.append((v, gi, pi))
            else:
                v = float(iou_matrix(np.array(g).reshape(1, 4),
                                     np.array(b).reshape(1, 4))[0, 0])
                if v >= thr:
                    pairs.append((-v, gi, pi))
    pairs.sort()
    ug, up, res = set(), set(), {}
    for _v, gi, pi in pairs:
        if gi in ug or pi in up:
            continue
        ug.add(gi)
        up.add(pi)
        res[gi] = preds[pi]
    return res


def load_yolo_boxes(path, w, h, cls=None):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding='utf-8').splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        if cls is not None and int(p[0]) != cls:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def geom(gt, pred):
    gw, gh = gt[2] - gt[0], gt[3] - gt[1]
    pw, ph = pred[2] - pred[0], pred[3] - pred[1]
    gc = ((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2)
    pc = ((pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2)
    return {
        'centre_err': float(np.hypot(pc[0] - gc[0], pc[1] - gc[1])),
        'iou': float(iou_matrix(np.array(gt).reshape(1, 4),
                                np.array(pred).reshape(1, 4))[0, 0]),
        'w_ratio': pw / gw if gw else None,
        'h_ratio': ph / gh if gh else None,
        'area_ratio': (pw * ph) / (gw * gh) if gw * gh else None,
        'pred_aspect': pw / ph if ph else None,
        'gt_aspect': gw / gh if gh else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', required=True)
    ap.add_argument('--opt', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    import cv2
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for p, exp, nm in ((Path(args.base), BASE_SHA, 'BASE'), (Path(args.opt), OPT_SHA, 'OPT')):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != exp:
            sys.exit(f'{nm} hash mismatch {h} != {exp}. STOP.')

    # ---------------- populations, loaded verbatim
    man = json.loads((TV / 'manifest.json').read_text(encoding='utf-8'))
    tframes = man['frames']
    r1 = json.loads(S1.R1.read_text(encoding='utf-8'))
    hard = r1['contact_set']['members']
    sub = json.loads(SUBGROUP.read_text(encoding='utf-8'))
    smem = {m['id']: m for m in sub['members']}

    tgt_c, tgt_b = {}, {}
    for f in tframes:
        im = cv2.imdecode(np.fromfile(str(TV / 'images' / f['file']), dtype=np.uint8),
                          cv2.IMREAD_COLOR)
        h_, w_ = im.shape[:2]
        b = load_yolo_boxes(TV / 'labels' / f'{Path(f["file"]).stem}.txt', w_, h_)
        tgt_b[f['file']] = b
        tgt_c[f['file']] = [((x[0] + x[2]) / 2, (x[1] + x[3]) / 2) for x in b]

    simgs = sorted(VAL_IMG.iterdir())
    sgt = {}
    for p in simgs:
        im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        h_, w_ = im.shape[:2]
        sgt[p.name] = load_yolo_boxes(VAL_LAB / f'{p.stem}.txt', w_, h_, BALL_CLS)

    # ---------------- GT geometry sanity
    def gt_geom(ids):
        W, H, MX, AR = [], [], [], []
        for i in ids:
            m = smem[i]
            b = m['bbox']
            w_, h_ = b[2] - b[0], b[3] - b[1]
            W.append(w_); H.append(h_); MX.append(max(w_, h_))
            AR.append(w_ / h_ if h_ else np.nan)
        AR = np.array(AR, dtype=float)
        return {'width': stats(W), 'height': stats(H), 'max_dim': stats(MX),
                'aspect': stats(AR[~np.isnan(AR)]),
                'fraction_approx_square_0.9_1.1':
                    round(float(np.mean((AR >= 0.9) & (AR <= 1.1))), 4)}

    allid = list(smem)
    geom_report = {
        'overall': gt_geom(allid),
        'youth': gt_geom([i for i in allid if smem[i]['youth']]),
        'non_youth': gt_geom([i for i in allid if not smem[i]['youth']]),
        'overlap': gt_geom([i for i in allid if smem[i]['overlap_either']])}

    # ---------------- inference
    models = {'BASE': Path(args.base), 'OPT': Path(args.opt)}
    preds_t, preds_s, ms = {}, {}, {}
    for nm, wpath in models.items():
        preds_t[nm], a = detect_all(wpath, [TV / 'images' / f['file'] for f in tframes], IMGSZ)
        preds_s[nm], b = detect_all(wpath, simgs, IMGSZ)
        ms[nm] = {'temporal_ms': round(a, 1), 'static_ms': round(b, 1)}
        print(f'{nm}: inference done  {a:.0f}/{b:.0f} ms/img', flush=True)

    # ---------------- temporal scoring
    def temporal_score(nm, thr):
        pr = {}
        for f in tframes:
            acc = preds_t[nm][f['file']]
            best = max(acc, key=lambda d: d['confidence']) if acc else None
            pr[f['file']] = (None if best is None else {
                'xy': ((best['bbox'][0] + best['bbox'][2]) / 2,
                       (best['bbox'][1] + best['bbox'][3]) / 2),
                'score': best['confidence']})
        return S1.score(pr, tframes, tgt_c, hard, thr)

    def temporal_iou(nm):
        tp = 0
        hit = set()
        for f in tframes:
            g = tgt_b[f['file']]
            if not g:
                continue
            m = match(g, preds_t[nm][f['file']], 'iou', IOU_T)
            tp += len(m)
            for gi in m:
                hit.add(f"{f['file']}#{gi}")
        return tp, hit

    # ---------------- static scoring
    def static_hits(nm, mode, thr):
        hits, matched = set(), {}
        for p in simgs:
            g = sgt[p.name]
            if not g:
                continue
            m = match(g, preds_s[nm][p.name], mode, thr)
            for gi, d in m.items():
                key = f'{Path(p.name).stem}#{gi}'
                hits.add(key)
                matched[key] = d
        return hits, matched

    report = {'EXPERIMENT': 'S1D contract-aware SN3D OPT gate',
              'DIAGNOSTIC_ONLY': True, 'device': 'cpu',
              'checkpoints': {'BASE': {'sha256': BASE_SHA},
                              'OPT': {'sha256': OPT_SHA}},
              'inference': {'imgsz': IMGSZ, 'conf': BALL_ACCEPT_CONF,
                            'floor': BALL_CANDIDATE_CONF,
                            'dedupe_iou': BALL_DEDUPE_IOU, 'ms': ms},
              'gt_geometry': geom_report,
              'temporal': {}, 'static': {}, 'paired': {}, 'geometry_decomp': {}}

    for nm in models:
        report['temporal'][nm] = {
            'centre_4px': temporal_score(nm, D4),
            'centre_1p333px': temporal_score(nm, D133),
            'iou50_matched': temporal_iou(nm)[0]}

    # static tables
    GROUPS = {'overall': allid,
              'youth': [i for i in allid if smem[i]['youth']],
              'non_youth': [i for i in allid if not smem[i]['youth']],
              'overlap': [i for i in allid if smem[i]['overlap_either']],
              'non_overlap': [i for i in allid if not smem[i]['overlap_either']]}
    shits = {}
    for nm in models:
        h4, m4 = static_hits(nm, 'centre', D4)
        h133, _ = static_hits(nm, 'centre', D133)
        hi, mi = static_hits(nm, 'iou', IOU_T)
        shits[nm] = {'c4': h4, 'c133': h133, 'iou': hi, 'm4': m4}
        report['static'][nm] = {}
        for g, ids in GROUPS.items():
            s = set(ids)
            report['static'][nm][g] = {
                'gt': len(ids),
                'centre_4px': len(s & h4),
                'centre_1p333px': len(s & h133),
                'iou50': len(s & hi),
                'recall_4px': round(len(s & h4) / len(ids), 4) if ids else None,
                'recall_iou50': round(len(s & hi) / len(ids), 4) if ids else None,
                'box_conversion': (round(len(s & hi) / len(s & h4), 4)
                                   if len(s & h4) else None)}

    # paired BASE vs OPT
    for scope, sets in (('static', {g: set(ids) for g, ids in GROUPS.items()}),):
        for g, ids in sets.items():
            for metric, key in (('centre_4px', 'c4'), ('iou50', 'iou')):
                a = ids & shits['BASE'][key]
                b = ids & shits['OPT'][key]
                report['paired'][f'{scope}_{g}_{metric}'] = {
                    'n': len(ids), 'both': len(a & b), 'base_only': len(a - b),
                    'opt_only': len(b - a), 'neither': len(ids - a - b)}

    # temporal paired on hard / overlap
    thit = {}
    for nm in models:
        pr = {}
        for f in tframes:
            acc = preds_t[nm][f['file']]
            best = max(acc, key=lambda d: d['confidence']) if acc else None
            if best is None:
                continue
            cx = (best['bbox'][0] + best['bbox'][2]) / 2
            cy = (best['bbox'][1] + best['bbox'][3]) / 2
            g = tgt_c[f['file']]
            if g and min(float(np.hypot(cx - c[0], cy - c[1])) for c in g) < D4:
                pr[f['file']] = True
        thit[nm] = set(pr)
    for gname, mem in (('temporal_hard', hard),
                       ('temporal_overlap', [m for m in hard if m['human_overlap']])):
        ids = {m['file'] for m in mem}
        a, b = ids & thit['BASE'], ids & thit['OPT']
        report['paired'][f'{gname}_centre_4px'] = {
            'n': len(ids), 'both': len(a & b), 'base_only': len(a - b),
            'opt_only': len(b - a), 'neither': len(ids - a - b)}

    # geometry decomposition on centre-correct static detections
    for nm in models:
        report['geometry_decomp'][nm] = {}
        for g, ids in GROUPS.items():
            rows = [geom(smem[i]['bbox'], shits[nm]['m4'][i]['bbox'])
                    for i in ids if i in shits[nm]['m4']]
            if not rows:
                report['geometry_decomp'][nm][g] = {'n': 0}
                continue
            report['geometry_decomp'][nm][g] = {
                'n': len(rows),
                'centre_err': stats([r['centre_err'] for r in rows]),
                'iou': stats([r['iou'] for r in rows]),
                'w_ratio': stats([r['w_ratio'] for r in rows]),
                'h_ratio': stats([r['h_ratio'] for r in rows]),
                'area_ratio': stats([r['area_ratio'] for r in rows]),
                'pred_aspect': stats([r['pred_aspect'] for r in rows]),
                'gt_aspect': stats([r['gt_aspect'] for r in rows])}

    (out / 'S1D_RESULTS.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    for nm in models:
        t = report['temporal'][nm]['centre_4px']
        s = report['static'][nm]['overall']
        print(f"{nm}: TEMPORAL {t['tp']}/77 hard {t['hard_recovered']}/24 "
              f"ovl {t['overlap_recovered']}/20 evt {t['events_touched']}/15 "
              f"empty {t['empty_fired_frames']}/27 IoU {report['temporal'][nm]['iou50_matched']}/77"
              f" | STATIC 4px {s['centre_4px']}/111 IoU {s['iou50']}/111 "
              f"conv {s['box_conversion']}", flush=True)
    print(f"\nwritten: {out / 'S1D_RESULTS.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
