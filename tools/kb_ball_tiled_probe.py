#!/usr/bin/env python
"""
Can tiled inference find the tiny footballs the full frame misses?

The full-frame generator recalls 23.0% of the 248 human-confirmed additions.
The obvious suspect is scale: a 6 px ball on a 1280x720 frame, resized to the
detector's input, is a handful of pixels by the time it reaches the network. If
that is the whole story, cropping the frame and running the detector on each
crop at native resolution should recover most of the missing balls, because the
same ball is then 3-4x larger relative to the input.

This tests that, and only that. Offline QA proposal generation has no runtime
budget to respect, so the question is purely whether the recall exists to be had.

    A  FULL FRAME    the current baseline
    B  2x2 TILES     20% overlap
    C  3x3 TILES     20% overlap
    D  PYRAMID       full frame + 3x3 tiles upscaled 2x

Overlap matters more than it looks: a ball sitting on a tile seam is cut in
half in both neighbours and detected in neither, which would show up as a
recall floor that no amount of tiling fixes. 20% is generous relative to a
ball's size at this scale.

THE REFERENCE IS HUMAN-DRAWN BOXES ONLY. The reviewed images are
HUMAN_REVIEWED_PARTIAL: existing blue GT was never exhaustively validated and
one annotation proved to be a player. Measuring against it would be measuring
against unverified data, so the denominator is the 248 additions a human drew.

    python tools/kb_ball_tiled_probe.py --run
    python tools/kb_ball_tiled_probe.py --run --limit 40    # quick pass

Inference only. No training, no verdict, no annotation touched.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import kb_ball_calibrate as CAL                                   # noqa: E402
import kb_ball_candidates as CAND                                 # noqa: E402
import kb_images                                                  # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
REPORT = PKG / 'BALL_TILED_PROBE.json'

CONF = 0.03
OVERLAP = 0.20
BUCKETS = CAL.BUCKETS
RECALL_TARGETS = (0.50, 0.70, 0.80, 0.90)


def tiles(w, h, cols, rows, overlap=OVERLAP):
    """Crop rectangles with overlap, in original image coordinates."""
    tw, th = w / cols, h / rows
    ow, oh = tw * overlap, th * overlap
    out = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0.0, c * tw - ow)
            y0 = max(0.0, r * th - oh)
            x1 = min(float(w), (c + 1) * tw + ow)
            y1 = min(float(h), (r + 1) * th + oh)
            out.append((x0, y0, x1, y1))
    return out


def _detect(model, img, imgsz, conf=CONF):
    res = model.predict(source=img, imgsz=imgsz, conf=conf,
                        classes=[CAND.BALL_CLASS], verbose=False)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        out.append({'bbox_xywh': [x1, y1, x2 - x1, y2 - y1],
                    'conf': float(b.conf[0])})
    return out


def dedupe(dets):
    """Merge detections of the same object, keeping the most confident.

    Same rule as the queue builder: centre distance within max(1.5*w, 8 px).
    Tiling makes this essential rather than cosmetic -- with 20% overlap a ball
    in an overlap band is genuinely detected twice, and counting it twice would
    inflate the workload figure this experiment exists to measure.
    """
    dets = sorted(dets, key=lambda d: -d['conf'])
    kept = []
    for d in dets:
        cx = d['bbox_xywh'][0] + d['bbox_xywh'][2] / 2
        cy = d['bbox_xywh'][1] + d['bbox_xywh'][3] / 2
        dup = False
        for k in kept:
            kx = k['bbox_xywh'][0] + k['bbox_xywh'][2] / 2
            ky = k['bbox_xywh'][1] + k['bbox_xywh'][3] / 2
            if ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5 <= max(
                    1.5 * k['bbox_xywh'][2], 8.0):
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def run_methods(rows, limit=None):
    """Every method over every calibration image. Returns {method: {img: dets}}."""
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(str(CAND.WEIGHTS))
    out = {m: defaultdict(list) for m in ('FULL', 'TILE_2x2', 'TILE_3x3',
                                          'PYRAMID')}
    t0 = time.time()
    for n, r in enumerate(rows, 1):
        path = kb_images.resolve(r['IMAGE'])
        im = Image.open(path).convert('RGB')
        W, H = im.size

        out['FULL'][r['IMAGE']] = [
            d for d in _detect(model, str(path), CAND.IMGSZ)
            if CAND.plausible(d['bbox_xywh'])[0]]

        for name, (cols, rws, scale) in (('TILE_2x2', (2, 2, 1)),
                                         ('TILE_3x3', (3, 3, 1)),
                                         ('PYRAMID', (3, 3, 2))):
            acc = []
            for (x0, y0, x1, y1) in tiles(W, H, cols, rws):
                crop = im.crop((int(x0), int(y0), int(x1), int(y1)))
                if scale != 1:
                    crop = crop.resize((crop.width * scale, crop.height * scale),
                                       Image.BICUBIC)
                for d in _detect(model, np.array(crop)[:, :, ::-1], CAND.IMGSZ):
                    b = d['bbox_xywh']
                    acc.append({'bbox_xywh': [b[0] / scale + x0,
                                              b[1] / scale + y0,
                                              b[2] / scale, b[3] / scale],
                                'conf': d['conf']})
            if name == 'PYRAMID':
                acc += out['FULL'][r['IMAGE']]
            acc = [d for d in acc if CAND.plausible(d['bbox_xywh'])[0]]
            out[name][r['IMAGE']] = dedupe(acc)
        if n % 25 == 0:
            print(f'  {n}/{len(rows)} images, {time.time()-t0:.0f}s', flush=True)
    return out


def score(per_image, adds, gts, conf=CONF):
    """Coverage of confirmed additions, plus the workload that buys it."""
    covered, missed = [], []
    for a in adds:
        dets = [d for d in per_image.get(a['IMAGE'], []) if d['conf'] >= conf]
        ref = [{'BOX_ID': 'human', 'bbox_xywh': a['bbox_xywh']}]
        hit = next((d for d in dets
                    if CAND.matches_gt(d['bbox_xywh'], ref)), None)
        (covered if hit else missed).append({**a,
                                             'conf': hit['conf'] if hit else None})
    total = unmatched = 0
    for im, dets in per_image.items():
        keep = [d for d in dets if d['conf'] >= conf]
        total += len(keep)
        unmatched += sum(1 for d in keep
                         if not CAND.matches_gt(d['bbox_xywh'], gts.get(im, [])))
    n = len(adds)
    return {
        'proposals_total': total,
        'proposals_unmatched': unmatched,
        'proposals_per_image': round(unmatched / max(len(per_image), 1), 2),
        'confirmed_covered': len(covered),
        'confirmed_missed': n - len(covered),
        'confirmed_recall': (len(covered) / n) if n else 0.0,
        'recall_by_size': {
            name: {'n': sum(1 for a in adds if a['size_bucket'] == name),
                   'covered': sum(1 for c in covered
                                  if c['size_bucket'] == name)}
            for _, name in BUCKETS},
        'covered_confs': sorted(c['conf'] for c in covered if c['conf']),
    }


def union(a, b):
    out = {}
    for im in set(a) | set(b):
        out[im] = dedupe(list(a.get(im, [])) + list(b.get(im, [])))
    return out


def workload_for_recall(per_image, adds, gts, targets=RECALL_TARGETS):
    """Proposals needed at each recall target, by sweeping the threshold down.

    Reported as the honest cost curve: a method that reaches 80% only by
    proposing forty boxes per image has not solved anything, and this is where
    that shows up.
    """
    confs = sorted({round(d['conf'], 4)
                    for v in per_image.values() for d in v}, reverse=True)
    out = {}
    for t in targets:
        hit = None
        for c in confs:
            s = score(per_image, adds, gts, conf=c)
            if s['confirmed_recall'] >= t:
                hit = {'threshold': c,
                       'proposals_unmatched': s['proposals_unmatched'],
                       'proposals_per_image': s['proposals_per_image'],
                       'recall': round(s['confirmed_recall'], 4)}
                break
        out[f'{int(t*100)}%'] = hit or 'NOT REACHABLE at any threshold'
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    if not args.run:
        print(REPORT.read_text(encoding='utf-8')[:3000] if REPORT.is_file()
              else 'run with --run')
        return

    rows = CAL.reviewed_images()
    if args.limit:
        rows = rows[:args.limit]
    frame = {r['IMAGE'] for r in rows}
    adds = [a for a in CAL.confirmed_additions() if a['IMAGE'] in frame]
    gts = CAND.existing_ball_gt(rows)

    print(f'calibration frame: {len(rows)} reviewed PP images')
    print(f'human-confirmed additions: {len(adds)}')
    print(f'size mix: {dict(Counter(a["size_bucket"] for a in adds))}')
    print(f'\nmethods: FULL, TILE_2x2, TILE_3x3, PYRAMID (+FULL), '
          f'overlap {int(OVERLAP*100)}%, conf {CONF}')
    print('INFERENCE ONLY. No training, no annotation modified.\n')

    per = run_methods(rows, args.limit)
    results = {m: score(per[m], adds, gts) for m in per}

    unions = {
        'FULL+2x2': union(per['FULL'], per['TILE_2x2']),
        'FULL+3x3': union(per['FULL'], per['TILE_3x3']),
        'FULL+PYRAMID': union(per['FULL'], per['PYRAMID']),
        'ALL': union(union(per['FULL'], per['TILE_2x2']),
                     union(per['TILE_3x3'], per['PYRAMID'])),
    }
    for k, v in unions.items():
        results[k] = score(v, adds, gts)

    print(f'\n{"method":<14}{"proposals":>10}{"unmatched":>11}{"/img":>7}'
          f'{"covered":>9}{"missed":>8}{"recall":>9}')
    for m, s in results.items():
        print(f'{m:<14}{s["proposals_total"]:>10}{s["proposals_unmatched"]:>11}'
              f'{s["proposals_per_image"]:>7.1f}{s["confirmed_covered"]:>9}'
              f'{s["confirmed_missed"]:>8}{100*s["confirmed_recall"]:>8.1f}%')

    print(f'\nrecall by size')
    print(f'{"method":<14}' + ''.join(f'{n:>12}' for _, n in BUCKETS))
    for m, s in results.items():
        cells = [f'{s["recall_by_size"][n]["covered"]}/'
                 f'{s["recall_by_size"][n]["n"]}' if s['recall_by_size'][n]['n']
                 else '-' for _, n in BUCKETS]
        print(f'{m:<14}' + ''.join(f'{c:>12}' for c in cells))

    best = max(results, key=lambda m: results[m]['confirmed_recall'])
    best_pool = unions.get(best) or per[best]
    curve = workload_for_recall(best_pool, adds, gts)
    print(f'\nbest method: {best} at {100*results[best]["confirmed_recall"]:.1f}%')
    print('proposals needed per recall target (best method):')
    for k, v in curve.items():
        print(f'  {k:>4}  {v}')

    viable = results[best]['confirmed_recall'] >= 0.80
    rep = {
        'experiment': 'tiled / zoomed inference for tiny-football proposals',
        'reference': 'HUMAN_CONFIRMED_ADDITIONS only; existing GT is unverified',
        'frame': {'images': len(rows), 'confirmed_additions': len(adds),
                  'size_mix': dict(Counter(a['size_bucket'] for a in adds))},
        'model': {'weights': CAND.WEIGHTS.name,
                  'weights_sha256': hashlib.sha256(
                      CAND.WEIGHTS.read_bytes()).hexdigest(),
                  'conf': CONF, 'imgsz': CAND.IMGSZ, 'tile_overlap': OVERLAP},
        'results': {m: {k: v for k, v in s.items() if k != 'covered_confs'}
                    for m, s in results.items()},
        'best_method': best,
        'best_recall': results[best]['confirmed_recall'],
        'workload_for_recall': curve,
        'target_80pc_achievable': viable,
        'verdict': ('MODEL_ASSISTED PP COMPLETION VIABLE' if viable else
                    'MODEL_ASSISTED PP COMPLETION NOT VIABLE'),
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    REPORT.write_text(json.dumps(rep, indent=1) + '\n', encoding='utf-8')
    print(f'\nVERDICT: {rep["verdict"]}')
    print(f'written: {REPORT.relative_to(REPO)}')
    print('\nNo annotation modified, no verdict written, no training.')


if __name__ == '__main__':
    main()
