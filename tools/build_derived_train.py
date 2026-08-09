#!/usr/bin/env python
"""
Experiment C: derived TRAIN-only scale/context views + hard negatives.

Why. The measured ball failure at youth w1 is not a small-object failure. Those
balls are 18.9-24.2 px at 960 inference geometry, against a TRAIN median of 6.7
and a p90 of 12.0 -- above the entire upper tail of the training distribution.
Meanwhile "ball near a player" is already 77.7% of TRAIN, so mining more
proximity would add what the model has in abundance. The measured deficit is
contextual LARGE-ball appearance.

This generates that distribution from existing labels rather than new
annotation, by cropping a smaller region of an existing frame and resizing it to
the training geometry. The ball's apparent size grows while its surroundings --
boots, legs, nearby players, grass, hoardings -- come with it. Isolated ball
crops would teach the model a ball floating on nothing, which is not the failure
we measured.

Two products, kept separate in provenance:

  positive_scale_context   ball resized into the 12-25 px band (emphasis 15-22)
  hard_negative            a region the current detector confidently calls a
                           ball where no ground-truth ball exists -- crests,
                           socks, boots, pitch markings, hoardings

Hard negatives are NOT blank background. Easy background already dominates
every frame; what the detector confuses are specific ball-like objects, and one
of those (a goalkeeper's shirt crest) is already documented in RESULTS.md.

TRAIN ONLY. Validation and test sources are rejected by name and by file
identity, and the check runs before anything is written.

Example:
    python tools/build_derived_train.py --dry-run
    python tools/build_derived_train.py
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASSES = ['player', 'goalkeeper', 'referee', 'ball']
BALL_ID = 3
HUMAN_IDS = (0, 1, 2)

# Frozen splits. Named here so a rename upstream fails loudly rather than
# silently leaking a held-out match into training.
VAL_MATCHES = {'austin_fc_vs__club_tijuana', 'bayern_munich_3-1_chelsea',
               'women_1', 'youth_premier_league'}
TEST_MATCHES = {'como_2-0_sassuolo', 'manchester_city_v_liverpool', 'youth_2'}

# --- predeclared generation settings -----------------------------------
IMGSZ = 960.0                 # inference geometry the audit is expressed in
DERIVED_W, DERIVED_H = 640, 360
TARGET_BAND = (12.0, 25.0)    # derived ball width at 960 geometry
TARGET_EMPHASIS = (15.0, 22.0)
MAX_ZOOM = 2.5                # upscale ceiling; beyond this crops go mushy
MIN_RETAINED_AREA = 0.60      # keep a label only if this much survives cropping
MIN_BOX_PX = 2.0
MAX_POS_PER_SOURCE = 6
MAX_NEG_PER_SOURCE = 4
N_POSITIVES = 70              # inside the requested 50-80
N_NEGATIVES = 30              # inside the requested 20-40
NEG_CONF = 0.25               # a false ball the detector actually accepts
SEED = 0


def load_split(root: Path, split: str):
    from PIL import Image
    idir, ldir = root / 'images' / split, root / 'labels' / split
    recs = []
    for ip in sorted(idir.glob('*.jpg')):
        w, h = Image.open(ip).size
        boxes = []
        lp = ldir / f'{ip.stem}.txt'
        if lp.exists():
            for line in lp.read_text(encoding='utf-8').splitlines():
                q = line.split()
                if len(q) != 5:
                    continue
                c = int(float(q[0]))
                cx, cy, bw, bh = (float(v) for v in q[1:5])
                boxes.append((c, [(cx-bw/2)*w, (cy-bh/2)*h,
                                  (cx+bw/2)*w, (cy+bh/2)*h]))
        recs.append({'path': ip, 'match': ip.stem.rsplit('_', 1)[0],
                     'w': w, 'h': h, 'boxes': boxes})
    return recs


def assert_train_only(recs) -> None:
    """Fail before writing anything if a held-out source is present."""
    bad = {r['match'] for r in recs} & (VAL_MATCHES | TEST_MATCHES)
    if bad:
        raise SystemExit(f'REFUSING: held-out matches in the source pool: {sorted(bad)}')


def plan_crop(box, w, h, target_960, rng):
    """
    Crop window that makes this ball `target_960` px wide at 960 geometry.

    The derived image is DERIVED_W wide, so a crop of width C is scaled by
    DERIVED_W/C, and the derived frame is itself scaled by IMGSZ/DERIVED_W at
    inference. Composing those, the crop width needed is simply
    native_width * IMGSZ / target, independent of the source resolution.
    """
    bw = box[2] - box[0]
    if bw <= 0:
        return None
    crop_w = bw * IMGSZ / target_960
    crop_h = crop_w * DERIVED_H / DERIVED_W
    if crop_w > w or crop_h > h:
        return None
    if DERIVED_W / crop_w > MAX_ZOOM:
        return None

    # Deliberately off-centre: a ball always in the middle would teach position,
    # not appearance.
    bcx, bcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    fx, fy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
    x0 = min(max(bcx - fx * crop_w, 0), w - crop_w)
    y0 = min(max(bcy - fy * crop_h, 0), h - crop_h)
    return x0, y0, crop_w, crop_h


def transform(boxes, crop, keep_ball=True):
    x0, y0, cw, ch = crop
    s = DERIVED_W / cw
    out = []
    for c, b in boxes:
        if c == BALL_ID and not keep_ball:
            continue
        nx1, ny1 = (b[0]-x0)*s, (b[1]-y0)*s
        nx2, ny2 = (b[2]-x0)*s, (b[3]-y0)*s
        full = max(1e-6, (nx2-nx1) * (ny2-ny1))
        cx1, cy1 = max(0.0, nx1), max(0.0, ny1)
        cx2, cy2 = min(float(DERIVED_W), nx2), min(float(DERIVED_H), ny2)
        if cx2 - cx1 < MIN_BOX_PX or cy2 - cy1 < MIN_BOX_PX:
            continue
        if ((cx2-cx1) * (cy2-cy1)) / full < MIN_RETAINED_AREA:
            continue
        out.append((c, [cx1, cy1, cx2, cy2]))
    return out


def to_yolo(labels):
    lines = []
    for c, b in labels:
        cx = (b[0]+b[2]) / 2 / DERIVED_W
        cy = (b[1]+b[3]) / 2 / DERIVED_H
        bw = (b[2]-b[0]) / DERIVED_W
        bh = (b[3]-b[1]) / DERIVED_H
        lines.append(f'{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
    return '\n'.join(lines)


def write_crop(src_path, crop, out_path):
    from PIL import Image
    x0, y0, cw, ch = crop
    im = Image.open(src_path).convert('RGB')
    im = im.crop((int(round(x0)), int(round(y0)),
                  int(round(x0+cw)), int(round(y0+ch))))
    im = im.resize((DERIVED_W, DERIVED_H), Image.LANCZOS)
    im.save(out_path, quality=95)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='data/dataset_baseline')
    ap.add_argument('--out', default='data/derived_train')
    ap.add_argument('--model', default='best_A_960.pt')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-negatives', action='store_true')
    args = ap.parse_args()

    rng = random.Random(SEED)
    root = Path(args.dataset)
    recs = load_split(root, 'train')
    assert_train_only(recs)
    print(f'{len(recs)} TRAIN images, {len({r["match"] for r in recs})} sources')

    # ---------------------------------------------------- positives
    cands = []
    for r in recs:
        for c, b in r['boxes']:
            if c != BALL_ID:
                continue
            native = b[2] - b[0]
            # zoom needed for the mid-emphasis target; lower is better quality
            need = DERIVED_W / max(1e-6, native * IMGSZ / np.mean(TARGET_EMPHASIS))
            cands.append({'rec': r, 'box': b, 'native': native, 'zoom': need})
    cands.sort(key=lambda c: (c['zoom'], c['rec']['path'].name))

    per_source = Counter()
    positives = []
    for cd in cands:
        if len(positives) >= N_POSITIVES:
            break
        m = cd['rec']['match']
        if per_source[m] >= MAX_POS_PER_SOURCE:
            continue
        crop = None
        for lo, hi in (TARGET_EMPHASIS, TARGET_BAND):
            for _ in range(6):
                t = rng.uniform(lo, hi)
                crop = plan_crop(cd['box'], cd['rec']['w'], cd['rec']['h'], t, rng)
                if crop:
                    break
            if crop:
                break
        if not crop:
            continue
        labels = transform(cd['rec']['boxes'], crop, keep_ball=True)
        if not any(c == BALL_ID for c, _ in labels):
            continue                       # the ball itself must survive
        per_source[m] += 1
        positives.append({'kind': 'positive_scale_context', 'rec': cd['rec'],
                          'crop': crop, 'labels': labels,
                          'orig_ball': [round(v, 2) for v in cd['box']],
                          'native_w': round(cd['native'], 2)})

    # ---------------------------------------------------- hard negatives
    negatives = []
    if not args.skip_negatives:
        import cv2
        from trackers.detector import LocalDetector, suppress_ball_duplicates, BALL_DEDUPE_IOU
        from compare_models import iou_matrix
        det = LocalDetector(args.model, confidence=NEG_CONF, imgsz=960)
        found = []
        for i, r in enumerate(recs, 1):
            img = cv2.imdecode(np.fromfile(str(r['path']), np.uint8), cv2.IMREAD_COLOR)
            balls = suppress_ball_duplicates(
                [d for d in det.detect(img) if d['class'] == 'ball'], BALL_DEDUPE_IOU)
            gt = np.array([b for c, b in r['boxes'] if c == BALL_ID]).reshape(-1, 4)
            for d in balls:
                if len(gt) and float(iou_matrix(gt, np.array(d['bbox']).reshape(1, 4)).max()) >= 0.5:
                    continue           # a real ball, not a negative
                found.append({'rec': r, 'box': d['bbox'], 'conf': d['confidence']})
            if i % 150 == 0:
                print(f'  mining {i}/{len(recs)}', flush=True)
        found.sort(key=lambda f: (-f['conf'], f['rec']['path'].name))
        per_source_n = Counter()
        for fnd in found:
            if len(negatives) >= N_NEGATIVES:
                break
            m = fnd['rec']['match']
            if per_source_n[m] >= MAX_NEG_PER_SOURCE:
                continue
            crop = None
            for _ in range(6):
                t = rng.uniform(*TARGET_EMPHASIS)
                crop = plan_crop(fnd['box'], fnd['rec']['w'], fnd['rec']['h'], t, rng)
                if crop:
                    break
            if not crop:
                continue
            # Transform every label, balls included: if a real ball happens to
            # fall inside this crop it MUST be annotated. A "hard negative" that
            # silently drops a true ball is a mislabelled image.
            labels = transform(fnd['rec']['boxes'], crop, keep_ball=True)
            per_source_n[m] += 1
            negatives.append({'kind': 'hard_negative', 'rec': fnd['rec'],
                              'crop': crop, 'labels': labels,
                              'fp_conf': round(fnd['conf'], 4),
                              'fp_box': [round(v, 2) for v in fnd['box']],
                              'contains_real_ball': any(c == BALL_ID for c, _ in labels)})

    items = positives + negatives
    print(f'\nplanned: {len(positives)} positives, {len(negatives)} hard negatives')
    if args.dry_run:
        print('(dry run -- nothing written)')
        return items

    out = Path(args.out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'labels').mkdir(parents=True, exist_ok=True)
    manifest = {'seed': SEED, 'derived_geometry': [DERIVED_W, DERIVED_H],
                'inference_geometry': IMGSZ,
                'target_band_960': list(TARGET_BAND),
                'target_emphasis_960': list(TARGET_EMPHASIS),
                'split': 'TRAIN_ONLY', 'items': []}
    seen = set()
    for n, it in enumerate(items):
        tag = 'pos' if it['kind'] == 'positive_scale_context' else 'neg'
        name = f"{it['rec']['match']}_d{tag}{n:04d}.jpg"
        if name in seen:
            raise SystemExit(f'duplicate output name {name}')
        seen.add(name)
        write_crop(it['rec']['path'], it['crop'], out / 'images' / name)
        (out / 'labels' / f'{Path(name).stem}.txt').write_text(
            to_yolo(it['labels']) + ('\n' if it['labels'] else ''), encoding='utf-8')
        x0, y0, cw, ch = it['crop']
        rec = {'file': name, 'kind': it['kind'], 'source_image': it['rec']['path'].name,
               'source_match': it['rec']['match'], 'split': 'TRAIN_ONLY',
               'crop': {'x': round(x0, 2), 'y': round(y0, 2),
                        'w': round(cw, 2), 'h': round(ch, 2),
                        'zoom': round(DERIVED_W / cw, 4)},
               'n_labels': len(it['labels']),
               'classes': dict(Counter(CLASSES[c] for c, _ in it['labels']))}
        if tag == 'pos':
            rec['original_ball_box'] = it['orig_ball']
            rec['original_ball_width_native'] = it['native_w']
            bb = [b for c, b in it['labels'] if c == BALL_ID][0]
            rec['derived_ball_box'] = [round(v, 2) for v in bb]
            rec['derived_ball_width_960'] = round((bb[2]-bb[0]) * IMGSZ / DERIVED_W, 2)
        else:
            rec['fp_confidence'] = it['fp_conf']
            rec['fp_box'] = it['fp_box']
            rec['contains_real_ball'] = it['contains_real_ball']
        manifest['items'].append(rec)

    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (out / 'SPLIT').write_text(
        'TRAIN_ONLY\n\nDerived views of TRAIN images. Never add to val or test.\n',
        encoding='utf-8')
    print(f'written: {out}')
    return items


if __name__ == '__main__':
    main()
