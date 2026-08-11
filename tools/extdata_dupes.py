#!/usr/bin/env python
"""
Stages 7/8/9/11: duplicates, augmentation copies, and EyeCU leakage.

Four independent mechanisms, deliberately not collapsed into one score:

1. EXACT, by content. sha256 of the file bytes, and separately sha256 of the
   DECODED pixels. The second catches the same frame re-encoded into a different
   container, which a byte hash misses. Filenames are never compared: Roboflow
   renames everything.

2. AUGMENTATION COPIES, by construction. Roboflow writes every image as
   "<source-stem>.rf.<hash>.<ext>". Files sharing a source stem are copies of one
   source image. This is metadata, not a guess -- it is exact, reproducible, and
   it separates real visual diversity from export-time inflation.

3. NEAR-DUPLICATES, two stages. A cheap perceptual hash proposes candidates; a
   pixel comparison decides. The canonical form strips letterboxing and resizes
   to a fixed square, which makes a 640x640 stretched export and a 640x360
   original comparable -- both carry the same content, differently shaped.

4. LEAKAGE, external vs EyeCU TRAIN / VAL / TEST, using the same machinery.

The trap this has to avoid is calling two different frames of the same match
duplicates. Verification therefore looks at how much the pixels actually MOVED:
a resize/recompression/exposure copy of one frame leaves players where they
were, while the next frame of the same video does not. Both numbers are kept on
every pair so a judgement can be re-checked rather than trusted.

TEST is read as IMAGES ONLY, for hashing. No TEST label is opened, and nothing
here evaluates anything.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / 'experiments' / 'external_data_audit'
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

SIG = 64            # canonical signature size
HAMMING_CANDIDATE = 8
RF_STEM = re.compile(r'^(.*?)[._]rf[._][0-9a-f]{6,}', re.I)


def imread_gray(p):
    import cv2
    a = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    return a


def canonical(bgr):
    """Letterbox-stripped, aspect-agnostic 64x64 grayscale signature.

    Aspect is deliberately discarded. A Roboflow 640x640 stretch of a 16:9 frame
    and the same frame at 640x360 differ only by that stretch, so normalising
    both to a square makes them comparable instead of invisible to each other.
    """
    import cv2
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    nz_r = np.where(g.max(axis=1) > 12)[0]
    nz_c = np.where(g.max(axis=0) > 12)[0]
    if len(nz_r) > 8 and len(nz_c) > 8:
        g = g[nz_r.min():nz_r.max() + 1, nz_c.min():nz_c.max() + 1]
    return cv2.resize(g, (SIG, SIG), interpolation=cv2.INTER_AREA)


def dhash(sig):
    import cv2
    s = cv2.resize(sig, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return np.packbits(bits)


POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)


def collect(name, root, recurse_dirs):
    out = []
    for d in recurse_dirs:
        p = root / d
        if not p.exists():
            continue
        for f in sorted(p.rglob('*')):
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                out.append((name, f))
    return out


def source_stem(fname: str) -> str:
    m = RF_STEM.match(fname)
    return m.group(1) if m else Path(fname).stem


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments/external_data_audit/reports')
    args = ap.parse_args()
    import cv2

    srcs = json.loads((AUDIT / 'raw' / 'SOURCES.json').read_text(encoding='utf-8'))['sources']
    items = []
    for sid, s in srcs.items():
        items += collect(sid, REPO / s['extracted_to'],
                         ['train/images', 'valid/images', 'test/images'])

    # EyeCU. TRAIN and VAL come from the built split; TEST is the three matches
    # reserved for the single final evaluation -- images only, never labels.
    items += collect('EYECU_TRAIN', REPO / 'data' / 'dataset_baseline' / 'images', ['train'])
    items += collect('EYECU_VAL', REPO / 'data' / 'dataset_baseline' / 'images', ['val'])
    items += collect('EYECU_TEST', REPO / 'data' / 'frames',
                     ['como_2-0_sassuolo', 'manchester_city_v_liverpool', 'youth_2'])

    print(f'hashing {len(items)} images from '
          f'{len({n for n, _ in items})} collections')

    recs, sigs, hashes = [], [], []
    for i, (grp, p) in enumerate(items):
        b = p.read_bytes()
        img = imread_gray(p)
        if img is None:
            recs.append({'group': grp, 'path': str(p.relative_to(REPO)).replace('\\', '/'),
                         'file': p.name, 'corrupt': True})
            sigs.append(np.zeros((SIG, SIG), np.uint8))
            hashes.append(np.zeros(8, np.uint8))
            continue
        sig = canonical(img)
        recs.append({
            'group': grp,
            'path': str(p.relative_to(REPO)).replace('\\', '/'),
            'file': p.name,
            'stem': source_stem(p.name),
            'bytes_sha256': hashlib.sha256(b).hexdigest(),
            'pixels_sha256': hashlib.sha256(np.ascontiguousarray(img)).hexdigest(),
            'wh': [img.shape[1], img.shape[0]],
        })
        sigs.append(sig)
        hashes.append(dhash(sig))
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(items)}')

    sigs = np.stack(sigs).astype(np.float32)
    H = np.stack(hashes)
    n = len(recs)

    # ---- 1. exact -----------------------------------------------------------
    by_bytes, by_pixels = defaultdict(list), defaultdict(list)
    for i, r in enumerate(recs):
        if r.get('corrupt'):
            continue
        by_bytes[r['bytes_sha256']].append(i)
        by_pixels[r['pixels_sha256']].append(i)
    exact_groups = [v for v in by_pixels.values() if len(v) > 1]
    byte_groups = [v for v in by_bytes.values() if len(v) > 1]

    # ---- 2. augmentation copies (Roboflow source stem) ----------------------
    by_stem = defaultdict(list)
    for i, r in enumerate(recs):
        if r['group'].startswith('EYECU') or r.get('corrupt'):
            continue
        by_stem[(r['group'], r['stem'])].append(i)

    # ---- 3/4. near-duplicate candidates ------------------------------------
    print('near-duplicate candidate stage (perceptual hash)')
    cand = []
    for i in range(n):
        d = POP[np.bitwise_xor(H[i + 1:], H[i])].sum(axis=1)
        for off in np.where(d <= HAMMING_CANDIDATE)[0]:
            cand.append((i, i + 1 + int(off), int(d[off])))
    print(f'  {len(cand)} candidate pairs at hamming <= {HAMMING_CANDIDATE}')

    # ---- verification -------------------------------------------------------
    print('verification stage (pixel comparison)')
    flat = sigs.reshape(n, -1)
    flat = (flat - flat.mean(1, keepdims=True))
    norm = np.linalg.norm(flat, axis=1) + 1e-6
    pairs = []
    for i, j, ham in cand:
        a, b = sigs[i], sigs[j]
        ncc = float(flat[i] @ flat[j] / (norm[i] * norm[j]))
        moved = float((np.abs(a - b) > 12).mean())     # fraction of pixels that moved
        mad = float(np.abs(a - b).mean())
        if recs[i].get('corrupt') or recs[j].get('corrupt'):
            continue
        same_pixels = recs[i]['pixels_sha256'] == recs[j]['pixels_sha256']
        if same_pixels:
            verdict = 'EXACT_DUPLICATE'
        elif ncc >= 0.985 and moved <= 0.02:
            verdict = 'HIGH_CONFIDENCE_NEAR_DUPLICATE'
        elif ncc >= 0.95 and moved <= 0.10:
            verdict = 'POSSIBLE_DUPLICATE_REVIEW'
        else:
            verdict = 'DISTINCT'
        pairs.append({'i': i, 'j': j, 'a': recs[i]['path'], 'b': recs[j]['path'],
                      'ga': recs[i]['group'], 'gb': recs[j]['group'],
                      'hamming': ham, 'ncc': round(ncc, 4),
                      'moved_frac': round(moved, 4), 'mad': round(mad, 2),
                      'same_source_stem': (recs[i]['group'] == recs[j]['group'] and
                                           recs[i].get('stem') == recs[j].get('stem')),
                      'verdict': verdict})

    out = Path(REPO / args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'image_records.json').write_text(json.dumps(recs, indent=0), encoding='utf-8')
    (out / 'pair_verdicts.json').write_text(json.dumps(pairs, indent=0), encoding='utf-8')

    # ---- summary ------------------------------------------------------------
    def grp(i):
        return recs[i]['group']
    summary = {
        'images_hashed': n,
        'collections': {g: sum(1 for r in recs if r['group'] == g)
                        for g in dict.fromkeys(r['group'] for r in recs)},
        'exact': {
            'byte_identical_groups': len(byte_groups),
            'pixel_identical_groups': len(exact_groups),
            'images_in_pixel_identical_groups': sum(len(g) for g in exact_groups),
            'within_source': defaultdict(int),
            'cross_source_pairs': defaultdict(int),
        },
        'augmentation': {},
        'near_duplicate': {
            'candidate_pairs': len(cand),
            'HIGH_CONFIDENCE_NEAR_DUPLICATE': sum(1 for p in pairs
                                                  if p['verdict'] == 'HIGH_CONFIDENCE_NEAR_DUPLICATE'),
            'POSSIBLE_DUPLICATE_REVIEW': sum(1 for p in pairs
                                             if p['verdict'] == 'POSSIBLE_DUPLICATE_REVIEW'),
            'DISTINCT': sum(1 for p in pairs if p['verdict'] == 'DISTINCT'),
            'EXACT_DUPLICATE': sum(1 for p in pairs if p['verdict'] == 'EXACT_DUPLICATE'),
        },
        'leakage': {},
    }
    for g in exact_groups:
        gs = {grp(i) for i in g}
        if len(gs) == 1:
            summary['exact']['within_source'][gs.pop()] += 1
        else:
            summary['exact']['cross_source_pairs']['|'.join(sorted(gs))] += 1
    summary['exact']['within_source'] = dict(summary['exact']['within_source'])
    summary['exact']['cross_source_pairs'] = dict(summary['exact']['cross_source_pairs'])

    for sid in srcs:
        stems = {k[1] for k in by_stem if k[0] == sid}
        tot = sum(len(v) for k, v in by_stem.items() if k[0] == sid)
        multi = {k: len(v) for k, v in by_stem.items() if k[0] == sid and len(v) > 1}
        summary['augmentation'][sid] = {
            'images': tot,
            'distinct_source_images': len(stems),
            'inflation_factor': round(tot / max(len(stems), 1), 3),
            'source_images_with_multiple_exports': len(multi),
            'generated_copies': tot - len(stems),
        }

    lk = defaultdict(lambda: {'EXACT': [], 'HIGH_CONFIDENCE_NEAR_DUPLICATE': [],
                              'POSSIBLE_DUPLICATE_REVIEW': []})
    for p in pairs:
        ga, gb = p['ga'], p['gb']
        ext, eye = None, None
        if ga.startswith('EYECU') and not gb.startswith('EYECU'):
            eye, ext = ga, gb
        elif gb.startswith('EYECU') and not ga.startswith('EYECU'):
            eye, ext = gb, ga
        if eye is None:
            continue
        if p['verdict'] == 'EXACT_DUPLICATE':
            lk[f'{ext}->{eye}']['EXACT'].append(p)
        elif p['verdict'] in ('HIGH_CONFIDENCE_NEAR_DUPLICATE', 'POSSIBLE_DUPLICATE_REVIEW'):
            lk[f'{ext}->{eye}'][p['verdict']].append(p)
    summary['leakage'] = {k: {kk: len(vv) for kk, vv in v.items()} for k, v in lk.items()}
    (out / 'leakage_pairs.json').write_text(
        json.dumps({k: v for k, v in lk.items()}, indent=1), encoding='utf-8')
    (out / 'duplicates.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')

    print('\nEXACT (pixel-identical) groups:', len(exact_groups),
          ' byte-identical groups:', len(byte_groups))
    print(' within source:', summary['exact']['within_source'])
    print(' cross source :', summary['exact']['cross_source_pairs'])
    print('\nAUGMENTATION / export inflation')
    for sid, v in summary['augmentation'].items():
        print(f'  {sid}  {v["images"]:>5} files from {v["distinct_source_images"]:>5} '
              f'source images  (x{v["inflation_factor"]})')
    print('\nNEAR-DUPLICATE verdicts:', summary['near_duplicate'])
    print('\nLEAKAGE vs EyeCU:', dict(summary['leakage']) or 'none detected')


if __name__ == '__main__':
    main()
