#!/usr/bin/env python
"""
Stage 9, done properly: orientation-invariant leakage check against EyeCU.

The first leakage pass compared signatures in their stored orientation. That is
not good enough here, and S4 proved it: 68 of its images are the same frames
rotated 90 degrees, and a rotation-sensitive signature scores them at NCC ~ 0 --
maximally different -- when they are in fact the same frame. A flipped or
rotated copy of an EyeCU validation frame is still an EyeCU validation frame,
and missing one would be the single worst error this audit could make.

So every external image is compared against every EyeCU image under all eight
dihedral transforms (4 rotations x mirror). Candidates come from a Hamming
search over the 8 hashes; survivors are verified on pixels.

TEST participates as IMAGES ONLY. No TEST label file is opened anywhere in this
tool, and no metric is computed on anything.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / 'experiments' / 'external_data_audit'
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SIG = 64
HAM = 10

POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)


def canonical(bgr):
    import cv2
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    r = np.where(g.max(axis=1) > 12)[0]
    c = np.where(g.max(axis=0) > 12)[0]
    if len(r) > 8 and len(c) > 8:
        g = g[r.min():r.max() + 1, c.min():c.max() + 1]
    return cv2.resize(g, (SIG, SIG), interpolation=cv2.INTER_AREA)


def dihedral(sig):
    """The eight orientations of a square: 4 rotations, each also mirrored."""
    out = []
    for k in range(4):
        r = np.rot90(sig, k)
        out.append(np.ascontiguousarray(r))
        out.append(np.ascontiguousarray(np.fliplr(r)))
    return out


def dhash(sig):
    import cv2
    s = cv2.resize(sig, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return np.packbits((s[:, 1:] > s[:, :-1]).flatten())


def load(paths):
    import cv2
    sigs = []
    for p in paths:
        a = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        sigs.append(None if a is None else canonical(a))
    return sigs


def ncc(a, b):
    a = a.ravel().astype(np.float32); b = b.ravel().astype(np.float32)
    a -= a.mean(); b -= b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    srcs = json.loads((AUDIT / 'raw' / 'SOURCES.json').read_text(encoding='utf-8'))['sources']

    ext = []
    for sid, s in srcs.items():
        for sub in ('train/images', 'valid/images', 'test/images'):
            d = REPO / s['extracted_to'] / sub
            if d.exists():
                ext += [(sid, p) for p in sorted(d.iterdir())
                        if p.suffix.lower() in IMG_EXT]

    eye = []
    for grp, d, subs in [
            ('EYECU_TRAIN', REPO / 'data/dataset_baseline/images', ['train']),
            ('EYECU_VAL', REPO / 'data/dataset_baseline/images', ['val']),
            ('EYECU_TEST', REPO / 'data/frames',
             ['como_2-0_sassuolo', 'manchester_city_v_liverpool', 'youth_2'])]:
        for sub in subs:
            p = d / sub
            if p.exists():
                eye += [(grp, f) for f in sorted(p.rglob('*'))
                        if f.is_file() and f.suffix.lower() in IMG_EXT]

    print(f'{len(ext)} external images vs {len(eye)} EyeCU images '
          f'({sum(1 for g, _ in eye if g == "EYECU_TRAIN")} train / '
          f'{sum(1 for g, _ in eye if g == "EYECU_VAL")} val / '
          f'{sum(1 for g, _ in eye if g == "EYECU_TEST")} test), 8 orientations each')

    esig = load([p for _, p in eye])
    ok = [i for i, s in enumerate(esig) if s is not None]
    EH = np.stack([dhash(esig[i]) for i in ok])

    hits = []
    for n, (sid, p) in enumerate(ext):
        s = load([p])[0]
        if s is None:
            continue
        for k, t in enumerate(dihedral(s)):
            h = dhash(t)
            d = POP[np.bitwise_xor(EH, h)].sum(axis=1)
            for w in np.where(d <= HAM)[0]:
                j = ok[int(w)]
                v = ncc(t, esig[j])
                moved = float((np.abs(t.astype(np.float32)
                                      - esig[j].astype(np.float32)) > 12).mean())
                if v >= 0.95:
                    hits.append({
                        'external': str(p.relative_to(REPO)).replace('\\', '/'),
                        'source': sid,
                        'eyecu': str(eye[j][1].relative_to(REPO)).replace('\\', '/'),
                        'eyecu_split': eye[j][0],
                        'orientation': ['rot0', 'rot0+mirror', 'rot90', 'rot90+mirror',
                                        'rot180', 'rot180+mirror', 'rot270',
                                        'rot270+mirror'][k],
                        'hamming': int(d[w]), 'ncc': round(v, 4),
                        'moved_frac': round(moved, 4),
                        'verdict': ('HIGH_CONFIDENCE_NEAR_DUPLICATE'
                                    if v >= 0.985 and moved <= 0.02
                                    else 'POSSIBLE_DUPLICATE_REVIEW'),
                    })
        if (n + 1) % 400 == 0:
            print(f'  {n + 1}/{len(ext)}  hits so far {len(hits)}')

    per = {}
    for h in hits:
        k = f"{h['source']}->{h['eyecu_split']}"
        per.setdefault(k, {'pairs': 0, 'external_images': set(),
                           'eyecu_images': set(), 'orientations': {},
                           'HIGH_CONFIDENCE_NEAR_DUPLICATE': 0,
                           'POSSIBLE_DUPLICATE_REVIEW': 0})
        e = per[k]
        e['pairs'] += 1
        e['external_images'].add(h['external'])
        e['eyecu_images'].add(h['eyecu'])
        e['orientations'][h['orientation']] = e['orientations'].get(h['orientation'], 0) + 1
        e[h['verdict']] += 1
    out = {k: {**v, 'external_images': len(v['external_images']),
               'eyecu_images': len(v['eyecu_images'])} for k, v in per.items()}

    report = {
        'method': ('8 dihedral orientations of every external image, hamming <= '
                   f'{HAM} candidate stage on a 64-bit dhash of a letterbox-'
                   'stripped 64x64 signature, verified by NCC >= 0.95'),
        'test_note': ('EyeCU TEST participated as IMAGES ONLY, for hashing. No '
                      'TEST label was read and no evaluation was performed.'),
        'counts': {'external_images': len(ext), 'eyecu_images': len(eye)},
        'by_source_and_split': out,
        'EXTERNAL_vs_TRAIN': sum(v['pairs'] for k, v in out.items() if k.endswith('TRAIN')),
        'EXTERNAL_vs_VAL': sum(v['pairs'] for k, v in out.items() if k.endswith('VAL')),
        'EXTERNAL_vs_TEST': sum(v['pairs'] for k, v in out.items() if k.endswith('TEST')),
    }
    (AUDIT / 'reports' / 'leakage.json').write_text(
        json.dumps({'summary': report, 'hits': hits}, indent=1), encoding='utf-8')

    print('\nLEAKAGE (orientation-invariant)')
    for k, v in sorted(out.items()):
        print(f'  {k:<24} {v["pairs"]:>5} pairs  {v["external_images"]:>5} external images  '
              f'{v["eyecu_images"]:>5} EyeCU images  orientations={v["orientations"]}')
    print(f'\n  EXTERNAL vs TRAIN: {report["EXTERNAL_vs_TRAIN"]}')
    print(f'  EXTERNAL vs VAL  : {report["EXTERNAL_vs_VAL"]}')
    print(f'  EXTERNAL vs TEST : {report["EXTERNAL_vs_TEST"]}')


if __name__ == '__main__':
    main()
