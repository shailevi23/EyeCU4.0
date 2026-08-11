#!/usr/bin/env python
"""
Orientation-invariant leakage check for any new external image set.

Same machinery as the Roboflow audit, generalised: a letterbox-stripped 64x64
signature, a dhash candidate stage, then NCC verification, with every external
image tested under all eight dihedral orientations. The rotation-sensitive
version of this check missed 72 real matches in the Roboflow audit, so the
orientation sweep is not optional.

Compares against EyeCU TRAIN, VAL and TEST, and against the six previously
audited Roboflow sources, because a "new" source that duplicates an already
rejected one is not new.

TEST participates as IMAGES ONLY. No TEST label is read and nothing is scored.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / 'EyeCU_external_data'
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


def dhash(s):
    import cv2
    x = cv2.resize(s, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return np.packbits((x[:, 1:] > x[:, :-1]).flatten())


def dihedral(s):
    out = []
    for k in range(4):
        r = np.rot90(s, k)
        out.append(np.ascontiguousarray(r))
        out.append(np.ascontiguousarray(np.fliplr(r)))
    return out


def ncc(a, b):
    a = a.ravel().astype(np.float32); b = b.ravel().astype(np.float32)
    a -= a.mean(); b -= b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def collect(label, root, patterns=('**/*',)):
    out = []
    for pat in patterns:
        for p in sorted(Path(root).glob(pat)):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                out.append((label, p))
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--new', required=True, help='directory of the new source')
    ap.add_argument('--name', required=True)
    ap.add_argument('--max-new', type=int, default=0, help='cap sampled new images')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import cv2

    ref = []
    ref += collect('EYECU_TRAIN', REPO / 'data/dataset_baseline/images/train')
    ref += collect('EYECU_VAL', REPO / 'data/dataset_baseline/images/val')
    for m in ('como_2-0_sassuolo', 'manchester_city_v_liverpool', 'youth_2'):
        ref += collect('EYECU_TEST', REPO / 'data/frames' / m)
    rb = REPO / 'experiments/external_data_audit/extracted'
    for sid in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6'):
        if (rb / sid).exists():
            ref += collect(f'ROBOFLOW_{sid}', rb / sid)

    new = collect(args.name, REPO / args.new)
    if args.max_new and len(new) > args.max_new:
        idx = np.linspace(0, len(new) - 1, args.max_new).round().astype(int)
        new = [new[i] for i in dict.fromkeys(idx.tolist())]
    print(f'{len(new)} new images vs {len(ref)} reference images '
          f'({len({l for l, _ in ref})} collections)')

    rsig, rok = [], []
    for lab, p in ref:
        a = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if a is None:
            continue
        rsig.append(canonical(a)); rok.append((lab, p))
    RH = np.stack([dhash(s) for s in rsig])

    hits = []
    for i, (lab, p) in enumerate(new, 1):
        a = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if a is None:
            continue
        s = canonical(a)
        for k, t in enumerate(dihedral(s)):
            h = dhash(t)
            d = POP[np.bitwise_xor(RH, h)].sum(axis=1)
            for w in np.where(d <= HAM)[0]:
                j = int(w)
                v = ncc(t, rsig[j])
                if v >= 0.95:
                    moved = float((np.abs(t.astype(np.float32)
                                          - rsig[j].astype(np.float32)) > 12).mean())
                    hits.append({
                        'new': str(p.relative_to(REPO)).replace('\\', '/'),
                        'reference': str(rok[j][1].relative_to(REPO)).replace('\\', '/'),
                        'reference_group': rok[j][0], 'orientation': k,
                        'hamming': int(d[w]), 'ncc': round(v, 4),
                        'moved_frac': round(moved, 4),
                        'verdict': ('HIGH_CONFIDENCE_NEAR_DUPLICATE'
                                    if v >= 0.985 and moved <= 0.02
                                    else 'POSSIBLE_DUPLICATE_REVIEW')})
        if i % 250 == 0:
            print(f'  {i}/{len(new)}  hits {len(hits)}')

    groups = {}
    for h in hits:
        g = h['reference_group']
        e = groups.setdefault(g, {'pairs': 0, 'new_images': set(),
                                  'reference_images': set(),
                                  'HIGH_CONFIDENCE_NEAR_DUPLICATE': 0,
                                  'POSSIBLE_DUPLICATE_REVIEW': 0})
        e['pairs'] += 1
        e['new_images'].add(h['new']); e['reference_images'].add(h['reference'])
        e[h['verdict']] += 1
    summary = {k: {**v, 'new_images': len(v['new_images']),
                   'reference_images': len(v['reference_images'])}
               for k, v in groups.items()}
    rep = {'source': args.name, 'new_images_compared': len(new),
           'reference_images': len(rok),
           'method': ('8 dihedral orientations, dhash hamming <= 10 candidate, '
                      'NCC >= 0.95 verification'),
           'test_handling': ('EyeCU TEST images hashed only; no TEST label read, '
                             'no evaluation performed'),
           'by_reference_group': summary,
           'EXTERNAL_vs_EYECU_TRAIN': summary.get('EYECU_TRAIN', {}).get('pairs', 0),
           'EXTERNAL_vs_EYECU_VAL': summary.get('EYECU_VAL', {}).get('pairs', 0),
           'EXTERNAL_vs_EYECU_TEST': summary.get('EYECU_TEST', {}).get('pairs', 0),
           'hits': hits[:500]}
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f'\nby reference group: {summary or "no hits"}')
    print(f'TRAIN {rep["EXTERNAL_vs_EYECU_TRAIN"]}  VAL {rep["EXTERNAL_vs_EYECU_VAL"]}  '
          f'TEST {rep["EXTERNAL_vs_EYECU_TEST"]}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
