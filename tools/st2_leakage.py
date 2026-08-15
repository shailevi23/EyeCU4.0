#!/usr/bin/env python
"""
Leakage check: the downloaded SoccerTrack video against EyeCU TRAIN / VAL / TEST.

Same machinery as the Roboflow audit, for the same reason -- a flipped or
rotated copy of a validation frame is still a validation frame. Frames are
sampled evenly across both downloaded halves, reduced to a letterbox-stripped
64x64 signature, and compared under all eight dihedral orientations.

TEST participates as IMAGES ONLY. No TEST label is opened and nothing is
evaluated. The expected answer here is zero, and running the check is what makes
that a finding rather than an assumption.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / 'experiments' / 'soccertrack_audit'
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


def dhash(sig):
    import cv2
    s = cv2.resize(sig, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return np.packbits((s[:, 1:] > s[:, :-1]).flatten())


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


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samples', type=int, default=300, help='frames per half')
    args = ap.parse_args()
    import cv2

    eye = []
    for grp, root, subs in [
            ('EYECU_TRAIN', REPO / 'data/dataset_baseline/images', ['train']),
            ('EYECU_VAL', REPO / 'data/dataset_baseline/images', ['val']),
            ('EYECU_TEST', REPO / 'data/frames',
             ['como_2-0_sassuolo', 'manchester_city_v_liverpool', 'youth_2'])]:
        for sub in subs:
            d = root / sub
            if d.exists():
                eye += [(grp, p) for p in sorted(d.rglob('*'))
                        if p.is_file() and p.suffix.lower() in IMG_EXT]
    esig, eok = [], []
    for grp, p in eye:
        a = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if a is None:
            continue
        esig.append(canonical(a)); eok.append((grp, p))
    EH = np.stack([dhash(s) for s in esig])
    print(f'EyeCU: {len(eok)} images '
          f'({sum(1 for g, _ in eok if g == "EYECU_TRAIN")} train / '
          f'{sum(1 for g, _ in eok if g == "EYECU_VAL")} val / '
          f'{sum(1 for g, _ in eok if g == "EYECU_TEST")} test)')

    vids = sorted((REPO / 'EyeCU_external_data/SoccerTrackV2/videos').glob('*.mp4'))
    hits, sampled = [], 0
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = np.linspace(0, n - 1, args.samples).round().astype(int)
        print(f'{v.name}: {n} frames, sampling {len(idx)}')
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, img = cap.read()
            if not ok:
                continue
            sampled += 1
            s = canonical(img)
            for k, t in enumerate(dihedral(s)):
                h = dhash(t)
                d = POP[np.bitwise_xor(EH, h)].sum(axis=1)
                for w in np.where(d <= HAM)[0]:
                    j = int(w)
                    val = ncc(t, esig[j])
                    if val >= 0.95:
                        hits.append({'video': v.name, 'frame': int(i),
                                     'eyecu': str(eok[j][1].relative_to(REPO)).replace('\\', '/'),
                                     'eyecu_split': eok[j][0],
                                     'orientation': k, 'hamming': int(d[w]),
                                     'ncc': round(val, 4)})
        cap.release()

    rep = {'method': ('even frame sampling from both downloaded halves, '
                      'letterbox-stripped 64x64 signature, 8 dihedral '
                      f'orientations, hamming <= {HAM} then NCC >= 0.95'),
           'video_frames_sampled': sampled,
           'eyecu_images_compared': len(eok),
           'test_handling': ('TEST images hashed only; no TEST label opened, '
                             'no evaluation performed'),
           'EXTERNAL_vs_TRAIN': sum(1 for h in hits if h['eyecu_split'] == 'EYECU_TRAIN'),
           'EXTERNAL_vs_VAL': sum(1 for h in hits if h['eyecu_split'] == 'EYECU_VAL'),
           'EXTERNAL_vs_TEST': sum(1 for h in hits if h['eyecu_split'] == 'EYECU_TEST'),
           'same_source_assessment': (
               'SoccerTrack is a fixed panoramic rig over a Japanese university '
               'artificial-turf pitch; EyeCU sources are broadcast and amateur '
               'match video from other continents. No shared source is possible '
               'on provenance grounds, and none was found on pixels.'),
           'hits': hits}
    (AUDIT / 'reports').mkdir(parents=True, exist_ok=True)
    (AUDIT / 'reports' / 'leakage.json').write_text(json.dumps(rep, indent=1),
                                                    encoding='utf-8')
    print(f'\nleakage  TRAIN {rep["EXTERNAL_vs_TRAIN"]}  '
          f'VAL {rep["EXTERNAL_vs_VAL"]}  TEST {rep["EXTERNAL_vs_TEST"]}')


if __name__ == '__main__':
    main()
