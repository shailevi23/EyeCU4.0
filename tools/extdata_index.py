#!/usr/bin/env python
"""
Stages 4/6/12: annotation completeness, domain relevance, candidate index.

Shot type is inferred from annotation geometry, not guessed: a wide tactical
broadcast frame contains many humans whose boxes are a small fraction of the
frame height, a close-up contains few large ones. The thresholds were set by
looking at the contact sheets first and then choosing cuts that reproduce what
is visible there -- the proxy is calibrated against the images, not the other
way round.

The index assigns exactly one provisional status per image, most severe first,
and records every reason that applied. Nothing is copied, moved, relabelled or
deleted. This file is metadata about images that stay exactly where they are.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / 'experiments' / 'external_data_audit'
HUMAN = {'player', 'goalkeeper', 'referee'}

# Orientation screen. Two of the six exports contain 90-degree-rotated copies --
# S2 declares them, S4 does not and has 68 anyway. A sideways pitch is not a
# thing a broadcast detector should ever be trained on, so rotated copies have
# to be found rather than assumed absent.
#
# The signal: in an upright broadcast frame the bottom third is far grassier
# than the top third, and left and right are alike. Rotating swaps that.
# Thresholds were chosen against the 272 S4 images whose true orientation is
# KNOWN, because the leakage pass aligned each of them to its EyeCU TRAIN twin
# (204 upright, 68 rotated). At margin 0.15 that set gives 2/272 confident-wrong
# and 49/272 uncertain. Selected on labelled data, so this is a SCREEN, not a
# verified classifier: 'rotated' excludes, 'uncertain' goes to human review.
ORIENT_MARGIN = 0.15

# status precedence: the first that applies wins
PRECEDENCE = [
    'EXCLUDE_VAL_TEST_LEAKAGE',
    'EXCLUDE_EXACT_DUPLICATE',
    'EXCLUDE_AUGMENTATION_COPY',
    'EXCLUDE_NEAR_DUPLICATE',
    'EXCLUDE_PARTIAL_ANNOTATION_RISK',
    'EXCLUDE_POOR_LABEL',
    'EXCLUDE_IRRELEVANT',
    'HUMAN_REVIEW',
    'KEEP_CANDIDATE',
]


def orientation(p: Path):
    """upright / rotated / uncertain, from where the grass is."""
    import cv2
    a = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if a is None:
        return 'uncertain', 0.0, 0.0
    # Strip letterboxing first. S6 pads 16:9 into a square, so its top and
    # bottom thirds are black bars; measuring grass there would score every S6
    # image 'uncertain' for a reason that has nothing to do with orientation.
    m = a.max(axis=2)
    rr, cc = np.where(m.max(axis=1) > 12)[0], np.where(m.max(axis=0) > 12)[0]
    if len(rr) > 8 and len(cc) > 8:
        a = a[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    a = a.astype(np.int16)
    b, g, r = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    grass = ((g > b + 12) & (g > r + 12)).astype(np.float32)
    h, w = grass.shape
    vert = float(grass[2 * h // 3:].mean() - grass[:h // 3].mean())
    horz = float(grass[:, 2 * w // 3:].mean() - grass[:, :w // 3].mean())
    if vert - abs(horz) > ORIENT_MARGIN:
        return 'upright', vert, horz
    if abs(horz) - vert > ORIENT_MARGIN:
        return 'rotated', vert, horz
    return 'uncertain', vert, horz


def shot_type(rec, boxes_h):
    """Broadcast shot classification from box geometry."""
    n_h = sum(v for k, v in rec.get('eyecu', {}).items() if k in HUMAN)
    if n_h == 0:
        return 'no_human_labels'
    med = float(np.median(boxes_h)) if len(boxes_h) else 0.0
    if n_h >= 8 and med < 0.14:
        return 'broadcast_wide_tactical'
    if med > 0.35 or n_h <= 2:
        return 'close_up'
    return 'broadcast_close_medium'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    srcs = json.loads((AUDIT / 'raw' / 'SOURCES.json').read_text(encoding='utf-8'))['sources']
    inv = json.loads((AUDIT / 'reports' / 'inventory.json').read_text(encoding='utf-8'))
    cmap = json.loads((AUDIT / 'reports' / 'class_map.json').read_text(encoding='utf-8'))
    recs = json.loads((AUDIT / 'reports' / 'image_records.json').read_text(encoding='utf-8'))
    pairs = json.loads((AUDIT / 'reports' / 'pair_verdicts.json').read_text(encoding='utf-8'))
    leak = json.loads((AUDIT / 'reports' / 'leakage.json').read_text(encoding='utf-8'))

    by_path = {r['path']: r for r in recs}

    # ---- leakage lookup -----------------------------------------------------
    leak_train, leak_valtest = set(), set()
    for h in leak['hits']:
        (leak_valtest if h['eyecu_split'] in ('EYECU_VAL', 'EYECU_TEST')
         else leak_train).add(h['external'])

    # ---- exact duplicates and near-duplicates -------------------------------
    exact_partner, near_partner = defaultdict(set), defaultdict(set)
    for p in pairs:
        if p['ga'].startswith('EYECU') or p['gb'].startswith('EYECU'):
            continue
        if p['verdict'] == 'EXACT_DUPLICATE':
            exact_partner[p['a']].add(p['b']); exact_partner[p['b']].add(p['a'])
        elif p['verdict'] == 'HIGH_CONFIDENCE_NEAR_DUPLICATE':
            near_partner[p['a']].add(p['b']); near_partner[p['b']].add(p['a'])

    # keep one representative per exact / near group; flag the rest
    seen_exact, seen_near = set(), set()
    drop_exact, drop_near = set(), set()
    for a in sorted(exact_partner):
        if a in seen_exact:
            continue
        grp = sorted({a} | exact_partner[a])
        seen_exact.update(grp)
        drop_exact.update(grp[1:])
    for a in sorted(near_partner):
        if a in seen_near:
            continue
        grp = sorted({a} | near_partner[a])
        seen_near.update(grp)
        drop_near.update(grp[1:])

    # ---- orientation screen -------------------------------------------------
    print('orientation screen')
    orient = {}
    for r in recs:
        if r['group'].startswith('EYECU') or r.get('corrupt'):
            continue
        o, v, h = orientation(REPO / r['path'])
        orient[r['path']] = {'orientation': o, 'vertical': round(v, 4),
                             'horizontal': round(h, 4)}
    (AUDIT / 'reports' / 'orientation.json').write_text(
        json.dumps(orient, indent=0), encoding='utf-8')
    print('  ', Counter(v['orientation'] for v in orient.values()))

    # ---- augmentation copies: keep one export per Roboflow source stem ------
    # Roboflow only augments the training split, so a valid/test export is the
    # unaugmented original where one exists. Failing that, prefer an image the
    # orientation screen calls upright over one it calls rotated.
    by_stem = defaultdict(list)
    for r in recs:
        if r['group'].startswith('EYECU') or r.get('corrupt'):
            continue
        by_stem[(r['group'], r['stem'])].append(r['path'])
    drop_aug = set()
    for k, v in by_stem.items():
        def rank(p):
            split = p.split('/')[-3]
            return (0 if split in ('valid', 'test') else 1,
                    {'upright': 0, 'uncertain': 1, 'rotated': 2}[
                        orient.get(p, {}).get('orientation', 'uncertain')],
                    p)
        for p in sorted(v, key=rank)[1:]:
            drop_aug.add(p)

    # ---- per-source partial-annotation verdict ------------------------------
    partial = {}
    index, shots = [], defaultdict(Counter)
    for sid, src in srcs.items():
        per = json.loads((AUDIT / 'reports' / f'{sid}_per_image.json')
                         .read_text(encoding='utf-8'))
        root = REPO / src['extracted_to']
        names = src['declared_classes']
        to_eyecu = {int(k): v for k, v in cmap[sid]['mapping'].items()}
        has_human_class = any(v in HUMAN for v in to_eyecu.values())
        n = len(per)
        zero = [r for r in per if r['n'] == 0]
        no_human = [r for r in per
                    if sum(v for k, v in r.get('eyecu', {}).items() if k in HUMAN) == 0]

        if not has_human_class:
            verdict, why = 'BALL_ONLY', (
                'the export declares a single ball class; every image contains '
                'visible players that carry no label at all')
        elif len(zero) / n > 0.05:
            verdict, why = 'PARTIAL_ANNOTATION_LIKELY', (
                f'{len(zero)}/{n} images ({100 * len(zero) / n:.1f}%) carry zero boxes '
                f'while visibly containing players, officials and in places the ball '
                f'-- see {sid}_zero_box_images.jpg. Annotation appears to cover the '
                f'wide tactical camera only.')
        elif 'ball' not in to_eyecu.values():
            verdict, why = 'PLAYER_ONLY', 'no ball class in the taxonomy'
        else:
            verdict, why = 'FULL_MULTICLASS_LIKELY', (
                f'every image carries labels; {n - len(no_human)}/{n} contain human '
                f'boxes and {sum(1 for r in per if r.get("eyecu", {}).get("ball", 0))}'
                f'/{n} contain a ball')
        missing_gk = 'goalkeeper' not in to_eyecu.values()
        # Pixel-destroying augmentation cannot be undone, and where every
        # training export passed through it the audit cannot point at an
        # unaugmented original. Say so per image rather than quietly keeping one.
        aug = ' '.join(src['augmentation']).lower()
        degraded = ('declared augmentation includes ' +
                    ', '.join(a for a in ('blur', 'noise', 'crop')
                              if a in aug) +
                    '; the unaugmented original is not identifiable in this export'
                    ) if any(a in aug for a in ('blur', 'noise', 'crop')) else None
        partial[sid] = {'verdict': verdict, 'evidence': why,
                        'images': n, 'zero_box_images': len(zero),
                        'images_without_human_labels': len(no_human),
                        'goalkeeper_class_absent': missing_gk}

        # ---- per image ------------------------------------------------------
        for r in per:
            split = r['split']
            path = f"{src['extracted_to']}/{split}/images/{r['file']}"
            lp = root / split / 'labels' / f"{Path(r['file']).stem}.txt"
            hs = []
            if lp.exists():
                for line in lp.read_text(encoding='utf-8', errors='replace').splitlines():
                    f = line.split()
                    if len(f) < 5:
                        continue
                    ci = int(float(f[0]))
                    v = [float(x) for x in f[1:]]
                    if to_eyecu.get(ci) in HUMAN:
                        hs.append(v[3] if len(v) == 4
                                  else max(v[1::2]) - min(v[1::2]))
            st = shot_type(r, hs)
            shots[sid][st] += 1

            reasons, status = [], None
            if path in leak_valtest:
                reasons.append('overlaps EyeCU VAL or TEST')
                status = 'EXCLUDE_VAL_TEST_LEAKAGE'
            if path in leak_train:
                reasons.append('same frame already in EyeCU TRAIN')
                status = status or 'EXCLUDE_NEAR_DUPLICATE'
            if path in drop_exact:
                reasons.append('pixel-identical to another image in this source')
                status = status or 'EXCLUDE_EXACT_DUPLICATE'
            if path in drop_aug:
                reasons.append('Roboflow augmentation copy of an earlier export '
                               'of the same source image')
                status = status or 'EXCLUDE_AUGMENTATION_COPY'
            if path in drop_near:
                reasons.append('near-duplicate of another image in this source')
                status = status or 'EXCLUDE_NEAR_DUPLICATE'
            if verdict in ('BALL_ONLY', 'PARTIAL_ANNOTATION_LIKELY'):
                if verdict == 'BALL_ONLY':
                    reasons.append('ball-only source: visible players are unlabelled')
                    status = status or 'EXCLUDE_PARTIAL_ANNOTATION_RISK'
                elif r['n'] == 0:
                    reasons.append('zero boxes on an image with visible people')
                    status = status or 'EXCLUDE_PARTIAL_ANNOTATION_RISK'
            o = orient.get(path, {}).get('orientation', 'uncertain')
            if o == 'rotated':
                reasons.append('orientation screen: 90-degree-rotated copy, a '
                               'sideways pitch that cannot occur in broadcast')
                status = status or 'EXCLUDE_IRRELEVANT'
            if missing_gk and sum(v for k, v in r.get('eyecu', {}).items()
                                  if k in HUMAN):
                reasons.append('source has no goalkeeper class: goalkeepers are '
                               'labelled player, which conflicts with EyeCU')
                status = status or 'HUMAN_REVIEW'
            if o == 'uncertain':
                reasons.append('orientation screen inconclusive')
                status = status or 'HUMAN_REVIEW'
            if degraded:
                reasons.append(degraded)
                status = status or 'HUMAN_REVIEW'
            if not status:
                status = 'KEEP_CANDIDATE'
            index.append({'source': sid, 'path': path, 'split_in_source': split,
                          'shot_type': st, 'boxes': r['n'], 'orientation': o,
                          'eyecu_counts': r.get('eyecu', {}),
                          'status': status, 'reasons': reasons})

    (AUDIT / 'candidate_index' / 'candidate_index.json').write_text(
        json.dumps(index, indent=0), encoding='utf-8')

    summary = {'per_source': {}, 'totals': Counter()}
    for sid in srcs:
        rows = [r for r in index if r['source'] == sid]
        c = Counter(r['status'] for r in rows)
        summary['per_source'][sid] = {
            'images': len(rows),
            'status': dict(c),
            'shot_types': dict(shots[sid]),
            'annotation_completeness': partial[sid],
        }
        summary['totals'].update(c)
    summary['totals'] = dict(summary['totals'])
    (AUDIT / 'candidate_index' / 'summary.json').write_text(
        json.dumps(summary, indent=1), encoding='utf-8')
    (AUDIT / 'reports' / 'annotation_completeness.json').write_text(
        json.dumps(partial, indent=1), encoding='utf-8')

    print(f'{"src":<5}{"imgs":>6}  {"completeness":<28} {"KEEP":>6}{"REVIEW":>8}{"EXCL":>7}')
    for sid, v in summary['per_source'].items():
        s = v['status']
        print(f'{sid:<5}{v["images"]:>6}  {partial[sid]["verdict"]:<28} '
              f'{s.get("KEEP_CANDIDATE", 0):>6}{s.get("HUMAN_REVIEW", 0):>8}'
              f'{sum(n for k, n in s.items() if k.startswith("EXCLUDE")):>7}')
    print('\nTOTALS', summary['totals'])
    print('\nSHOT TYPES')
    for sid in srcs:
        print(f'  {sid}: {dict(shots[sid])}')


if __name__ == '__main__':
    main()
