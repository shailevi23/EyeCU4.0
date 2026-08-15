#!/usr/bin/env python
"""
Keremberke review workload and clean-subset options, corrected.

CORRECTION THIS FILE EXISTS FOR. The first pass assumed consecutive filename
indices were consecutive VIDEO frames and linked candidate boxes by IoU across
them. They are not: measured over 120 consecutive-id pairs the frame-to-frame
correlation is ~0.52 median, with 55 of 60 plain-family pairs below 0.8. The
images are sparsely sampled from four broadcast runs, players are in different
places between them, and IoU propagation is therefore invalid -- which is
exactly why those tracklets came out with a median length of 1.

So identity is NOT propagated. What survives is a weaker but real grouping: in
one broadcast run the officials and keepers are the same few people wearing the
same few kits all match, so candidate boxes cluster by kit colour into a small
number of ROLE GROUPS. That is a fact about football, not an invented ID, and it
is what makes the review tractable: a human decides "in run X, the yellow kit is
the referee" once, then applies it.

The clean subset is reported at THREE strictness levels rather than one, because
the single strict rule used first returned 17 images and that number says more
about the rule than about the data.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'experiments' / 'external_sources'
SRC = REPO / 'EyeCU_external_data/huggingface/keremberke_football_object_detection'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import cv2
    rows = json.loads((OUT / 'reports' / 'keremberke_role_triage_boxes.json')
                      .read_text(encoding='utf-8'))
    print(f'{len(rows)} human boxes')

    # ---- kit-colour role groups per run -------------------------------------
    # recompute torso hue for candidate boxes only (the ones a human must judge)
    paths = {}
    for split in ('train', 'valid', 'test'):
        aj = list((SRC / 'extracted' / split).rglob('_annotations.coco.json'))
        if aj:
            for p in (aj[0].parent).glob('*.jpg'):
                paths[(split, p.name)] = p
    cands = [r for r in rows if r['triage'] != 'LIKELY_PLAYER']
    print(f'{len(cands)} candidate boxes needing judgement')

    feats = defaultdict(list)
    cache = {}
    for r in cands:
        p = paths.get((r['split'], r['file']))
        if p is None:
            continue
        img = cache.get(p)
        if img is None:
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if len(cache) > 60:
                cache.clear()
            cache[p] = img
        if img is None:
            continue
        x, y, w, h = r['bbox']
        H, W = img.shape[:2]
        cx1 = max(0, int(x + 0.2 * w)); cx2 = min(W, int(x + 0.8 * w))
        cy1 = max(0, int(y + 0.20 * h)); cy2 = min(H, int(y + 0.55 * h))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        hsv = cv2.cvtColor(img[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2HSV)
        med = np.median(hsv.reshape(-1, 3), axis=0)
        feats[r['run']].append((r, med))

    groups = {}
    for run, items in feats.items():
        # coarse 15-degree hue bins x 2 value bands: enough to separate a yellow
        # referee from a green keeper from a white player, and no finer, because
        # finer would invent distinctions the pixels do not support
        buckets = Counter()
        for r, med in items:
            hb = int(med[0] // 15)
            vb = 0 if med[2] < 110 else 1
            sb = 0 if med[1] < 60 else 1
            buckets[(hb, sb, vb)] += 1
        # a group worth a human decision is one with enough support to be real
        real = {k: v for k, v in buckets.items() if v >= 10}
        groups[run] = {'candidate_boxes': len(items),
                       'distinct_kit_buckets': len(buckets),
                       'buckets_with_at_least_10_boxes': len(real),
                       'boxes_in_those_buckets': int(sum(real.values())),
                       'largest_buckets': [[list(map(int, k)), int(v)]
                                           for k, v in buckets.most_common(6)]}
        print(f'  {run:<9} {len(items):>5} candidates -> '
              f'{len(buckets)} kit buckets, {len(real)} with >=10 boxes')

    total_groups = sum(g['buckets_with_at_least_10_boxes'] for g in groups.values())

    # ---- clean subset at three strictness levels ----------------------------
    per_img = defaultdict(list)
    for r in rows:
        per_img[(r['split'], r['file'])].append(r)
    ball = json.loads((SRC / 'manifests' / 'ball_instances.json')
                      .read_text(encoding='utf-8'))
    ball_by_file = defaultdict(list)
    for b in ball:
        ball_by_file[b['file']].append(b)

    levels = {
        'STRICT_both_signals_agree_on_every_box': lambda v: all(
            x['triage'] == 'LIKELY_PLAYER' and x['agreement'] == 'BOTH_SIGNALS'
            for x in v),
        'MODERATE_no_candidate_and_no_ambiguous_box': lambda v: all(
            x['triage'] == 'LIKELY_PLAYER' for x in v),
        'PERMISSIVE_no_gk_or_referee_candidate': lambda v: all(
            x['triage'] not in ('POSSIBLE_GOALKEEPER', 'POSSIBLE_REFEREE')
            for x in v),
    }
    subsets = {}
    for name, fn in levels.items():
        imgs = {k for k, v in per_img.items() if fn(v)}
        files = {f for _, f in imgs}
        cb = [b for f in files for b in ball_by_file.get(f, [])]
        w = np.array([b['w'] for b in cb]) if cb else np.array([])
        nw = np.array([b['w_native1920'] for b in cb]) if cb else np.array([])
        runs = Counter(r['run'] for r in rows if (r['split'], r['file']) in imgs)
        subsets[name] = {
            'images': len(imgs),
            'share_of_1232': round(100 * len(imgs) / len(per_img), 1),
            'ball_instances': len(cb),
            'human_boxes_retained': sum(len(per_img[k]) for k in imgs),
            'stored_px': {'median': round(float(np.median(w)), 2) if len(w) else None,
                          'le5': int((w <= 5).sum()), 'le8': int((w <= 8).sum()),
                          'le12': int((w <= 12).sum())} if len(w) else None,
            'px_1920_equiv': {'median': round(float(np.median(nw)), 2) if len(nw) else None,
                              'le5': int((nw <= 5).sum()), 'le8': int((nw <= 8).sum()),
                              'le12': int((nw <= 12).sum())} if len(nw) else None,
            'broadcast_runs': dict(runs),
        }
        s = subsets[name]
        print(f'\n{name}: {s["images"]} images ({s["share_of_1232"]}%), '
              f'{s["ball_instances"]} balls')
        if s['stored_px']:
            print(f'   stored  median {s["stored_px"]["median"]}  '
                  f'<=5 {s["stored_px"]["le5"]}  <=8 {s["stored_px"]["le8"]}  '
                  f'<=12 {s["stored_px"]["le12"]}')
            print(f'   1920eq  median {s["px_1920_equiv"]["median"]}  '
                  f'<=5 {s["px_1920_equiv"]["le5"]}  <=8 {s["px_1920_equiv"]["le8"]}  '
                  f'<=12 {s["px_1920_equiv"]["le12"]}')
        print(f'   runs {s["broadcast_runs"]}')

    rep = {
        'correction': ('the first workload pass assumed consecutive filename '
                       'indices were consecutive video frames and propagated '
                       'identity by IoU across them. Measured consecutive-id '
                       'correlation is ~0.52 median (55/60 plain pairs below '
                       '0.8), so the images are sparsely sampled and IoU '
                       'propagation is invalid. Identity is not propagated here.'),
        'total_human_boxes': len(rows),
        'triage': dict(Counter(r['triage'] for r in rows)),
        'candidate_boxes_needing_judgement': len(cands),
        'role_groups_by_kit_within_run': groups,
        'approximate_distinct_role_groups': total_groups,
        'workload_model': {
            'per_box_upper_bound': len(cands),
            'per_image_upper_bound': len({(r['split'], r['file']) for r in cands}),
            'role_group_lower_bound': total_groups,
            'reading': ('the true cost sits between the role-group count and the '
                        'per-image count: a reviewer settles a kit/role rule once '
                        'per run, then confirms it image by image where a '
                        'candidate appears'),
        },
        'clean_subset_levels': subsets,
    }
    (OUT / 'reports' / 'keremberke_workload.json').write_text(
        json.dumps(rep, indent=1), encoding='utf-8')
    print(f'\napprox distinct role groups across all runs: {total_groups}')
    print('wrote reports/keremberke_workload.json')


if __name__ == '__main__':
    main()
