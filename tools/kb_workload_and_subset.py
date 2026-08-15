#!/usr/bin/env python
"""
Turn the keremberke triage into a human workload figure, and test the clean-subset option.

TWO QUESTIONS.

WORKLOAD. 21,615 boxes is not 21,615 decisions. The frames are consecutive
within four broadcast runs, so a candidate box in frame N and one in frame N+1
at nearly the same place are the same person, and one human decision settles
both. Boxes are linked into tracklets by IoU between ADJACENT frames only --
a link that consecutive-frame evidence actually supports. No identity is
asserted across a gap, across a run, or where the geometry does not carry it;
those simply become separate tracklets, which overestimates the workload rather
than understating it.

CLEAN SUBSET. Is there a substantial set of images where every labelled human is
convincingly a field player, so the four-class ontology is not violated by using
them as-is? The bar is deliberately strict: an image qualifies only if EVERY one
of its human boxes is LIKELY_PLAYER with BOTH independent signals agreeing. One
uncertain official disqualifies the whole image, because a wrong `player` box is
exactly the damage this task exists to prevent.
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
BINS = [(0, 3, '<3'), (3, 5, '3-5'), (5, 8, '>5-8'), (8, 12, '>8-12'),
        (12, 20, '>12-20'), (20, 40, '>20-40'), (40, 1e9, '>40')]


def bin_of(w):
    for lo, hi, n in BINS:
        if lo == 0 and w < hi:
            return n
        if lo < w <= hi:
            return n
    return '>40'


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def frame_index(stem):
    m = re.match(r'^(\d+)(_pp)?[_.]', stem)
    return int(m.group(1)) if m else None


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    rows = json.loads((OUT / 'reports' / 'keremberke_role_triage_boxes.json')
                      .read_text(encoding='utf-8'))
    for r in rows:
        r['idx'] = frame_index(r['stem'])
    print(f'{len(rows)} human boxes')

    # ---- tracklets over ADJACENT frames within a run ------------------------
    by_run = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['idx'] is not None:
            by_run[r['run']][r['idx']].append(r)

    tracklets = []
    for run, frames in by_run.items():
        order = sorted(frames)
        open_tracks = []           # list of (last_idx, last_bbox, triages)
        for fi in order:
            cur = frames[fi]
            used = set()
            nxt = []
            for last_idx, last_box, tri, nboxes in open_tracks:
                if fi - last_idx != 1:          # only truly adjacent frames link
                    tracklets.append((run, tri, nboxes))
                    continue
                best, bj = 0.0, None
                for j, c in enumerate(cur):
                    if j in used:
                        continue
                    v = iou_xywh(last_box, c['bbox'])
                    if v > best:
                        best, bj = v, j
                if bj is not None and best >= 0.4:
                    used.add(bj)
                    tri2 = Counter(tri); tri2[cur[bj]['triage']] += 1
                    nxt.append((fi, cur[bj]['bbox'], tri2, nboxes + 1))
                else:
                    tracklets.append((run, tri, nboxes))
            for j, c in enumerate(cur):
                if j not in used:
                    nxt.append((fi, c['bbox'], Counter({c['triage']: 1}), 1))
            open_tracks = nxt
        for last_idx, last_box, tri, nboxes in open_tracks:
            tracklets.append((run, tri, nboxes))

    def track_label(tri):
        """A tracklet needs review if ANY of its boxes is not a clean player."""
        if tri.get('POSSIBLE_GOALKEEPER', 0) or tri.get('POSSIBLE_REFEREE', 0) \
                or tri.get('AMBIGUOUS', 0):
            if tri.get('POSSIBLE_GOALKEEPER', 0) >= tri.get('POSSIBLE_REFEREE', 0) \
                    and tri.get('POSSIBLE_GOALKEEPER', 0) > 0:
                return 'REVIEW_GK_CANDIDATE'
            if tri.get('POSSIBLE_REFEREE', 0) > 0:
                return 'REVIEW_REF_CANDIDATE'
            return 'REVIEW_AMBIGUOUS'
        return 'CLEAN_PLAYER_TRACK'

    tl = Counter(track_label(t[1]) for t in tracklets)
    lens = [t[2] for t in tracklets]
    review_tracks = [t for t in tracklets if track_label(t[1]) != 'CLEAN_PLAYER_TRACK']
    review_boxes = sum(t[2] for t in review_tracks)
    print(f'\n{len(tracklets)} tracklets from adjacent-frame linking '
          f'(median length {int(np.median(lens))}, max {max(lens)})')
    for k, v in tl.most_common():
        print(f'   {k:<26} {v:>6}')
    print(f'   tracklets needing a human decision: {len(review_tracks)} '
          f'covering {review_boxes} boxes '
          f'({review_boxes/max(len(review_tracks),1):.1f} boxes per decision)')

    # ---- clean subset --------------------------------------------------------
    per_img = defaultdict(list)
    for r in rows:
        per_img[(r['split'], r['file'])].append(r)
    clean_imgs, dirty = set(), Counter()
    for k, v in per_img.items():
        if all(x['triage'] == 'LIKELY_PLAYER' and x['agreement'] == 'BOTH_SIGNALS'
               for x in v):
            clean_imgs.add(k)
        else:
            dirty[Counter(x['triage'] for x in v).most_common(1)[0][0]] += 1

    ball = json.loads((SRC / 'manifests' / 'ball_instances.json').read_text(encoding='utf-8'))
    clean_files = {f for _, f in clean_imgs}
    cb = [b for b in ball if b['file'] in clean_files]
    w = np.array([b['w'] for b in cb]) if cb else np.array([0.0])
    nw = np.array([b['w_native1920'] for b in cb]) if cb else np.array([0.0])
    runs_in_clean = Counter()
    for r in rows:
        if (r['split'], r['file']) in clean_imgs:
            runs_in_clean[r['run']] += 1

    total_imgs = len(per_img)
    print(f'\nCLEAN SUBSET: {len(clean_imgs)}/{total_imgs} images '
          f'({100*len(clean_imgs)/total_imgs:.1f}%)')
    print(f'   disqualifying reason (dominant): {dict(dirty)}')
    print(f'   ball instances retained: {len(cb)} of {len(ball)}')
    if cb:
        print(f'   stored px: median {np.median(w):.2f}  '
              f'<=5 {(w<=5).sum()}  <=8 {(w<=8).sum()}  <=12 {(w<=12).sum()}')
        print(f'   1920-equiv: median {np.median(nw):.2f}  '
              f'<=5 {(nw<=5).sum()}  <=8 {(nw<=8).sum()}  <=12 {(nw<=12).sum()}')
    print(f'   broadcast runs represented: {dict(runs_in_clean)}')

    rep = {
        'total_human_boxes': len(rows),
        'triage': dict(Counter(r['triage'] for r in rows)),
        'tracklets': {
            'method': ('IoU >= 0.4 between ADJACENT frames only, within one '
                       'broadcast run; no identity asserted across a gap or a run'),
            'count': len(tracklets),
            'median_length': int(np.median(lens)), 'max_length': int(max(lens)),
            'labels': dict(tl),
            'tracklets_needing_a_human_decision': len(review_tracks),
            'boxes_covered_by_those_tracklets': review_boxes,
            'boxes_settled_per_decision': round(review_boxes / max(len(review_tracks), 1), 1),
        },
        'clean_subset': {
            'rule': ('every human box in the image is LIKELY_PLAYER AND both the '
                     'detector proposal and the independent kit signal agree; one '
                     'uncertain official disqualifies the whole image'),
            'images': len(clean_imgs), 'of_total': total_imgs,
            'ball_instances': len(cb),
            'ball_stored_px': {'median': round(float(np.median(w)), 2),
                               'le5': int((w <= 5).sum()), 'le8': int((w <= 8).sum()),
                               'le12': int((w <= 12).sum())} if cb else None,
            'ball_1920_equivalent': {'median': round(float(np.median(nw)), 2),
                                     'le5': int((nw <= 5).sum()),
                                     'le8': int((nw <= 8).sum()),
                                     'le12': int((nw <= 12).sum())} if cb else None,
            'broadcast_runs_represented': dict(runs_in_clean),
            'human_verifiable': True,
            'caveat': ('"clean" here means NO SIGNAL FLAGGED IT, which is not the '
                       'same as verified. A human must still confirm the subset '
                       'before it is used; the subset is small enough that this is '
                       'feasible, which is the point.'),
        },
    }
    (OUT / 'reports' / 'keremberke_workload.json').write_text(
        json.dumps(rep, indent=1), encoding='utf-8')
    (OUT / 'reports' / 'keremberke_clean_subset.json').write_text(
        json.dumps(sorted(f'{s}/{f}' for s, f in clean_imgs), indent=0),
        encoding='utf-8')
    print('\nwrote reports/keremberke_workload.json and keremberke_clean_subset.json')


if __name__ == '__main__':
    main()
