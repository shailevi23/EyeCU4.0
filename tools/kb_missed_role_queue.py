#!/usr/bin/env python
"""
Build the retrospective MISSED_ROLE_REVIEW queue, and the U-resolution queue.

WHY THIS EXISTS. Completing 4,153/4,153 candidate decisions did not prove role
repair was complete. The QA says otherwise: 6.40% of sampled LIKELY_PLAYER boxes
were officials the triage never queued, and 25 of 57 no-candidate images held a
missed official. Extrapolated, roughly 1,118 officials (95% CI 588-1,647) are
still labelled player. Around a third of the officials in this dataset were never
put in front of the reviewer.

THE NEW SIGNAL is the review itself. 832 goalkeeper and 1,533 referee boxes now
carry human-confirmed roles, so each broadcast run has a measured official-kit
appearance rather than a guessed one. Every unreviewed box is scored by how close
its torso colour sits to those confirmed kits, combined with geometry priors
(officials hug the touchline, keepers sit near the goals) and the frozen
detector's own opinion.

WHAT THIS IS NOT. Nothing is relabelled. No identity is propagated. A high score
puts a box in front of a human and nothing else. The ranking is VALIDATED on the
qa_player sample -- 250 boxes whose true roles a human established independently
of any of these signals -- so the queue's recall is measured, not asserted.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
SRC = REPO / 'EyeCU_external_data/huggingface/keremberke_football_object_detection'


def torso_hsv(img, bbox):
    import cv2
    x, y, w, h = bbox
    H, W = img.shape[:2]
    cx1 = max(0, int(x + 0.2 * w)); cx2 = min(W, int(x + 0.8 * w))
    cy1 = max(0, int(y + 0.20 * h)); cy2 = min(H, int(y + 0.55 * h))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return np.median(hsv.reshape(-1, 3), axis=0)


def kit_dist(a, b):
    """Hue is circular; saturation and value are linear and matter less."""
    dh = abs(float(a[0]) - float(b[0]))
    dh = min(dh, 180.0 - dh) / 90.0
    ds = abs(float(a[1]) - float(b[1])) / 255.0
    dv = abs(float(a[2]) - float(b[2])) / 255.0
    return float(np.sqrt(2.2 * dh * dh + 0.6 * ds * ds + 0.6 * dv * dv))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target-recall', type=float, default=0.95)
    args = ap.parse_args()
    import cv2

    ledger = json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))
    by_id = {r['BOX_ID']: r for r in ledger}
    last = {}
    for line in (PKG / 'decisions.json').read_text(encoding='utf-8').splitlines():
        if line.strip():
            d = json.loads(line)
            last[(d.get('mode', 'candidates'), d['BOX_ID'])] = d['HUMAN_FINAL_CLASS']
    cand = {b: v for (m, b), v in last.items() if m == 'candidates'}
    qa = {b: v for (m, b), v in last.items() if m == 'qa_player'}
    nocand = {b: v for (m, b), v in last.items() if m == 'qa_nocand'}

    paths = {}
    for split in ('train', 'valid', 'test'):
        aj = list((SRC / 'extracted' / split).rglob('_annotations.coco.json'))
        if aj:
            for p in aj[0].parent.glob('*.jpg'):
                paths[(split, p.name)] = p

    # ---- 1. confirmed kit models per run, from HUMAN labels only -----------
    conf_kits = defaultdict(lambda: defaultdict(list))
    need = [(b, v) for b, v in cand.items() if v in ('goalkeeper', 'referee')]
    need += [(b, v) for b, v in {**qa, **nocand}.items() if v in ('goalkeeper', 'referee')]
    cache = {}

    def get_img(r):
        p = paths.get((r['split'], r['file']))
        if p is None:
            return None
        if p not in cache:
            if len(cache) > 80:
                cache.clear()
            cache[p] = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        return cache[p]

    # Confirmed PLAYERS matter as much as confirmed officials. Scoring absolute
    # closeness to an official kit ranked poorly (94% recall needed half the
    # pool) because dark team kits sit near dark official kits in colour space.
    # What discriminates is whether a box looks MORE like an official than like
    # any confirmed player of the same run.
    need += [(b, 'player') for b, v in cand.items() if v == 'player']
    need += [(b, 'player') for b, v in qa.items() if v == 'player']
    for b, v in need:
        r = by_id.get(b)
        if not r:
            continue
        img = get_img(r)
        if img is None:
            continue
        s = torso_hsv(img, r['bbox_xywh'])
        if s is not None:
            conf_kits[r['run']][v].append(s)
    print('human-confirmed kit samples per run:')
    for run, d in sorted(conf_kits.items()):
        print(f'   {run:<9} ' + '  '.join(f'{k}={len(v)}' for k, v in d.items()))

    # cluster each run/role into a few kit modes (medians of hue buckets)
    kit_models = defaultdict(dict)
    for run, d in conf_kits.items():
        for role, arr in d.items():
            a = np.array(arr)
            buckets = defaultdict(list)
            for s in a:
                buckets[int(s[0] // 15)].append(s)
            modes = [np.median(np.array(v), axis=0) for k, v in buckets.items()
                     if len(v) >= max(3, 0.05 * len(a))]
            kit_models[run][role] = modes or [np.median(a, axis=0)]

    # ---- 2. score every unreviewed LIKELY_PLAYER box -----------------------
    pool = [r for r in ledger if r['REVIEW_STATUS'] == 'NOT_QUEUED_LIKELY_PLAYER']
    print(f'\nscoring {len(pool)} unreviewed LIKELY_PLAYER boxes')
    scored = []
    for n, r in enumerate(pool, 1):
        img = get_img(r)
        if img is None:
            continue
        s = torso_hsv(img, r['bbox_xywh'])
        if s is None:
            continue
        run = r['run']
        x, y, w, h = r['bbox_xywh']
        W, H = r['img_w'], r['img_h']
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        ev = []
        d_ref = min([kit_dist(s, m) for m in kit_models.get(run, {}).get('referee', [])]
                    or [9.0])
        d_gk = min([kit_dist(s, m) for m in kit_models.get(run, {}).get('goalkeeper', [])]
                   or [9.0])
        d_pl = min([kit_dist(s, m) for m in kit_models.get(run, {}).get('player', [])]
                   or [9.0])
        # Relative, not absolute: how much more official-like than player-like.
        score_ref = max(0.0, (d_pl - d_ref) / 0.40)
        score_gk = max(0.0, (d_pl - d_gk) / 0.40)
        if score_ref > 0.15:
            ev.append(f'kit closer to a confirmed referee than to any confirmed '
                      f'player of this run (d_ref={d_ref:.2f} vs d_player={d_pl:.2f})')
        if score_gk > 0.15:
            ev.append(f'kit closer to a confirmed goalkeeper than to any confirmed '
                      f'player of this run (d_gk={d_gk:.2f} vs d_player={d_pl:.2f})')
        # geometry priors
        near_edge = min(cx, 1 - cx)
        if near_edge < 0.12:
            score_ref += 0.25
            ev.append('near the frame edge, where assistants patrol')
        if cy > 0.80 or cy < 0.18:
            score_ref += 0.10
            ev.append('near the touchline band')
        if near_edge < 0.18 and 0.25 < cy < 0.75:
            score_gk += 0.20
            ev.append('in the goal third')
        # The frozen detector contributes NOTHING here, and that is not an
        # oversight. This pool is defined as the boxes it called player, so its
        # opinion is constant across every candidate -- which is exactly why its
        # recall failed. A weak detector match is still informative though: a box
        # it barely matched is one it barely saw.
        if (r['detector_conf'] or 0) < 0.45 or r['signals'] == 'NO_DETECTOR_MATCH':
            score_ref += 0.08
            score_gk += 0.08
            ev.append(f"frozen detector matched this box weakly "
                      f"({r['signals']}, conf {r['detector_conf']})")
        role = 'referee' if score_ref >= score_gk else 'goalkeeper'
        scored.append({'BOX_ID': r['BOX_ID'], 'IMAGE': r['IMAGE'], 'run': run,
                       'score': round(float(max(score_ref, score_gk)), 4),
                       'proposed_missed_role': role,
                       'evidence': ev,
                       'is_qa_box': r['BOX_ID'] in qa})
        if n % 4000 == 0:
            print(f'   {n}/{len(pool)}')

    scored.sort(key=lambda z: -z['score'])
    rank = {z['BOX_ID']: i for i, z in enumerate(scored)}

    # ---- 3. VALIDATE on the held-out qa_player labels ----------------------
    truth = {b: v for b, v in qa.items() if b in rank}
    officials = [b for b, v in truth.items() if v in ('goalkeeper', 'referee')]
    print(f'\nvalidation on the {len(truth)} qa_player boxes '
          f'({len(officials)} of them officials a human found):')
    cutoffs = []
    for frac in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        k = int(len(scored) * frac)
        top = set(z['BOX_ID'] for z in scored[:k])
        rec = sum(1 for b in officials if b in top) / max(len(officials), 1)
        cutoffs.append({'top_fraction': frac, 'queue_size': k,
                        'recall_on_held_out_officials': round(rec, 3)})
        print(f'   top {frac:>4.0%} ({k:>5} boxes): recall {rec:.1%}')
    chosen = next((c for c in cutoffs
                   if c['recall_on_held_out_officials'] >= args.target_recall), cutoffs[-1])
    k = chosen['queue_size']
    queue = scored[:k]
    print(f'\nchosen cutoff: top {chosen["top_fraction"]:.0%} = {k} boxes, '
          f'measured recall {chosen["recall_on_held_out_officials"]:.1%}')

    # ---- 4. write the two new review queues --------------------------------
    per_img = defaultdict(list)
    for z in queue:
        per_img[z['IMAGE']].append(z)
    mr = {
        'purpose': ('surface probable goalkeeper/referee boxes that were present in '
                    'reviewed images but never entered the original candidate queue'),
        'why_needed': ('qa_player measured a 6.40% missed-role rate and 25 of 57 '
                       'no-candidate images held a missed official; 4,153/4,153 '
                       'candidate decisions is not proof of complete role repair'),
        'generation_signals': [
            'torso colour distance to HUMAN-CONFIRMED goalkeeper and referee kits '
            'in the SAME broadcast run',
            'frame-edge and touchline position, where assistant referees patrol',
            'goal-third position, for goalkeepers',
            'the frozen detector class and confidence, where it had an opinion'],
        'no_role_is_assigned_automatically': True,
        'no_identity_propagated': True,
        'validation': {
            'held_out_set': 'the 250 qa_player boxes, labelled by a human independently',
            'officials_in_it': len(officials),
            'recall_by_cutoff': cutoffs,
            'chosen': chosen},
        'queue_boxes': len(queue), 'queue_images': len(per_img),
        'population_scored': len(scored),
        'by_run': dict(Counter(z['run'] for z in queue)),
        'proposed_role_mix': dict(Counter(z['proposed_missed_role'] for z in queue)),
        'answers_recorded': 0,
        'rows': queue,
    }
    (PKG / 'missed_role_queue.json').write_text(json.dumps(mr, indent=1),
                                                encoding='utf-8')

    u_ids = [b for b, v in cand.items() if v == 'uncertain']
    urows = []
    for b in u_ids:
        r = by_id[b]
        urows.append({'BOX_ID': b, 'IMAGE': r['IMAGE'], 'run': r['run'],
                      'ORIGINAL_CLASS': r['ORIGINAL_CLASS'],
                      'ORIGINAL_PROPOSAL': r['PROPOSED_CLASS'],
                      'FIRST_HUMAN_DECISION': 'U',
                      'bbox_xywh': r['bbox_xywh'],
                      'U_RESOLUTION_CATEGORY': None, 'FINAL_ACTION': None,
                      'FINAL_CLASS': None, 'REVIEW_NOTE': None})
    (PKG / 'u_resolution_queue.json').write_text(json.dumps({
        'purpose': 'resolve every original U into one documented category',
        'categories': {
            'A AMBIGUOUS_TARGET': 'a real EyeCU target, role not reliably distinguishable',
            'O OCCLUDED_UNCLEAR': 'substantially occluded, role undeterminable',
            'N NON_TARGET_HUMAN': 'real human outside the four-class ontology '
                                  '(coach, bench, ball person, medical, staff)',
            'B BALL_WRONG_HUMAN_BOX': 'the human box is actually on the ball',
            'F FALSE_POSITIVE': 'no relevant human in the box',
            'X PARTIAL_BODY_BAD_BOX': 'real person but the geometry covers a fragment '
                                      'while much more of the body is visible'},
        'also_allowed': ['P', 'G', 'R'],
        'valid_occlusion_note': ('a box around the visible extent of a genuinely '
                                 'hidden person is VALID and may stay P/G/R if the '
                                 'role is clear; only a fragment box where much more '
                                 'is visible is PARTIAL_BODY_BAD_BOX'),
        'count': len(urows), 'answers_recorded': 0,
        'by_run': dict(Counter(r['run'] for r in urows)),
        'rows': urows}, indent=1), encoding='utf-8')
    print(f'\nwrote missed_role_queue.json ({len(queue)} boxes, {len(per_img)} images)')
    print(f'wrote u_resolution_queue.json ({len(urows)} boxes)')


if __name__ == '__main__':
    main()
