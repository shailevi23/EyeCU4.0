#!/usr/bin/env python
"""
Build the Keremberke human review package for Option A.

Nothing here decides a class. It creates a working COPY of the annotations, a
ledger with one row per human box, an ordered review queue, two independent QA
samples, and per-run kit reference sheets. Every HUMAN_FINAL_CLASS starts empty
and only a person may fill it.

Three design choices worth stating:

  * The ledger carries ORIGINAL_CLASS and PROPOSED_CLASS side by side and keeps
    them forever. A proposal that silently becomes a label is the failure mode
    this whole package exists to prevent, so the two can never be confused.

  * Review is ordered by broadcast run, then by proposal type, then by signal
    agreement. A reviewer settles "in this run the teal shirt is the referee"
    once and then moves fast. Clustering is presentation only -- no decision is
    ever copied from one box to another.

  * QA is TWO samples, not one. Candidate precision and triage recall fail
    differently: reviewing candidates can never reveal an official the triage
    missed entirely, so a separate sample of no-candidate images is drawn.
"""

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'EyeCU_external_data/huggingface/keremberke_football_object_detection'
PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
XS = REPO / 'experiments' / 'external_sources'

EYECU = ['player', 'goalkeeper', 'referee', 'ball']


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--qa-player', type=int, default=250)
    ap.add_argument('--qa-nocand', type=int, default=120)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    import cv2

    PKG.mkdir(parents=True, exist_ok=True)
    (PKG / 'working_copy').mkdir(exist_ok=True)
    (PKG / 'reference_sheets').mkdir(exist_ok=True)

    # ---- 1. immutable original, working copy -------------------------------
    origs = {}
    for split in ('train', 'valid', 'test'):
        aj = list((SRC / 'extracted' / split).rglob('_annotations.coco.json'))
        if not aj:
            continue
        origs[split] = aj[0]
        dst = PKG / 'working_copy' / f'{split}_annotations.coco.json'
        if not dst.exists():
            shutil.copy2(aj[0], dst)
    src_hashes = {s: sha256(p) for s, p in origs.items()}
    print(f'working copy: {len(origs)} splits')

    # ---- 2. ledger ----------------------------------------------------------
    triage = json.loads((XS / 'reports' / 'keremberke_role_triage_boxes.json')
                        .read_text(encoding='utf-8'))
    tri_by = {(r['split'], r['file'], tuple(r['bbox'])): r for r in triage}

    ledger, img_index = [], {}
    for split, ap_ in origs.items():
        a = json.loads(ap_.read_text(encoding='utf-8'))
        cats = {c['id']: c['name'] for c in a['categories']}
        imgs = {i['id']: i for i in a['images']}
        for ann in a['annotations']:
            im = imgs[ann['image_id']]
            cname = cats.get(ann['category_id'], str(ann['category_id']))
            key = (split, im['file_name'],
                   tuple(round(float(v), 1) for v in ann['bbox']))
            t = tri_by.get(key)
            is_human = cname == 'player'
            ledger.append({
                'BOX_ID': f"{split}:{ann['id']}",
                'IMAGE': f"{split}/{im['file_name']}",
                'split': split, 'file': im['file_name'],
                'img_w': im['width'], 'img_h': im['height'],
                'bbox_xywh': [round(float(v), 2) for v in ann['bbox']],
                'ORIGINAL_CLASS': cname,
                'eyecu_original_class': 'player' if is_human else 'ball',
                'PROPOSED_CLASS': (
                    {'POSSIBLE_GOALKEEPER': 'goalkeeper',
                     'POSSIBLE_REFEREE': 'referee',
                     'LIKELY_PLAYER': 'player',
                     'AMBIGUOUS': None}.get(t['triage']) if t else
                    ('ball' if not is_human else None)),
                'triage': (t['triage'] if t else ('BALL' if not is_human else 'NO_TRIAGE')),
                'signals': (t['agreement'] if t else 'n/a'),
                'detector_conf': (t['detector_conf'] if t else None),
                'detector_iou': (t['detector_iou'] if t else None),
                'run': (t['run'] if t else None),
                'HUMAN_FINAL_CLASS': None,
                'REVIEW_STATUS': ('NOT_REQUIRED_BALL' if not is_human else
                                  'PENDING' if (t and t['triage'] != 'LIKELY_PLAYER')
                                  else 'NOT_QUEUED_LIKELY_PLAYER'),
                'REASON_OR_GROUP': None,
            })
            img_index.setdefault(f"{split}/{im['file_name']}", []).append(ledger[-1]['BOX_ID'])

    (PKG / 'ledger.json').write_text(json.dumps(ledger, indent=0), encoding='utf-8')
    st = Counter(r['REVIEW_STATUS'] for r in ledger)
    print(f'ledger: {len(ledger)} rows  {dict(st)}')

    # ---- 3. review queue, ordered for human speed --------------------------
    ORDER = {'POSSIBLE_REFEREE': 0, 'POSSIBLE_GOALKEEPER': 1, 'AMBIGUOUS': 2}
    AGREE = {'BOTH_SIGNALS': 0, 'DETECTOR_ONLY': 1, 'KIT_ONLY': 2,
             'NO_DETECTOR_MATCH': 3}
    RUNS = ['plain_A', 'plain_B', 'pp_A', 'pp_B']
    pend = [r for r in ledger if r['REVIEW_STATUS'] == 'PENDING']
    # group into per-image review units so a reviewer opens each image once
    units = defaultdict(list)
    for r in pend:
        units[r['IMAGE']].append(r)
    queue = []
    for img, rows in units.items():
        run = rows[0]['run'] or 'other'
        prio = min(ORDER.get(x['triage'], 9) for x in rows)
        agr = min(AGREE.get(x['signals'], 9) for x in rows)
        # disagreement cases last within their band, as instructed
        disagree = any(x['signals'] in ('KIT_ONLY', 'DETECTOR_ONLY') for x in rows)
        queue.append({
            'IMAGE': img, 'run': run,
            'sort': (RUNS.index(run) if run in RUNS else 9, prio, agr, int(disagree), img),
            'candidate_box_ids': [x['BOX_ID'] for x in rows],
            'all_box_ids': img_index[img],
            'candidate_summary': dict(Counter(x['triage'] for x in rows)),
        })
    queue.sort(key=lambda q: q['sort'])
    for i, q in enumerate(queue):
        q['position'] = i
        del q['sort']
    (PKG / 'review_queue.json').write_text(json.dumps(queue, indent=0), encoding='utf-8')
    print(f'review queue: {len(queue)} images, {len(pend)} candidate boxes')
    print(f'   by run: {dict(Counter(q["run"] for q in queue))}')

    # ---- 4. stratified LIKELY_PLAYER QA sample ------------------------------
    lp = [r for r in ledger if r['REVIEW_STATUS'] == 'NOT_QUEUED_LIKELY_PLAYER']
    for r in lp:
        x, y, w, h = r['bbox_xywh']
        r['_size'] = ('small' if h < 40 else 'medium' if h < 80 else 'large')
        r['_conf'] = ('lo' if (r['detector_conf'] or 0) < 0.5 else
                      'mid' if (r['detector_conf'] or 0) < 0.75 else 'hi')
        cx = (x + w / 2) / max(r['img_w'], 1)
        r['_region'] = ('left' if cx < 0.33 else 'centre' if cx < 0.66 else 'right')
        cy = (y + h / 2) / max(r['img_h'], 1)
        r['_depth'] = 'far' if cy < 0.45 else 'near'
    strata = defaultdict(list)
    for r in lp:
        strata[(r['run'], r['_size'], r['_conf'], r['_region'], r['_depth'])].append(r)
    keys = sorted(strata, key=lambda k: tuple(str(x) for x in k))
    qa, i = [], 0
    # proportional-ish round robin so every stratum that exists can appear
    while len(qa) < min(args.qa_player, len(lp)):
        k = keys[i % len(keys)]
        pool = strata[k]
        if pool:
            qa.append(pool.pop(rng.randrange(len(pool))))
        i += 1
        if i > len(keys) * 200:
            break
    qa_rows = [{'BOX_ID': r['BOX_ID'], 'IMAGE': r['IMAGE'],
                'bbox_xywh': r['bbox_xywh'], 'run': r['run'],
                'stratum': {'size': r['_size'], 'detector_conf_band': r['_conf'],
                            'region': r['_region'], 'depth': r['_depth']},
                'detector_conf': r['detector_conf'],
                'HUMAN_ANSWER': None,
                'allowed': ['TRUE_PLAYER', 'MISSED_GOALKEEPER',
                            'MISSED_REFEREE', 'UNCERTAIN']} for r in qa]
    (PKG / 'qa_likely_player.json').write_text(
        json.dumps({'purpose': 'measure triage RECALL -- officials wrongly left as player',
                    'sampling': 'stratified round-robin over run x size x detector-confidence x region x depth',
                    'seed': args.seed, 'population': len(lp),
                    'sample_size': len(qa_rows),
                    'strata_available': len(keys),
                    'strata_covered': len({tuple(r['stratum'].values()) for r in qa_rows}),
                    'answers_recorded': 0, 'rows': qa_rows}, indent=1),
        encoding='utf-8')
    print(f'LIKELY_PLAYER QA: {len(qa_rows)} boxes over {len(keys)} strata')

    # ---- 5. no-candidate image QA ------------------------------------------
    cand_imgs = {q['IMAGE'] for q in queue}
    all_imgs = defaultdict(list)
    for r in ledger:
        if r['eyecu_original_class'] == 'player':
            all_imgs[r['IMAGE']].append(r)
    nocand = [i for i in all_imgs if i not in cand_imgs]
    by_run = defaultdict(list)
    for i in nocand:
        by_run[all_imgs[i][0]['run'] or 'other'].append(i)
    nc = []
    runs = sorted(by_run)
    j = 0
    while len(nc) < min(args.qa_nocand, len(nocand)) and runs:
        r = runs[j % len(runs)]
        if by_run[r]:
            nc.append(by_run[r].pop(rng.randrange(len(by_run[r]))))
        j += 1
        if j > len(nocand) * 4:
            break
    (PKG / 'qa_no_candidate_images.json').write_text(
        json.dumps({'purpose': ('detect the failure mode candidate review cannot see: '
                                'an official the triage missed entirely, in an image '
                                'where nothing was flagged'),
                    'kept_separate_from_candidate_precision': True,
                    'seed': args.seed, 'population': len(nocand),
                    'sample_size': len(nc),
                    'by_run': dict(Counter(all_imgs[i][0]['run'] or 'other' for i in nc)),
                    'answers_recorded': 0,
                    'rows': [{'IMAGE': i, 'human_boxes': len(all_imgs[i]),
                              'run': all_imgs[i][0]['run'],
                              'HUMAN_ANSWER': None,
                              'allowed': ['NO_OFFICIAL_PRESENT',
                                          'MISSED_GOALKEEPER_PRESENT',
                                          'MISSED_REFEREE_PRESENT', 'UNCERTAIN']}
                             for i in nc]}, indent=1), encoding='utf-8')
    print(f'no-candidate QA: {len(nc)} images from {len(nocand)} available')

    # ---- 6. per-run kit reference sheets -----------------------------------
    paths = {}
    for split in origs:
        base = origs[split].parent
        for p in base.glob('*.jpg'):
            paths[(split, p.name)] = p
    by_run_boxes = defaultdict(lambda: defaultdict(list))
    for r in ledger:
        if r['eyecu_original_class'] != 'player' or not r['run']:
            continue
        bucket = {'POSSIBLE_REFEREE': 'referee candidates',
                  'POSSIBLE_GOALKEEPER': 'goalkeeper candidates',
                  'AMBIGUOUS': 'ambiguous',
                  'LIKELY_PLAYER': 'team players'}.get(r['triage'])
        if bucket:
            by_run_boxes[r['run']][bucket].append(r)
    for run, buckets in by_run_boxes.items():
        tiles, labels = [], []
        for bname in ('referee candidates', 'goalkeeper candidates',
                      'team players', 'ambiguous'):
            rows = buckets.get(bname, [])
            if not rows:
                continue
            sel = [rows[i] for i in np.linspace(0, len(rows) - 1,
                                                min(8, len(rows))).round().astype(int)]
            for r in dict.fromkeys(x['BOX_ID'] for x in sel):
                rr = next(x for x in sel if x['BOX_ID'] == r)
                p = paths.get((rr['split'], rr['file']))
                if p is None:
                    continue
                img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                x, y, w, h = rr['bbox_xywh']
                H, W = img.shape[:2]
                m = max(4, 0.15 * h)
                c = img[int(max(0, y - m)):int(min(H, y + h + m)),
                        int(max(0, x - m)):int(min(W, x + w + m))].copy()
                if c.size == 0:
                    continue
                s = 150 / max(c.shape[:2])
                c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
                tiles.append(c)
                labels.append({'referee candidates': 'REF?',
                               'goalkeeper candidates': 'GK?',
                               'team players': 'team',
                               'ambiguous': 'ambig'}[bname])
        if not tiles:
            continue
        hh = max(t.shape[0] for t in tiles); ww = max(t.shape[1] for t in tiles)
        cols = 8
        g = np.full(((len(tiles) + cols - 1) // cols * (hh + 24) + 6,
                     cols * (ww + 6) + 6, 3), 32, np.uint8)
        for i, (t, lb) in enumerate(zip(tiles, labels)):
            yy = 6 + (i // cols) * (hh + 24); xx = 6 + (i % cols) * (ww + 6)
            g[yy + 18:yy + 18 + t.shape[0], xx:xx + t.shape[1]] = t
            cv2.putText(g, lb[:22], (xx + 2, yy + 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imencode('.jpg', g, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(
            str(PKG / 'reference_sheets' / f'{run}_kits.jpg'))
        print(f'   reference sheet {run}_kits.jpg ({len(tiles)} crops)')

    # ---- 7. package manifest -----------------------------------------------
    man = {
        'package': 'Keremberke human review, Option A (reclassify existing boxes)',
        'created_utc': __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ',
                                                   __import__('time').gmtime()),
        'original_source_immutable': True,
        'original_annotation_sha256': src_hashes,
        'working_copy': 'working_copy/<split>_annotations.coco.json',
        'geometry_may_change': False,
        'only_class_ids_may_change': True,
        'ledger_rows': len(ledger),
        'human_boxes': sum(1 for r in ledger if r['eyecu_original_class'] == 'player'),
        'ball_boxes': sum(1 for r in ledger if r['eyecu_original_class'] == 'ball'),
        'queued_candidate_boxes': len(pend),
        'queued_images': len(queue),
        'qa_likely_player_sample': len(qa_rows),
        'qa_no_candidate_sample': len(nc),
        'human_decisions_recorded': 0,
        'review_status': 'PACKAGE_BUILT_AWAITING_HUMAN',
        'no_proposal_is_ground_truth': True,
    }
    (PKG / 'PACKAGE_MANIFEST.json').write_text(json.dumps(man, indent=1),
                                               encoding='utf-8')
    print(f'\nwrote {PKG.relative_to(REPO)}/PACKAGE_MANIFEST.json')
    print('review_status: PACKAGE_BUILT_AWAITING_HUMAN')


if __name__ == '__main__':
    main()
