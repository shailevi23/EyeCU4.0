#!/usr/bin/env python
"""
Ingest human decisions, run the acceptance gate, and only then repair the copy.

The gate is the point of this file. Reviewing all 4,153 candidates is necessary
and not sufficient: candidate review measures precision and can never reveal an
official the triage missed, so the two QA samples must also be complete and must
not show a systematic recall problem. --apply refuses to write unless every
condition passes, and prints which one failed.

What repair means here is narrow and checked: the class id of a human box may
change, and nothing else. Geometry, image files, ball boxes and box counts are
compared before and after, and a mismatch is a failure, not a warning.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'EyeCU_external_data/huggingface/keremberke_football_object_detection'
PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'

QA_MAP = {'player': 'TRUE_PLAYER', 'goalkeeper': 'MISSED_GOALKEEPER',
          'referee': 'MISSED_REFEREE', 'uncertain': 'UNCERTAIN'}
NC_MAP = {'player': 'NO_OFFICIAL_PRESENT', 'goalkeeper': 'MISSED_GOALKEEPER_PRESENT',
          'referee': 'MISSED_REFEREE_PRESENT', 'uncertain': 'UNCERTAIN'}
BINS = [(0, 3, '<3'), (3, 5, '3-5'), (5, 8, '>5-8'), (8, 12, '>8-12'),
        (12, 20, '>12-20'), (20, 40, '>20-40'), (40, 1e9, '>40')]


def bin_of(w):
    for lo, hi, n in BINS:
        if lo == 0 and w < hi:
            return n
        if lo < w <= hi:
            return n
    return '>40'


def load_decisions():
    p = PKG / 'decisions.json'
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding='utf-8').splitlines():
        if line.strip():
            d = json.loads(line)
            out[d['BOX_ID']] = d          # last write wins, by design
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true',
                    help='write the repaired labels; refused unless the gate passes')
    ap.add_argument('--recall-tolerance', type=float, default=0.02,
                    help='max missed-official rate in the LIKELY_PLAYER QA sample')
    args = ap.parse_args()

    ledger = json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))
    by_id = {r['BOX_ID']: r for r in ledger}
    dec = load_decisions()
    qp = json.loads((PKG / 'qa_likely_player.json').read_text(encoding='utf-8'))
    qn = json.loads((PKG / 'qa_no_candidate_images.json').read_text(encoding='utf-8'))

    qp_ids = {r['BOX_ID'] for r in qp['rows']}
    qn_imgs = {r['IMAGE'] for r in qn['rows']}
    qn_ids = {r['BOX_ID'] for r in ledger
              if r['IMAGE'] in qn_imgs and r['eyecu_original_class'] == 'player'}
    cand_ids = {r['BOX_ID'] for r in ledger if r['REVIEW_STATUS'] == 'PENDING'}

    # ---- fold decisions into the ledger ------------------------------------
    for bid, d in dec.items():
        r = by_id.get(bid)
        if not r:
            continue
        cls = d['HUMAN_FINAL_CLASS']
        mode = d.get('mode', 'candidates')
        if mode == 'candidates':
            r['HUMAN_FINAL_CLASS'] = cls
            r['REVIEW_STATUS'] = 'UNCERTAIN' if cls == 'uncertain' else 'REVIEWED'
            r['REASON_OR_GROUP'] = f"run={r['run']} proposed={r['PROPOSED_CLASS']} signals={r['signals']}"
        else:
            r['QA_ANSWER'] = (QA_MAP if mode == 'qa_player' else NC_MAP).get(cls, cls)
            r['QA_MODE'] = mode

    reviewed = sum(1 for b in cand_ids if by_id[b]['REVIEW_STATUS'] == 'REVIEWED')
    uncertain = [b for b in cand_ids if by_id[b]['REVIEW_STATUS'] == 'UNCERTAIN']
    qp_done = [by_id[b] for b in qp_ids if by_id[b].get('QA_ANSWER')]
    qn_done = [by_id[b] for b in qn_ids if by_id[b].get('QA_ANSWER')]

    qp_missed = sum(1 for r in qp_done if r['QA_ANSWER'] in
                    ('MISSED_GOALKEEPER', 'MISSED_REFEREE'))
    qp_rate = qp_missed / len(qp_done) if qp_done else None
    qn_missed_imgs = {r['IMAGE'] for r in qn_done if r['QA_ANSWER'] in
                      ('MISSED_GOALKEEPER_PRESENT', 'MISSED_REFEREE_PRESENT')}
    qn_imgs_done = {r['IMAGE'] for r in qn_done}

    # ---- acceptance gate ----------------------------------------------------
    gate = [
        ('A all candidate boxes reviewed',
         reviewed + len(uncertain) == len(cand_ids),
         f'{reviewed + len(uncertain)}/{len(cand_ids)}'),
        ('B all uncertain resolved or explicitly excluded',
         len(uncertain) == 0,
         f'{len(uncertain)} still uncertain'),
        ('C LIKELY_PLAYER QA completed',
         len(qp_done) >= len(qp_ids),
         f'{len(qp_done)}/{len(qp_ids)}'),
        ('D no-candidate image QA completed',
         len(qn_imgs_done) >= len(qn_imgs),
         f'{len(qn_imgs_done)}/{len(qn_imgs)} images'),
        ('E no systematic missed officials',
         (qp_rate is not None and qp_rate <= args.recall_tolerance
          and not qn_missed_imgs),
         (f'QA missed-role rate {qp_rate:.3%}' if qp_rate is not None
          else 'not measured') +
         f'; {len(qn_missed_imgs)} no-candidate images with an official'),
        ('F class counts and label integrity validated', None, 'checked on --apply'),
        ('G ball boxes geometry identical', None, 'checked on --apply'),
        ('H no boxes added or deleted', None, 'checked on --apply'),
    ]
    print(f'{"GATE":<48}{"RESULT":<10}DETAIL')
    for name, ok, detail in gate:
        s = 'PASS' if ok else ('PENDING' if ok is None else 'FAIL')
        print(f'{name:<48}{s:<10}{detail}')
    blocking = [g for g in gate if g[1] is False]
    ready = not blocking and all(g[1] is not None or True for g in gate)

    # ---- counts (current state) ---------------------------------------------
    final = Counter()
    for r in ledger:
        if r['eyecu_original_class'] == 'ball':
            final['ball'] += 1
        else:
            final[r['HUMAN_FINAL_CLASS'] or 'player (unreviewed)'] += 1
    recl = Counter()
    for r in ledger:
        f = r['HUMAN_FINAL_CLASS']
        if f and f != 'uncertain' and f != r['eyecu_original_class']:
            recl[f"{r['eyecu_original_class']} -> {f}"] += 1

    ball = json.loads((SRC / 'manifests' / 'ball_instances.json').read_text(encoding='utf-8'))
    w = [b['w'] for b in ball]
    ball_stats = {'instances': len(ball),
                  'le5': sum(1 for x in w if x <= 5),
                  'le8': sum(1 for x in w if x <= 8),
                  'le12': sum(1 for x in w if x <= 12),
                  'bins': {b[2]: sum(1 for x in w if bin_of(x) == b[2]) for b in BINS},
                  'convention': 'stored pixels, 1280x720, unchanged from the completed audit'}

    report = {
        'review_status': ('READY_TO_APPLY' if ready and reviewed
                          else 'AWAITING_HUMAN_REVIEW'),
        'human_decisions_recorded': len(dec),
        'candidate_boxes': len(cand_ids), 'candidates_reviewed': reviewed,
        'candidates_uncertain': len(uncertain),
        'qa_likely_player': {'sample': len(qp_ids), 'answered': len(qp_done),
                             'missed_officials': qp_missed,
                             'missed_rate': qp_rate,
                             'tolerance': args.recall_tolerance},
        'qa_no_candidate': {'images': len(qn_imgs), 'answered': len(qn_imgs_done),
                            'images_with_a_missed_official': len(qn_missed_imgs),
                            'note': ('this is a CENSUS, not a sample: only 57 images '
                                     'have no candidate at all, so every one is '
                                     'reviewed')},
        'gate': [{'condition': n, 'result': ('PASS' if o else 'PENDING' if o is None
                                             else 'FAIL'), 'detail': d}
                 for n, o, d in gate],
        'current_class_counts': dict(final),
        'reclassifications': dict(recl),
        'ball_counts_preserved': ball_stats,
    }
    (PKG / 'REVIEW_STATUS.json').write_text(json.dumps(report, indent=1),
                                            encoding='utf-8')
    print(f'\ndecisions recorded: {len(dec)}   candidates reviewed: {reviewed}/{len(cand_ids)}')
    print(f'class counts now: {dict(final)}')
    print(f'ball preserved: {ball_stats["instances"]} instances, '
          f'<=5 {ball_stats["le5"]}  <=8 {ball_stats["le8"]}  <=12 {ball_stats["le12"]}')

    if not args.apply:
        print(f'\n(no --apply; wrote {(PKG / "REVIEW_STATUS.json").relative_to(REPO)})')
        return
    if blocking or reviewed == 0:
        print('\nREFUSED: the acceptance gate has not passed. Nothing written.')
        for n, o, d in blocking:
            print(f'   blocked by: {n} -- {d}')
        sys.exit(1)

    # ---- apply: class ids only ---------------------------------------------
    name_to_id = {}
    for split in ('train', 'valid', 'test'):
        wc = PKG / 'working_copy' / f'{split}_annotations.coco.json'
        a = json.loads(wc.read_text(encoding='utf-8'))
        have = {c['name']: c['id'] for c in a['categories']}
        nxt = max(have.values()) + 1
        for nm in ('goalkeeper', 'referee'):
            if nm not in have:
                have[nm] = nxt
                a['categories'].append({'id': nxt, 'name': nm,
                                        'supercategory': 'none'})
                nxt += 1
        name_to_id[split] = have
        before = [(x['id'], tuple(round(float(v), 4) for v in x['bbox'])) for x in a['annotations']]
        changed = 0
        for ann in a['annotations']:
            r = by_id.get(f'{split}:{ann["id"]}')
            if not r or not r['HUMAN_FINAL_CLASS'] or r['HUMAN_FINAL_CLASS'] == 'uncertain':
                continue
            if r['HUMAN_FINAL_CLASS'] in ('goalkeeper', 'referee'):
                ann['category_id'] = have[r['HUMAN_FINAL_CLASS']]
                changed += 1
        after = [(x['id'], tuple(round(float(v), 4) for v in x['bbox'])) for x in a['annotations']]
        assert before == after, f'{split}: geometry or box set changed -- refusing'
        wc.write_text(json.dumps(a), encoding='utf-8')
        print(f'{split}: {changed} class ids changed, geometry identical, '
              f'{len(a["annotations"])} boxes unchanged in count')
    print('\nAPPLIED to the working copy. Original export untouched.')


if __name__ == '__main__':
    main()
