#!/usr/bin/env python
"""
Measure what the human review ACTUALLY covers. No assumptions from the prompt.

decisions.json is an append-only log, so a box can appear more than once when the
reviewer changed their mind. Last write wins, and the number of lines is
therefore NOT the number of decisions -- reporting 5,095 as "decisions" would
overcount corrections as work.

Completion is measured per mode against the population each mode was built from,
and unanswered items are named rather than summarised away.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'

QA_MAP = {'player': 'TRUE_PLAYER', 'goalkeeper': 'MISSED_GOALKEEPER',
          'referee': 'MISSED_REFEREE', 'uncertain': 'UNCERTAIN'}
NC_MAP = {'player': 'NO_OFFICIAL_PRESENT', 'goalkeeper': 'MISSED_GOALKEEPER_PRESENT',
          'referee': 'MISSED_REFEREE_PRESENT', 'uncertain': 'UNCERTAIN'}


def load_log():
    p = PKG / 'decisions.json'
    lines, last = 0, {}
    revisions = Counter()
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        lines += 1
        k = (d.get('mode', 'candidates'), d['BOX_ID'])
        if k in last and last[k]['HUMAN_FINAL_CLASS'] != d['HUMAN_FINAL_CLASS']:
            revisions[k] += 1
        last[k] = d
    return lines, last, revisions


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ledger = json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))
    by_id = {r['BOX_ID']: r for r in ledger}
    lines, last, revisions = load_log()

    cand_ids = {r['BOX_ID'] for r in ledger if r['REVIEW_STATUS'] == 'PENDING'}
    qp = json.loads((PKG / 'qa_likely_player.json').read_text(encoding='utf-8'))
    qn = json.loads((PKG / 'qa_no_candidate_images.json').read_text(encoding='utf-8'))
    qp_ids = {r['BOX_ID'] for r in qp['rows']}
    qn_imgs = {r['IMAGE'] for r in qn['rows']}
    qn_ids = {r['BOX_ID'] for r in ledger
              if r['IMAGE'] in qn_imgs and r['eyecu_original_class'] == 'player'}

    got = defaultdict(dict)
    for (mode, bid), d in last.items():
        got[mode][bid] = d['HUMAN_FINAL_CLASS']

    print(f'log lines {lines}, unique (mode,box) decisions {len(last)}, '
          f'boxes the reviewer changed their mind on {len(revisions)}')

    rep = {'log_lines': lines, 'unique_decisions': len(last),
           'boxes_revised_by_reviewer': len(revisions), 'modes': {}}

    # ---- candidates ---------------------------------------------------------
    c = got.get('candidates', {})
    answered = {b: v for b, v in c.items() if b in cand_ids}
    stray = {b: v for b, v in c.items() if b not in cand_ids}
    counts = Counter(answered.values())
    missing = sorted(cand_ids - set(answered))
    rep['modes']['candidates'] = {
        'population': len(cand_ids), 'answered': len(answered),
        'complete': len(missing) == 0,
        'unanswered': len(missing), 'unanswered_examples': missing[:10],
        'P_player': counts.get('player', 0),
        'G_goalkeeper': counts.get('goalkeeper', 0),
        'R_referee': counts.get('referee', 0),
        'U_unresolved': counts.get('uncertain', 0),
        'decisions_outside_the_candidate_queue': len(stray),
    }
    print(f"\ncandidates : {len(answered)}/{len(cand_ids)} "
          f"{'COMPLETE' if not missing else f'INCOMPLETE ({len(missing)} left)'}")
    print(f"   P {counts.get('player',0)}  G {counts.get('goalkeeper',0)}  "
          f"R {counts.get('referee',0)}  U {counts.get('uncertain',0)}")
    if stray:
        print(f'   note: {len(stray)} decisions on boxes outside the queue')

    # ---- qa_player ----------------------------------------------------------
    q = got.get('qa_player', {})
    ans = {b: QA_MAP.get(v, v) for b, v in q.items() if b in qp_ids}
    qc = Counter(ans.values())
    miss = qc.get('MISSED_GOALKEEPER', 0) + qc.get('MISSED_REFEREE', 0)
    rate = miss / len(ans) if ans else None
    rep['modes']['qa_player'] = {
        'population': len(qp_ids), 'answered': len(ans),
        'complete': len(ans) >= len(qp_ids),
        'TRUE_PLAYER': qc.get('TRUE_PLAYER', 0),
        'MISSED_GOALKEEPER': qc.get('MISSED_GOALKEEPER', 0),
        'MISSED_REFEREE': qc.get('MISSED_REFEREE', 0),
        'UNCERTAIN': qc.get('UNCERTAIN', 0),
        'missed_role_rate': rate,
        'note': ('reported as separate GK and referee counts, never one combined '
                 'error number'),
    }
    print(f"\nqa_player  : {len(ans)}/{len(qp_ids)}  {dict(qc)}")
    if rate is not None:
        print(f'   measured missed-role rate {rate:.2%}')

    # ---- qa_nocand ----------------------------------------------------------
    n = got.get('qa_nocand', {})
    nans = {b: NC_MAP.get(v, v) for b, v in n.items() if b in qn_ids}
    img_ans = defaultdict(set)
    for b, v in nans.items():
        img_ans[by_id[b]['IMAGE']].add(v)
    gk_imgs = [i for i, v in img_ans.items() if 'MISSED_GOALKEEPER_PRESENT' in v]
    rf_imgs = [i for i, v in img_ans.items() if 'MISSED_REFEREE_PRESENT' in v]
    unc_imgs = [i for i, v in img_ans.items() if 'UNCERTAIN' in v]
    rep['modes']['qa_nocand'] = {
        'population_images': len(qn_imgs), 'answered_images': len(img_ans),
        'complete': len(img_ans) >= len(qn_imgs),
        'boxes_answered': len(nans),
        'images_with_missed_goalkeeper': len(gk_imgs),
        'images_with_missed_referee': len(rf_imgs),
        'images_with_uncertain': len(unc_imgs),
        'clean_images': len([i for i, v in img_ans.items()
                             if v <= {'NO_OFFICIAL_PRESENT'}]),
        'limitation': ('this mode only covered images with ZERO original '
                       'candidates. It cannot detect the failure where one '
                       'candidate exists in an image while a second goalkeeper '
                       'or referee remains an unqueued context box.'),
    }
    print(f"\nqa_nocand  : {len(img_ans)}/{len(qn_imgs)} images, {len(nans)} boxes")
    print(f'   missed GK images {len(gk_imgs)}  missed referee images {len(rf_imgs)}'
          f'  uncertain {len(unc_imgs)}')

    # ---- U by run, for the second pass --------------------------------------
    u_ids = [b for b, v in answered.items() if v == 'uncertain']
    by_run = Counter(by_id[b]['run'] for b in u_ids)
    rep['U_boxes'] = {'count': len(u_ids), 'by_run': dict(by_run),
                      'by_original_proposal': dict(Counter(
                          by_id[b]['PROPOSED_CLASS'] or 'ambiguous' for b in u_ids)),
                      'images': len({by_id[b]['IMAGE'] for b in u_ids})}
    print(f'\nU boxes: {len(u_ids)} in {rep["U_boxes"]["images"]} images, '
          f'by run {dict(by_run)}')

    (PKG / 'REVIEW_COMPLETION.json').write_text(json.dumps(rep, indent=1),
                                                encoding='utf-8')
    print('\nwrote REVIEW_COMPLETION.json')


if __name__ == '__main__':
    main()
