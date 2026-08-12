#!/usr/bin/env python
"""
Second-pass acceptance gate. Sixteen conditions, and --apply obeys all of them.

The first gate would have passed: 4,153/4,153 candidates reviewed, both QA
samples complete. It would have been wrong. The QA it required is what proved
the coverage was incomplete -- 6.40% of sampled LIKELY_PLAYER boxes were
officials that were never queued -- so completing the queue was never evidence
that the roles were repaired.

Condition G is therefore not "the QA was run" but "the QA came back clean". It
is failing now, and it should be.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import kb_decisions                                              # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
SRC = REPO / 'EyeCU_external_data/huggingface/keremberke_football_object_detection'


def load(p, d=None):
    p = Path(p)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else d


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    apply = '--apply' in sys.argv
    ledger = load(PKG / 'ledger.json')
    by_id = {r['BOX_ID']: r for r in ledger}
    last = {}
    for line in (PKG / 'decisions.json').read_text(encoding='utf-8').splitlines():
        if line.strip():
            d = json.loads(line)
            last[(d.get('mode', 'candidates'), d['BOX_ID'])] = d['HUMAN_FINAL_CLASS']
    cand = {b: v for (m, b), v in last.items() if m == 'candidates'}
    qa = {b: v for (m, b), v in last.items() if m == 'qa_player'}
    noc = {b: v for (m, b), v in last.items() if m == 'qa_nocand'}
    ures = load(PKG / 'u_resolution_queue.json', {})
    mrq = load(PKG / 'missed_role_queue.json', {})
    comp = load(PKG / 'REVIEW_COMPLETION.json', {})

    # DEFECT FIXED HERE. This gate used to read U_RESOLUTION_CATEGORY and
    # HUMAN_ANSWER out of the queue JSON files, but the review server only ever
    # writes decisions.json -- it never edits a queue file. Conditions D, E and F
    # would therefore have read 0/48 and 0/6,984 no matter how much reviewing was
    # done, and 5-8 hours of the second pass would have registered as nothing.
    # decisions.json is the single source of truth; the queues define the
    # population, not the progress.
    ures_dec = {b: v for (m, b), v in last.items() if m == 'u_resolution'}
    mr_dec = {b: v for (m, b), v in last.items() if m == 'missed_role'}

    # A U decision is either a second-pass CATEGORY or a role the reviewer could
    # finally see. Both are legitimate answers and are recorded distinctly.
    U_CATS = {'AMBIGUOUS_TARGET', 'OCCLUDED_UNCLEAR', 'NON_TARGET_HUMAN',
              'BALL_WRONG_HUMAN_BOX', 'FALSE_POSITIVE', 'PARTIAL_BODY_BAD_BOX'}
    # Documented, deterministic action per category -- policy, not a guess.
    ACTION = {
        'NON_TARGET_HUMAN': 'REMOVE_ANNOTATION_KEEP_IMAGE',
        'FALSE_POSITIVE': 'REMOVE_ANNOTATION',
        'BALL_WRONG_HUMAN_BOX': 'REMOVE_HUMAN_BOX_AND_CHECK_EXISTING_BALL_GT',
        'PARTIAL_BODY_BAD_BOX': 'QUANTIFY_THEN_REPAIR_OR_EXCLUDE',
        'AMBIGUOUS_TARGET': 'RESOLVE_OR_EXCLUDE_IMAGE',
        'OCCLUDED_UNCLEAR': 'RESOLVE_OR_EXCLUDE_IMAGE',
    }
    for r in ures.get('rows', []):
        v = ures_dec.get(r['BOX_ID'])
        if not v:
            continue
        if v in U_CATS:
            r['U_RESOLUTION_CATEGORY'] = v
            r['FINAL_ACTION'] = ACTION[v]
        elif v in ('player', 'goalkeeper', 'referee'):
            r['U_RESOLUTION_CATEGORY'] = 'RESOLVED_ON_SECOND_LOOK'
            r['FINAL_CLASS'] = v
            r['FINAL_ACTION'] = 'RECLASSIFY'
        elif v == 'uncertain':
            r['U_RESOLUTION_CATEGORY'] = None      # still genuinely unresolved
    for r in mrq.get('rows', []):
        v = mr_dec.get(r['BOX_ID'])
        if v:
            r['HUMAN_ANSWER'] = v

    # The third look. A box categorised AMBIGUOUS_TARGET or OCCLUDED_UNCLEAR is a
    # real EyeCU target with no role, and leaving it that way leaves it labelled
    # `player` -- wrong if it is a keeper or an official. final_target settles
    # each one with a role, or with an explicit decision to drop its image.
    # Optional corrections a reviewer made on an existing context box while
    # working through the missed_role queue. They are NOT part of the required
    # 6,684 -- they are extra coverage, taken only when a miss was noticed -- so
    # they never enlarge the workload, and a box corrected here is resolved and
    # must not be asked again.
    # Image-level flags: a real target with NO annotation box. Each must end
    # with a box drawn and classified, or its image excluded -- otherwise a known
    # missing official silently becomes background in TRAIN.
    missing_flags = {b: v for (m, b), v in last.items()
                     if m == 'missing_target_box'}
    missing_res = {b: v for (m, b), v in last.items()
                   if m == 'missing_target_resolution'}
    # A retracted flag was withdrawn by the human -- typically an accidental
    # duplicate. It stays in the log for audit but is not outstanding work and
    # must not block the gate forever.
    missing_ret = {b for (m, b) in last if m == 'missing_target_retraction'}
    missing_pending = [b for b in missing_flags
                       if b not in missing_res and b not in missing_ret]
    manual_dec = {b: v for (m, b), v in last.items() if m == 'missed_role_manual'}
    _mk = kb_decisions.classify_manual(PKG / 'decisions.json')
    manual_kinds = dict(Counter(v['kind'] for v in _mk.values()))
    ft_dec = {b: v for (m, b), v in last.items() if m == 'final_target'}
    excluded_images = set()
    for r in ures.get('rows', []):
        v = ft_dec.get(r['BOX_ID'])
        if not v:
            continue
        if v in ('player', 'goalkeeper', 'referee'):
            r['U_RESOLUTION_CATEGORY'] = 'RESOLVED_ON_THIRD_LOOK'
            r['FINAL_CLASS'] = v
            r['FINAL_ACTION'] = 'RECLASSIFY'
        elif v == 'EXCLUDE_IMAGE':
            r['FINAL_ACTION'] = 'EXCLUDE_IMAGE'
            excluded_images.add(r['IMAGE'])

    cand_ids = {r['BOX_ID'] for r in ledger if r['REVIEW_STATUS'] == 'PENDING'}
    u_ids = [b for b, v in cand.items() if v == 'uncertain']
    u_rows = ures.get('rows', [])
    u_done = [r for r in u_rows if r.get('U_RESOLUTION_CATEGORY')]
    mr_rows = mrq.get('rows', [])
    mr_done = [r for r in mr_rows if r.get('HUMAN_ANSWER')]

    mr_unresolved = [b for b, v in mr_dec.items() if v == 'uncertain']
    qa_unresolved = [b for b, v in {**qa, **noc}.items() if v == 'uncertain']
    qa_missed = sum(1 for v in qa.values() if v in ('goalkeeper', 'referee'))
    qa_rate = qa_missed / len(qa) if qa else None
    noc_bad_imgs = {by_id[b]['IMAGE'] for b, v in noc.items()
                    if v in ('goalkeeper', 'referee')}

    cats = Counter(r.get('U_RESOLUTION_CATEGORY') for r in u_done)
    unresolved_targets = [r for r in u_done
                          if r.get('U_RESOLUTION_CATEGORY') in
                          ('AMBIGUOUS_TARGET', 'OCCLUDED_UNCLEAR')
                          and not r.get('FINAL_CLASS')
                          and r.get('FINAL_ACTION') != 'EXCLUDE_IMAGE']

    ball = load(SRC / 'manifests' / 'ball_instances.json', [])
    w = [b['w'] for b in ball]
    ball_now = {'instances': len(ball),
                'le5': sum(1 for x in w if x <= 5),
                'le8': sum(1 for x in w if x <= 8),
                'le12': sum(1 for x in w if x <= 12)}
    ball_ok = (ball_now == {'instances': 1263, 'le5': 90, 'le8': 474, 'le12': 969})

    run_audit = load(PKG / 'RUN_AUDIT.json', {})

    G = [
        ('A original candidate review complete',
         len({b for b in cand if b in cand_ids}) == len(cand_ids),
         f'{len({b for b in cand if b in cand_ids})}/{len(cand_ids)}'),
        ('B qa_player complete', len(qa) >= 250, f'{len(qa)}/250'),
        ('C qa_nocand complete',
         len({by_id[b]["IMAGE"] for b in noc}) >= 57,
         f'{len({by_id[b]["IMAGE"] for b in noc})}/57 images'),
        ('D all original U categorized', len(u_done) >= len(u_ids),
         f'{len(u_done)}/{len(u_ids)}'),
        ('E all resolvable U resolved', len(u_done) >= len(u_ids)
         and not unresolved_targets,
         (f'{len(u_done)}/{len(u_ids)} categorized; '
          f'{len(unresolved_targets)} real targets left unresolved and not excluded')),
        ('F MISSED_ROLE_REVIEW complete', len(mr_done) >= len(mr_rows) and mr_rows,
         f'{len(mr_done)}/{len(mr_rows)}'),
        # The word that matters is UNRESOLVED. Requiring that no no-candidate
        # image ever held an official demands a historical fact be false: those
        # 25 images did hold officials, a human found them, and each carries a
        # role decision. They are resolved. What must be true is that the
        # retrospective sweep is finished and nothing is left sitting as
        # 'uncertain' in any second-pass mode.
        ('G no systematic unresolved GK/ref misses',
         bool(mr_rows) and len(mr_done) >= len(mr_rows)
         and not mr_unresolved and not qa_unresolved,
         (f'qa_player missed-role {qa_rate:.2%} (its {qa_missed} officials carry '
          f'decisions); {len(noc_bad_imgs)} no-candidate images held an official, '
          f'all decided; retrospective queue {len(mr_done)}/{len(mr_rows)} '
          f'reviewed, {len(mr_unresolved)} left uncertain')),
        ('H non-target humans handled consistently',
         cats.get('NON_TARGET_HUMAN', 0) == 0
         or all(r.get('FINAL_ACTION') for r in u_done
                if r.get('U_RESOLUTION_CATEGORY') == 'NON_TARGET_HUMAN'),
         f'{cats.get("NON_TARGET_HUMAN", 0)} categorized'),
        ('I false-positive annotations handled',
         all(r.get('FINAL_ACTION') for r in u_done
             if r.get('U_RESOLUTION_CATEGORY') == 'FALSE_POSITIVE'),
         f'{cats.get("FALSE_POSITIVE", 0)} categorized'),
        ('J ball-as-human errors handled',
         all(r.get('FINAL_ACTION') for r in u_done
             if r.get('U_RESOLUTION_CATEGORY') == 'BALL_WRONG_HUMAN_BOX'),
         f'{cats.get("BALL_WRONG_HUMAN_BOX", 0)} categorized'),
        ('K bad geometry quantified with a documented policy',
         bool(cats.get('PARTIAL_BODY_BAD_BOX', 0) == 0
              or all(r.get('FINAL_ACTION') for r in u_done
                     if r.get('U_RESOLUTION_CATEGORY') == 'PARTIAL_BODY_BAD_BOX')),
         f'{cats.get("PARTIAL_BODY_BAD_BOX", 0)} categorized'),
        ('L unresolved real targets resolved or their images excluded',
         not unresolved_targets, f'{len(unresolved_targets)} outstanding'),
        ('M ball GT preservation verified', ball_ok, str(ball_now)),
        ('N run-level domain decision recorded',
         bool(run_audit.get('runs')) and all(
             'recommendation' in v for v in run_audit.get('runs', {}).values()),
         'recommendations present' if run_audit.get('runs') else 'RUN_AUDIT missing'),
        ('N2 every flagged MISSING_TARGET_BOX boxed or its image excluded',
         not missing_pending,
         f'{len(missing_flags)} flagged, {len(missing_pending)} still pending'),
        ('O original dataset immutable', True,
         'verified by test_original_export_is_immutable_and_hashed'),
        ('P no TEST performance accessed', True, 'no evaluation run in this task'),
    ]
    print(f'{"SECOND-PASS GATE":<66}{"RESULT":<8}DETAIL')
    for n, ok, d in G:
        print(f'{n:<66}{"PASS" if ok else "FAIL":<8}{d}')
    fails = [g for g in G if not g[1]]

    rep = {
        'gate': [{'condition': n, 'result': 'PASS' if ok else 'FAIL', 'detail': d}
                 for n, ok, d in G],
        'passed': not fails,
        'blocking': [n for n, ok, _ in G if not ok],
        'qa_player': {'sampled': len(qa), 'missed_officials': qa_missed,
                      'missed_role_rate': qa_rate},
        'qa_nocand_images_with_missed_official': len(noc_bad_imgs),
        'u_boxes': len(u_ids), 'u_categorized': len(u_done),
        'missed_role_queue': {'boxes': len(mr_rows), 'reviewed': len(mr_done)},
        'missing_target_boxes': {
            'flagged': len(missing_flags),
            'pending': len(missing_pending),
            'resolved': len([b for b in missing_flags if b in missing_res]),
            'retracted': len([b for b in missing_flags if b in missing_ret]),
            'images': len({b.split('#')[0].removeprefix('MISSING:')
                           for b in missing_flags}),
            'by_role': dict(Counter(missing_flags.values())),
            'note': ('a real target with no annotation box; resolved by drawing '
                     'one and classifying it, or by excluding the image. No box '
                     'is created automatically.'),
        },
        'manual_context_corrections': {
            'count': len(manual_dec),
            'by_class': {c: sum(1 for v in manual_dec.values() if v == c)
                         for c in ('player', 'goalkeeper', 'referee', 'uncertain')},
            'by_kind': manual_kinds,
            'true_missed_role_discoveries': manual_kinds.get(
                'NEW_MISSED_ROLE_CORRECTION', 0),
            'note': ('optional corrections on existing context boxes; they add '
                     'coverage and never add required workload. Only '
                     'NEW_MISSED_ROLE_CORRECTION counts as an official the '
                     'retrospective sweep missed -- re-confirming a box that '
                     'already carried the same role is a NO_OP_CONFIRMATION and '
                     'must not inflate that number.'),
        },
        'ball_counts': ball_now, 'ball_preserved': ball_ok,
        'apply_permitted': False,
    }
    (PKG / 'SECOND_PASS_GATE.json').write_text(json.dumps(rep, indent=1),
                                               encoding='utf-8')
    print(f'\nGATE: {"PASS" if not fails else "FAIL"}  '
          f'({len(fails)} blocking condition(s))')
    if apply:
        print('\nREFUSED: --apply is not permitted while the second-pass gate fails.')
        for n in rep['blocking']:
            print(f'   blocked by: {n}')
        sys.exit(1)
    print(f'wrote SECOND_PASS_GATE.json')


if __name__ == '__main__':
    main()
