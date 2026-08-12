#!/usr/bin/env python
"""
The single, explicit precedence rule for decisions.json. One implementation.

Before this existed the two consumers disagreed. The applier keyed by BOX_ID
alone, so whichever line happened to sit last in the file won -- chronological by
accident, not by rule, and silently wrong if lines were ever reordered or two
servers appended concurrently. The gate keyed by (mode, BOX_ID), treating modes
as independent namespaces, so it had no cross-mode precedence at all. For a box
answered 'referee' in qa_nocand and later 'player' in missed_role, those two
consumers would not have agreed on the answer.

THE RULE, and there is only one:

    the human's LATEST decision for a box wins.

    1. later recorded_utc wins
    2. same timestamp -> later line in the append-only file wins
    3. mode is NOT a rank. Time is the only authority, because a later look is
       by definition the more informed one. A second-pass resolution therefore
       always beats an earlier proposal or decision, and can never be silently
       overridden by one.

Roles and dispositions are kept apart. player/goalkeeper/referee are ROLES and
set the final class. 'uncertain' is an explicit non-resolution and clears it. The
U categories (NON_TARGET_HUMAN, FALSE_POSITIVE, ...) are DISPOSITIONS -- what to
do with the annotation -- and never masquerade as a class.

Nothing here mutates decisions.json. It is append-only and is never rewritten.
"""

import json
from pathlib import Path

ROLES = ('player', 'goalkeeper', 'referee')
UNRESOLVED = 'uncertain'
U_CATEGORIES = ('AMBIGUOUS_TARGET', 'OCCLUDED_UNCLEAR', 'NON_TARGET_HUMAN',
                'BALL_WRONG_HUMAN_BOX', 'FALSE_POSITIVE', 'PARTIAL_BODY_BAD_BOX',
                # a real target whose role could not be read even on a third look;
                # the honest answer is to drop its image rather than guess a class
                'EXCLUDE_IMAGE')
# Documented action per disposition. Policy, applied deterministically.
DISPOSITION_ACTION = {
    'NON_TARGET_HUMAN': 'REMOVE_ANNOTATION_KEEP_IMAGE',
    'FALSE_POSITIVE': 'REMOVE_ANNOTATION',
    'BALL_WRONG_HUMAN_BOX': 'REMOVE_HUMAN_BOX_AND_CHECK_EXISTING_BALL_GT',
    'PARTIAL_BODY_BAD_BOX': 'QUANTIFY_THEN_REPAIR_OR_EXCLUDE',
    'AMBIGUOUS_TARGET': 'RESOLVE_OR_EXCLUDE_IMAGE',
    'OCCLUDED_UNCLEAR': 'RESOLVE_OR_EXCLUDE_IMAGE',
    'EXCLUDE_IMAGE': 'EXCLUDE_IMAGE_FROM_CANDIDATE_SET',
}


def read_log(path: Path):
    """Every decision line, in file order, with its index. Never rewritten."""
    rows = []
    if not Path(path).exists():
        return rows
    for i, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
        if line.strip():
            d = json.loads(line)
            d.setdefault('mode', 'candidates')
            d['_line'] = i
            rows.append(d)
    return rows


def _key(d):
    # missing timestamps sort first, so an untimestamped row can never
    # outrank a timestamped one purely by luck of file position
    return (d.get('recorded_utc') or '', d['_line'])


def resolve(path: Path):
    """Final state per BOX_ID under the documented rule.

    Returns {BOX_ID: {final_class, disposition, decided_in_mode, recorded_utc,
                      history, superseded}}.
    """
    rows = read_log(path)
    per = {}
    for d in rows:
        per.setdefault(d['BOX_ID'], []).append(d)
    out = {}
    for box, hist in per.items():
        hist = sorted(hist, key=_key)
        winner = hist[-1]
        v = winner['HUMAN_FINAL_CLASS']
        final_class = v if v in ROLES else None
        disposition = v if v in U_CATEGORIES else (
            'UNRESOLVED' if v == UNRESOLVED else None)
        out[box] = {
            'final_class': final_class,
            'disposition': disposition,
            'action': DISPOSITION_ACTION.get(v),
            'decided_in_mode': winner['mode'],
            'recorded_utc': winner.get('recorded_utc'),
            'decisions_recorded': len(hist),
            'superseded': [{'mode': h['mode'],
                            'value': h['HUMAN_FINAL_CLASS'],
                            'recorded_utc': h.get('recorded_utc')}
                           for h in hist[:-1]],
            'history': [{'mode': h['mode'], 'value': h['HUMAN_FINAL_CLASS'],
                         'recorded_utc': h.get('recorded_utc'),
                         'line': h['_line']} for h in hist],
        }
    return out


def by_mode(path: Path):
    """Latest answer per (mode, BOX_ID). For population/completion counting only.

    The gate legitimately needs this -- "is the missed_role queue finished" is a
    per-mode question -- but it must never be used to decide a box's class.
    """
    rows = read_log(path)
    per = {}
    for d in rows:
        per.setdefault((d['mode'], d['BOX_ID']), []).append(d)
    return {k: sorted(v, key=_key)[-1]['HUMAN_FINAL_CLASS'] for k, v in per.items()}


MANUAL_MODE = 'missed_role_manual'
NEW_CORRECTION = 'NEW_MISSED_ROLE_CORRECTION'
OVERRIDE = 'HUMAN_OVERRIDE'
NO_OP = 'NO_OP_CONFIRMATION'
FLAGGED = 'FLAGGED_UNCERTAIN'
DISPOSITIONED = 'DISPOSITION_SET'


def prior_non_manual(path: Path, box: str):
    """The box's state before any manual click: (raw value, mode) or (None, None).

    Manual clicks are excluded deliberately. "Did this click find a missed
    official" is a question about what the earlier passes had recorded, so an
    earlier manual click on the same box must not be what a later one is judged
    against -- otherwise a correction followed by a re-confirmation would read as
    two separate findings.
    """
    hist = sorted((d for d in read_log(path)
                   if d['BOX_ID'] == box and d['mode'] != MANUAL_MODE), key=_key)
    if not hist:
        return None, None
    return hist[-1]['HUMAN_FINAL_CLASS'], hist[-1]['mode']


def classify_click(prior_value, value):
    """One click, classified against what the box held before it. One rule.

    The server classifies a click as it arrives and the auditor re-classifies the
    whole log afterwards. Those two answers have to agree, so they call this.
    """
    if value in U_CATEGORIES:
        # e.g. NON_TARGET_HUMAN on a context box: a real decision that
        # settles the box, but not a role and not a missed-role discovery
        return DISPOSITIONED
    if value == UNRESOLVED:
        return FLAGGED
    if prior_value is not None and value == prior_value:
        return NO_OP
    if prior_value in (None, 'player') and value in ('goalkeeper', 'referee'):
        return NEW_CORRECTION
    if prior_value in (None, 'player') and value == 'player':
        return NO_OP
    if prior_value in ROLES:
        return OVERRIDE
    return NO_OP


def classify_manual(path: Path):
    """What each manual context-box click actually did.

    Clicking a box that already carries the same human role is not a discovery.
    Counting it as one would inflate the headline number this whole pass exists
    to produce -- "how many officials did the retrospective sweep miss" -- with
    re-confirmations of officials that were already found.

    Each missed_role_manual decision is compared against the box's state as it
    stood BEFORE that click, taken from any other mode:

        NEW_MISSED_ROLE_CORRECTION  it was unresolved or plain player, and is now
                                    goalkeeper or referee -- a real find
        HUMAN_OVERRIDE              it already had a role and the human changed it
                                    to a different one -- intentional, kept
        NO_OP_CONFIRMATION          the answer matches what was already there
        FLAGGED_UNCERTAIN           marked uncertain; neither a find nor a no-op
        DISPOSITION_SET             a disposition such as NON_TARGET_HUMAN --
                                    settles the box, but is not a role and is
                                    not a missed-role discovery
    """
    rows = read_log(path)
    per = {}
    for d in rows:
        per.setdefault(d['BOX_ID'], []).append(d)
    out = {}
    for d in rows:
        if d['mode'] != MANUAL_MODE:
            continue
        hist = sorted(per[d['BOX_ID']], key=_key)
        prior = [h for h in hist
                 if _key(h) < _key(d) and h['mode'] != MANUAL_MODE]
        pv = prior[-1]['HUMAN_FINAL_CLASS'] if prior else None
        pm = prior[-1]['mode'] if prior else None
        v = d['HUMAN_FINAL_CLASS']
        kind = classify_click(pv, v)
        # a later manual click supersedes an earlier one on the same box
        out[d['BOX_ID']] = {'kind': kind, 'manual_class': v,
                            'prior_class': pv, 'prior_mode': pm,
                            'recorded_utc': d.get('recorded_utc')}
    return out


def conflicts(path: Path):
    """Boxes whose latest answer differs from an earlier one. Not an error."""
    out = []
    for box, r in resolve(path).items():
        vals = {h['value'] for h in r['history']}
        if len(vals) > 1:
            out.append({'BOX_ID': box, 'final': r['history'][-1]['value'],
                        'won_in_mode': r['decided_in_mode'],
                        'history': r['history']})
    return out


if __name__ == '__main__':
    import sys
    p = Path(sys.argv[1] if len(sys.argv) > 1 else
             'experiments/external_sources/keremberke_review/decisions.json')
    r = resolve(p)
    c = conflicts(p)
    print(f'{len(read_log(p))} log lines -> {len(r)} boxes')
    print(f'roles assigned: {sum(1 for v in r.values() if v["final_class"])}')
    print(f'unresolved    : {sum(1 for v in r.values() if v["disposition"] == "UNRESOLVED")}')
    print(f'dispositions  : {sum(1 for v in r.values() if v["disposition"] and v["disposition"] != "UNRESOLVED")}')
    print(f'boxes with a changed answer: {len(c)}')
