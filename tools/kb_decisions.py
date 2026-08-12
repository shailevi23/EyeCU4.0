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
                'BALL_WRONG_HUMAN_BOX', 'FALSE_POSITIVE', 'PARTIAL_BODY_BAD_BOX')
# Documented action per disposition. Policy, applied deterministically.
DISPOSITION_ACTION = {
    'NON_TARGET_HUMAN': 'REMOVE_ANNOTATION_KEEP_IMAGE',
    'FALSE_POSITIVE': 'REMOVE_ANNOTATION',
    'BALL_WRONG_HUMAN_BOX': 'REMOVE_HUMAN_BOX_AND_CHECK_EXISTING_BALL_GT',
    'PARTIAL_BODY_BAD_BOX': 'QUANTIFY_THEN_REPAIR_OR_EXCLUDE',
    'AMBIGUOUS_TARGET': 'RESOLVE_OR_EXCLUDE_IMAGE',
    'OCCLUDED_UNCLEAR': 'RESOLVE_OR_EXCLUDE_IMAGE',
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
