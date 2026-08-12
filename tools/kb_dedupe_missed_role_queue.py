#!/usr/bin/env python
"""
Drop already-answered boxes from the missed_role queue, keeping a record of each.

qa_player and qa_nocand sampled FROM the same LIKELY_PLAYER pool the retrospective
queue was scored over, so 301 boxes a human had already judged were queued to be
judged again. That is 4.3% of the queue wasted, and worse: re-asking a settled
question is an invitation to answer it differently, which is exactly the
precedence hazard this round of checks exists to remove.

Boxes still genuinely open -- answered 'uncertain' -- are KEPT. They are not
settled, and the second pass is the right place to settle them.

Nothing is deleted silently. Every removed box is written to `prefilled` with the
answer it already carries and the mode that produced it, so the queue's original
population stays reconstructable.
"""

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import kb_decisions                                              # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    q = json.loads((PKG / 'missed_role_queue.json').read_text(encoding='utf-8'))
    resolved = kb_decisions.resolve(PKG / 'decisions.json')

    keep, prefilled = [], []
    for r in q['rows']:
        st = resolved.get(r['BOX_ID'])
        if st and st['final_class'] in kb_decisions.ROLES:
            prefilled.append({
                'BOX_ID': r['BOX_ID'], 'IMAGE': r['IMAGE'], 'run': r['run'],
                'proposed_missed_role': r['proposed_missed_role'],
                'already_answered': st['final_class'],
                'answered_in_mode': st['decided_in_mode'],
                'recorded_utc': st['recorded_utc'],
            })
        else:
            keep.append(r)

    open_unresolved = [r for r in keep
                       if resolved.get(r['BOX_ID'], {}).get('disposition') == 'UNRESOLVED']
    print(f'queue was {len(q["rows"])} boxes')
    print(f'  already answered elsewhere, removed : {len(prefilled)}')
    print(f'     by mode  : {dict(Counter(p["answered_in_mode"] for p in prefilled))}')
    print(f'     answers  : {dict(Counter(p["already_answered"] for p in prefilled))}')
    print(f'  answered "uncertain", KEPT as open  : {len(open_unresolved)}')
    print(f'queue is now {len(keep)} boxes in '
          f'{len({r["IMAGE"] for r in keep})} images')

    q['rows'] = keep
    q['queue_boxes'] = len(keep)
    q['queue_images'] = len({r['IMAGE'] for r in keep})
    q['by_run'] = dict(Counter(r['run'] for r in keep))
    q['proposed_role_mix'] = dict(Counter(r['proposed_missed_role'] for r in keep))
    q['deduplication'] = {
        'why': ('qa_player and qa_nocand sampled from the same LIKELY_PLAYER pool '
                'this queue was scored over, so boxes a human had already judged '
                'were queued again'),
        'original_queue_boxes': len(q['rows']) + len(prefilled),
        'removed_already_answered': len(prefilled),
        'kept_because_still_unresolved': len(open_unresolved),
        'nothing_deleted_silently': True,
    }
    q['prefilled'] = prefilled
    (PKG / 'missed_role_queue.json').write_text(json.dumps(q, indent=1),
                                                encoding='utf-8')
    print('\nwrote missed_role_queue.json (prefilled section keeps every removal)')


if __name__ == '__main__':
    main()
