#!/usr/bin/env python
"""
Build the MISSING_TARGET_BOX queue: only the images a human flagged.

Some real EyeCU targets carry no annotation at all, so there is nothing to click
and nothing to reclassify. Those were flagged image-level during the missed_role
pass. This turns the flags into a work list.

The queue contains ONLY flagged images. It is deliberately not another sweep of
all 1,133 -- the whole point of flagging in-pass was to avoid that.

No box is created or inferred here. Each entry is a request for a human to draw
one, or to exclude the image. Both are resolved by recording a
`missing_target_resolution` decision:

    boxed_player / boxed_goalkeeper / boxed_referee   a box was drawn and classed
    EXCLUDE_IMAGE                                     drop the image instead

The gate blocks until every flag has one.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import kb_decisions                                              # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
RESOLUTIONS = ('boxed_player', 'boxed_goalkeeper', 'boxed_referee',
               'EXCLUDE_IMAGE')


def collect():
    """Flags, retractions and resolutions, from the append-only log.

    A retracted flag stays in history -- the log is never rewritten -- but stops
    being live work. That distinction matters: an accidental duplicate should not
    generate annotation effort, and it should not block the gate either, but the
    fact that it was raised and withdrawn remains visible.
    """
    flags, resolved, retracted = {}, {}, {}
    for r in kb_decisions.read_log(PKG / 'decisions.json'):
        if r['mode'] == 'missing_target_box':
            flags[r['BOX_ID']] = r
        elif r['mode'] == 'missing_target_resolution':
            resolved[r['BOX_ID']] = r
        elif r['mode'] == 'missing_target_retraction':
            retracted[r['BOX_ID']] = r
    return flags, resolved, retracted


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ledger = {r['BOX_ID']: r for r in
              json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))}
    by_img = defaultdict(list)
    for r in ledger.values():
        by_img[r['IMAGE']].append(r)

    flags, resolved, retracted = collect()
    rows = []
    for key, f in flags.items():
        img = f.get('IMAGE')
        res = resolved.get(key)
        ret = retracted.get(key)
        rows.append({
            'key': key, 'IMAGE': img, 'run': f.get('run'),
            'missing_role': f['HUMAN_FINAL_CLASS'],
            'flagged_utc': f.get('recorded_utc'),
            'existing_boxes_in_image': len(by_img.get(img, [])),
            'RESOLUTION': res['HUMAN_FINAL_CLASS'] if res else None,
            'resolved_utc': res.get('recorded_utc') if res else None,
            'retracted': bool(ret),
            'retraction_reason': ret.get('reason') if ret else None,
            'retracted_utc': ret.get('recorded_utc') if ret else None,
            'status': ('RETRACTED' if ret else
                       'RESOLVED' if res else 'PENDING'),
        })
    rows.sort(key=lambda r: (r['status'] != 'PENDING', r['IMAGE'] or '',
                             r['flagged_utc'] or ''))
    pending = [r for r in rows if r['status'] == 'PENDING']
    imgs = {r['IMAGE'] for r in rows if r['status'] != 'RETRACTED'}

    q = {
        'source_log': kb_decisions.log_version(PKG / 'decisions.json'),
        'purpose': ('images where a human saw a real EyeCU target with NO '
                    'annotation box; each entry is a request to draw one or to '
                    'exclude the image'),
        'no_box_created_or_inferred': True,
        'not_another_full_pass': ('contains only flagged images -- '
                                  f'{len(imgs)}, not the 1,133 reviewed'),
        'allowed_resolutions': list(RESOLUTIONS),
        'resolution_mode': 'missing_target_resolution',
        'flags': len(rows), 'images': len(imgs),
        'pending': len(pending),
        'resolved': sum(1 for r in rows if r['status'] == 'RESOLVED'),
        'retracted': sum(1 for r in rows if r['status'] == 'RETRACTED'),
        'retraction_note': ('a retracted flag stays in this file for audit but is '
                            'not pending work and does not block the gate'),
        'by_missing_role': dict(Counter(r['missing_role'] for r in rows)),
        'by_run': dict(Counter(r['run'] for r in rows)),
        'rows': rows,
    }
    (PKG / 'missing_target_queue.json').write_text(json.dumps(q, indent=1),
                                                   encoding='utf-8')
    print(f"MISSING_TARGET_BOX queue: {q['flags']} flags across {q['images']} "
          f"live images ({q['pending']} pending, {q['resolved']} resolved, "
          f"{q['retracted']} retracted)")
    if rows:
        print(f"  by missing role: {q['by_missing_role']}")
        print(f"  by run         : {q['by_run']}")
        for r in pending[:10]:
            print(f"    PENDING {r['missing_role']:<11} {r['IMAGE']}")
    print('wrote missing_target_queue.json')


if __name__ == '__main__':
    main()
