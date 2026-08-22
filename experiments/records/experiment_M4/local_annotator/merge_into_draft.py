#!/usr/bin/env python
"""
Merge LOCAL_ANNOTATIONS.json (from annotator_server.py) into
ANNOTATIONS_DRAFT.json. Nearly a straight concatenation since both files
already share the same per-frame record schema -- this just drops the
annotator's extra bookkeeping fields (width/height/saved/saved_utc) so the
merged file matches the schema the first 20 in-session records already use,
and refuses to run until all 40 handoff frames are saved.

    python experiments/records/experiment_M4/local_annotator/merge_into_draft.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
M4 = HERE.parent
LOCAL = HERE / 'LOCAL_ANNOTATIONS.json'
DRAFT = M4 / 'ANNOTATIONS_DRAFT.json'
MANIFEST = M4 / 'HANDOFF_EXTERNAL_ANNOTATION' / 'HANDOFF_MANIFEST.json'


def main():
    expected = {(e['sequence'], e['frame_number_1based'])
               for e in json.loads(MANIFEST.read_text(encoding='utf-8'))}
    local = json.loads(LOCAL.read_text(encoding='utf-8')) if LOCAL.exists() else []
    got = {(r['sequence'], r['frame_number_1based']) for r in local}
    missing = sorted(expected - got)
    if missing:
        print(f'{len(missing)} of {len(expected)} handoff frames not yet saved:')
        for s, n in missing[:20]:
            print(f'  {s} {n}')
        if len(missing) > 20:
            print(f'  ... and {len(missing) - 20} more')
        print('\nNot merging -- finish annotating first (or re-run with all frames saved).')
        return 1

    draft = json.loads(DRAFT.read_text(encoding='utf-8')) if DRAFT.exists() else []
    existing = {(r['sequence'], r['frame_number_1based']) for r in draft}
    dup = existing & got
    if dup:
        print(f'{len(dup)} frames already present in ANNOTATIONS_DRAFT.json -- not overwriting:')
        for s, n in sorted(dup):
            print(f'  {s} {n}')
        print('\nRefusing to merge to avoid silently replacing existing records.')
        return 1

    for r in sorted(local, key=lambda r: (r['sequence'], r['frame_number_1based'])):
        draft.append({
            'sequence': r['sequence'],
            'frame_number_1based': r['frame_number_1based'],
            'file': r['file'],
            'objects': r['objects'],
            'notes': r.get('notes', '') or 'annotated locally via experiment_M4/local_annotator',
        })
    draft.sort(key=lambda r: (r['sequence'], r['frame_number_1based']))
    DRAFT.write_text(json.dumps(draft, indent=1), encoding='utf-8')
    print(f'merged {len(local)} frames into {DRAFT} ({len(draft)} total records)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
