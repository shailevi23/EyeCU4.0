#!/usr/bin/env python
"""
Consolidate every external source into one canonical workspace.

Moves, never copies, and never trusts the move. Every file is hashed before it
is moved and re-hashed at its destination; a mismatch aborts. 59 GB of video and
GSR JSON is not something to re-download because a rename went wrong.

Byte-identical duplicates are found by hash, not by filename, and only one copy
survives -- with the removal recorded, so a missing file is explainable later.
The SoccerTrack download shipped its RAW archive twice, under raw/ and again
under gsr/, which is 282 MB of nothing.

Nothing here deletes a source. Deduplication removes redundant COPIES of a file
that still exists at the canonical path, and says which hash it matched.
"""

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / 'EyeCU_external_data'

TREE = [
    'roboflow_audit/raw_zips', 'roboflow_audit/reports', 'roboflow_audit/manifests',
    'huggingface/keremberke_football_object_detection/raw',
    'huggingface/keremberke_football_object_detection/metadata',
    'huggingface/keremberke_football_object_detection/manifests',
    'huggingface/soccernet_v3/metadata_only',
    'huggingface/soccernet_v3/full_dataset',
    'huggingface/soccernet_v3/manifests',
    'huggingface/manifests', 'huggingface/download_logs',
    'soccertrack_v2/gsr', 'soccertrack_v2/bas', 'soccertrack_v2/raw',
    'soccertrack_v2/videos', 'soccertrack_v2/public_repo',
    'soccertrack_v2/reports', 'soccertrack_v2/manifests',
    'research_sources/teamtrack', 'research_sources/sportslabkit',
    'research_sources/manifests',
]

# (source relative to EXT, destination directory relative to EXT)
MOVES = [
    ('SoccerTrackV2/gsr', 'soccertrack_v2/gsr'),
    ('SoccerTrackV2/bas', 'soccertrack_v2/bas'),
    ('SoccerTrackV2/raw', 'soccertrack_v2/raw'),
    ('SoccerTrackV2/mot', 'soccertrack_v2/raw'),      # the empty archive, kept as evidence
    ('SoccerTrackV2/videos', 'soccertrack_v2/videos'),
    ('SoccerTrack-v2', 'soccertrack_v2/public_repo'),
]


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
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    log = {'created': [], 'moved': [], 'verified': 0, 'deduplicated': [],
           'errors': [], 'skipped': []}

    for d in TREE:
        p = EXT / d
        if not p.exists():
            if not args.dry_run:
                p.mkdir(parents=True, exist_ok=True)
            log['created'].append(d)

    for src_rel, dst_rel in MOVES:
        src = EXT / src_rel
        dst = EXT / dst_rel
        if not src.exists():
            log['skipped'].append(f'{src_rel} (absent)')
            continue
        files = [p for p in src.rglob('*') if p.is_file()]
        print(f'{src_rel} -> {dst_rel}  ({len(files)} files)')
        for f in files:
            rel = f.relative_to(src)
            # the public repo keeps its internal layout; data dirs flatten
            target = dst / rel if src_rel == 'SoccerTrack-v2' else dst / f.name
            if target.exists() and target.resolve() == f.resolve():
                continue
            before = sha256(f)
            if args.dry_run:
                log['moved'].append({'from': str(f.relative_to(EXT)).replace('\\', '/'),
                                     'to': str(target.relative_to(EXT)).replace('\\', '/'),
                                     'sha256': before, 'dry_run': True})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha256(target) == before:
                    f.unlink()
                    log['deduplicated'].append(
                        {'removed': str(f.relative_to(EXT)).replace('\\', '/'),
                         'identical_to': str(target.relative_to(EXT)).replace('\\', '/'),
                         'sha256': before})
                    continue
                log['errors'].append(f'{target} exists with different content')
                continue
            shutil.move(str(f), str(target))
            after = sha256(target)
            if after != before:
                log['errors'].append(f'HASH MISMATCH moving {f} -> {target}')
                sys.exit(f'HASH MISMATCH: {f}')
            log['verified'] += 1
            log['moved'].append({'from': str(f.relative_to(EXT)).replace('\\', '/'),
                                 'to': str(target.relative_to(EXT)).replace('\\', '/'),
                                 'sha256': after})

    # remove now-empty source trees, but never a directory that still holds data
    if not args.dry_run:
        for src_rel, _ in MOVES:
            src = EXT / src_rel
            if src.exists() and not any(p.is_file() for p in src.rglob('*')):
                shutil.rmtree(src, ignore_errors=True)
                log['created'].append(f'removed empty {src_rel}')
        for leftover in ('SoccerTrackV2', 'soccertrack_mot'):
            p = EXT / leftover
            if p.exists() and not any(x.is_file() for x in p.rglob('*')):
                shutil.rmtree(p, ignore_errors=True)
                log['created'].append(f'removed empty {leftover}')

    # cross-workspace duplicate sweep, by content
    by_hash = defaultdict(list)
    for p in sorted(EXT.rglob('*')):
        if p.is_file() and 'public_repo' not in p.parts and p.stat().st_size > 1 << 20:
            by_hash[sha256(p)].append(str(p.relative_to(EXT)).replace('\\', '/'))
    dupes = {h: v for h, v in by_hash.items() if len(v) > 1}
    log['remaining_duplicate_groups'] = dupes

    print(f'\ncreated {len(log["created"])} dirs, moved {len(log["moved"])} files, '
          f'verified {log["verified"]}, deduplicated {len(log["deduplicated"])}')
    if dupes:
        print('remaining duplicate groups:')
        for h, v in dupes.items():
            print(f'  {h[:16]}  {v}')
    if log['errors']:
        print('ERRORS:', log['errors'])

    if not args.dry_run:
        (EXT / 'huggingface' / 'download_logs').mkdir(parents=True, exist_ok=True)
        (EXT / 'consolidation_log.json').write_text(
            json.dumps(log, indent=1), encoding='utf-8')
        print('wrote EyeCU_external_data/consolidation_log.json')


if __name__ == '__main__':
    main()
