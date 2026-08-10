#!/usr/bin/env python
"""
Compare identity annotation against the reference geometry that already exists
for some frames (see tools/build_human_gt_seed.py).

The reference has UNCONFIRMED provenance, so a disagreement does not mean the
annotation is wrong -- it means the two passes differ and a person should look.
The annotation is the authority; this is a prompt to check, never a correction.

Report only. It never edits annotations.

What a disagreement means:

  missing      the reference has a person the identity pass did not annotate
  extra        the identity pass has a person the reference does not -- usually
               fine, the detector-era pass was not required to be exhaustive
  role         same box, different class: worth a look either way
  loose        matched but IoU below --iou: geometry differs

Run it before QC confirmation, as a prompt for review.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MATCH_IOU = 0.5


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def compare(root: Path, seq: str, ann_path: Path, iou_thresh=MATCH_IOU):
    seed_path = root / 'human_seed' / f'{seq}.json'
    if not seed_path.exists():
        return [], Counter()
    seed = json.loads(seed_path.read_text(encoding='utf-8'))
    ann = json.loads(ann_path.read_text(encoding='utf-8'))

    by_frame = defaultdict(list)
    for b in ann['boxes']:
        by_frame[b['frame']].append(b)

    issues, tally = [], Counter()
    for fr in seed['frames']:
        pf = fr['package_frame']
        got = by_frame.get(pf, [])
        used = set()
        for sb in fr['boxes']:
            best, bi = 0.0, -1
            for i, ab in enumerate(got):
                if i in used:
                    continue
                v = iou(sb['bbox'], ab['bbox'])
                if v > best:
                    best, bi = v, i
            if bi < 0 or best < iou_thresh:
                tally['missing'] += 1
                issues.append(f'{seq} f{pf}: reference has a {sb["role"]} at '
                              f'{sb["bbox"]} with no annotated counterpart')
                continue
            used.add(bi)
            tally['matched'] += 1
            if got[bi]['role'] != sb['role']:
                tally['role'] += 1
                issues.append(f'{seq} f{pf} id{got[bi]["id"]}: annotated as '
                              f'{got[bi]["role"]}, reference geometry says '
                              f'{sb["role"]}')
            if best < 0.8:
                tally['loose'] += 1
                issues.append(f'{seq} f{pf} id{got[bi]["id"]}: IoU {best:.2f} '
                              f'against reference geometry')
        tally['extra'] += len(got) - len(used)
        tally['seed_frames'] += 1
    return issues, tally


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/tracking_val_gt')
    ap.add_argument('--iou', type=float, default=MATCH_IOU)
    ap.add_argument('--max-print', type=int, default=25)
    args = ap.parse_args()

    root = Path(args.root)
    man = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    grand = Counter()
    for s in man['sequences']:
        ann = root / s['annotation_file_expected']
        if not ann.exists():
            print(f'{s["sequence"]}: not annotated yet, skipped')
            continue
        issues, tally = compare(root, s['sequence'], ann, args.iou)
        grand.update(tally)
        print(f'\n=== {s["sequence"]}  ({tally["seed_frames"]} seeded frames)')
        print(f'   matched {tally["matched"]}  missing {tally["missing"]}  '
              f'extra {tally["extra"]}  role {tally["role"]}  '
              f'loose {tally["loose"]}')
        for m in issues[:args.max_print]:
            print(f'   - {m}')
        if len(issues) > args.max_print:
            print(f'   ... {len(issues) - args.max_print} more')

    print(f'\nTOTAL  matched {grand["matched"]}  missing {grand["missing"]}  '
          f'extra {grand["extra"]}  role {grand["role"]}  loose {grand["loose"]}')
    print('Report only -- nothing was modified. The reference has UNCONFIRMED')
    print('provenance; the annotation is the authority.')


if __name__ == '__main__':
    main()
