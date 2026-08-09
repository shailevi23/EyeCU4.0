#!/usr/bin/env python
"""
Convert manually verified identity GT to the MOTChallenge layout TrackEval reads.

Format verified against trackeval 1.3.0 `mot_challenge_2d_box.py`, not assumed:

    row      frame,id,x,y,w,h,conf,class,visibility
    frame    1-BASED            L215  [str(t+1) for t in range(num_timesteps)]
                                L227  time_key = str(t+1)
    columns  L240 dets = time_data[:, 2:6]  -> x,y,w,h (top-left + size)
             L241 ids  = time_data[:, 1]
             L252 classes = time_data[:, 7]
             L261 zero_marked = time_data[:, 6]
    conf     MUST be non-zero. L394 gt_to_keep_mask = not_equal(zero_marked, 0)
             drops every GT row whose conf column is 0.
    class    MUST be 1 (pedestrian). L78 maps 'pedestrian' -> 1; L365-373 raise
             on any class outside the valid set; L353-356 reject non-pedestrian.

The commonly quoted `frame,id,x,y,w,h,conf,-1,-1,-1` row is the TRACKER
prediction convention. Used for GT it makes class = -1, which TrackEval rejects.

Refuses to run unless the benchmark is VERIFIED -- imported, post-validated,
and human QC-confirmed with the confirmation still matching the artifacts on
disk. Exporting an unreviewed benchmark into a TrackEval layout is exactly how
fake results get produced.
"""

import argparse
import configparser
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GT_CONF = 1
GT_CLASS = 1          # pedestrian
GT_VISIBILITY = 1


def export(root: Path, out: Path, benchmark_split: str = 'EyeCU-val'):
    man = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    from tools.validate_tracking_gt import validate_verified
    errors, _ = validate_verified(root)
    if errors:
        sep = '\n  '
        raise SystemExit(
            'REFUSING: GT is not VERIFIED.' + sep + sep.join(errors[:6]) +
            '\nExport is only permitted for human-confirmed GT.')

    seqmap = []
    for s in man['sequences']:
        tag = s['sequence']
        ann = json.loads((root / s['annotation_file_expected']).read_text(encoding='utf-8'))
        seq_out = out / benchmark_split / tag
        (seq_out / 'gt').mkdir(parents=True, exist_ok=True)

        rows = []
        for b in sorted(ann['boxes'], key=lambda r: (r['frame'], r['id'])):
            x1, y1, x2, y2 = b['bbox']
            # package frames are already 1-based; assert rather than convert
            assert 1 <= b['frame'] <= s['frame_count'], b['frame']
            rows.append(f"{b['frame']},{b['id']},{x1:.2f},{y1:.2f},"
                        f"{x2-x1:.2f},{y2-y1:.2f},{GT_CONF},{GT_CLASS},{GT_VISIBILITY}")
        (seq_out / 'gt' / 'gt.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')
        shutil.copy2(root / 'sequences' / tag / 'seqinfo.ini', seq_out / 'seqinfo.ini')
        seqmap.append(tag)
        print(f'  {tag:<32}{len(rows):>7} rows')

    smdir = out / 'seqmaps'
    smdir.mkdir(parents=True, exist_ok=True)
    (smdir / f'{benchmark_split}.txt').write_text(
        'name\n' + '\n'.join(seqmap) + '\n', encoding='utf-8')
    print(f'\nwritten: {out}  ({len(seqmap)} sequences)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/tracking_val_gt')
    ap.add_argument('--out', default='data/tracking_val_gt/mot')
    args = ap.parse_args()
    root = Path(args.root)
    # the CLI additionally re-checks package structure; export() itself enforces
    # the identity gate (VERIFIED + intact QC record + valid content)
    from tools.validate_tracking_gt import validate_final
    errors, _ = validate_final(root)
    if errors:
        sep = '\n  '
        raise SystemExit('REFUSING: GT is not VERIFIED.' + sep +
                         sep.join(errors[:6]))
    export(root, Path(args.out))


if __name__ == '__main__':
    main()
