#!/usr/bin/env python
"""
Extract the detector labels that already cover tracking frames.

PROVENANCE IS UNCONFIRMED. These labels began as output of one Roboflow model
and were later modified -- boxes added, classes changed -- by something this
repository does not record. The label pipeline has no reviewed/approved flag,
only `status: draft`. The user states they have never annotated in CVAT, so the
review path, if any, is unknown.

They are therefore REFERENCE GEOMETRY, not ground truth. They may be consulted
during annotation and used afterwards as a cross-check, and they must not be
adopted as tracking GT unless someone can say how they were reviewed.

This tool finds them and writes them to `human_seed/<seq>.json`. It is separate
from the frozen detector preannotations, which are pure model output and stay
untouched: mixing the two would destroy the distinction the whole benchmark
rests on.

Overlap is established from source-frame provenance -- data/manifest.json
`frame_index` against the tracking manifest `source_frame_range` -- and then
CONFIRMED pixel-wise against the packaged frame. Filename agreement is not
proof: a seek that landed on the wrong frame would still produce matching
names. Each seed frame records the mean absolute difference to the packaged
frame and to its neighbours, so the alignment claim can be re-checked later
without rerunning anything.

The seed carries geometry and class only. It carries NO identity -- identity is
the new work, and no detector-era artifact can supply it.

The file name says `human_seed` for continuity with existing paths; the
`provenance_status` field inside each file is what governs how it may be used.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackers.detector import HUMAN_CLASSES  # noqa: E402

# label index order, as written by tools/import_roboflow.py
CLASSES = ['player', 'goalkeeper', 'referee', 'ball']
NEIGHBOUR_OFFSETS = (-1, 1)
ALIGN_MARGIN = 2.0      # neighbour MAD must exceed offset-0 MAD by this factor


def imread(p: Path):
    """cv2.imread cannot open non-ASCII paths on Windows; this repo has one."""
    if not p.exists():
        return None
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def mad(a, b):
    if a is None or b is None or a.shape != b.shape:
        return float('nan')
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def frame_alignment(ds_img, seq_dir: Path, pf: int):
    """(mad at pf, worst-case neighbour ratio). Ratio <= 1 means misalignment."""
    m0 = mad(ds_img, imread(seq_dir / f'{pf:06d}.jpg'))
    ratios = []
    for off in NEIGHBOUR_OFFSETS:
        mn = mad(ds_img, imread(seq_dir / f'{pf + off:06d}.jpg'))
        if not np.isnan(mn) and m0 > 0:
            ratios.append(mn / m0)
    return m0, (min(ratios) if ratios else float('nan'))


def yolo_to_xyxy(cx, cy, w, h, W, H):
    return [round((cx - w / 2) * W, 2), round((cy - h / 2) * H, 2),
            round((cx + w / 2) * W, 2), round((cy + h / 2) * H, 2)]


def build(root: Path, data: Path, dry_run=False):
    man = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    tman = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))

    by_src = {}
    for im in man['images']:
        by_src.setdefault(im['source'], {})[im['frame_index']] = im['image_path']

    summary = []
    for s in tman['sequences']:
        seq, lo, hi = s['sequence'], *s['source_frame_range']
        W, H = s['frame_width'], s['frame_height']
        seq_dir = root / 'sequences' / seq / 'img1'
        got = by_src.get(s['match'], {})

        frames, counts, rejected = [], Counter(), []
        for f in sorted(x for x in got if lo <= x <= hi):
            rel = got[f]
            pf = f - lo + 1
            ds_img = imread(data / 'frames' / rel)
            m0, ratio = frame_alignment(ds_img, seq_dir, pf)
            if not (ratio > ALIGN_MARGIN):
                rejected.append((f, pf, m0, ratio))
                continue

            lab = data / 'labels' / rel.replace('.jpg', '.txt')
            meta = data / 'pseudo_meta' / rel.replace('.jpg', '.json')
            boxes = []
            for line in lab.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                c, cx, cy, w, h = line.split()[:5]
                name = CLASSES[int(c)]
                if name not in HUMAN_CLASSES:
                    counts['ball_excluded'] += 1
                    continue
                boxes.append({'bbox': yolo_to_xyxy(float(cx), float(cy), float(w),
                                                   float(h), W, H),
                              'role': name})
                counts[name] += 1

            frames.append({
                'package_frame': pf,
                'source_frame': f,
                'detector_label_file': lab.as_posix(),
                'detector_label_sha256':
                    hashlib.sha256(lab.read_bytes()).hexdigest(),
                'model_draft_record': meta.as_posix() if meta.exists() else None,
                'alignment': {'mad_to_packaged_frame': round(m0, 3),
                              'nearest_neighbour_ratio': round(ratio, 2)},
                'boxes': boxes,
            })

        out = {
            'sequence': seq,
            'purpose': 'reference geometry for frames already labelled during '
                       'detector annotation',
            'provenance_status': 'UNCONFIRMED',
            'authoritative': False,
            'usage': 'Consult during annotation and cross-check afterwards. '
                     'Do NOT adopt as tracking GT.',
            'contains_identity': False,
            'identity_note': 'Identity is NOT present and cannot be derived '
                             'from this file. It remains manual work.',
            'geometry': f'absolute pixels [x1,y1,x2,y2] on {W}x{H} package frames',
            'class_mapping': {'reused': sorted(HUMAN_CLASSES),
                              'dropped': ['ball']},
            'provenance': {
                'source': 'data/labels',
                'known': 'drafted by the Roboflow model recorded in '
                         'model_draft_record, then modified by an unrecorded '
                         'process (boxes added, classes changed)',
                'unknown': 'whether, by whom, and to what standard those '
                           'modifications were reviewed; the label pipeline '
                           'carries no reviewed flag, only status=draft',
                'overlap_established_by': 'source frame provenance '
                                          '(data/manifest.json frame_index)',
                'overlap_confirmed_by': 'pixel comparison against the packaged '
                                        'frame and its neighbours',
                'alignment_margin_required': ALIGN_MARGIN,
                'is_not': 'the frozen detector preannotation, which is model '
                          'output and is left untouched',
            },
            'frames': frames,
        }
        summary.append((seq, len(frames), counts, rejected))
        if not dry_run:
            (root / 'human_seed').mkdir(exist_ok=True)
            (root / 'human_seed' / f'{seq}.json').write_text(
                json.dumps(out, indent=1), encoding='utf-8')

    print(f'{"sequence":<32}{"frames":>7}{"player":>8}{"gk":>5}{"ref":>5}'
          f'{"ball drop":>11}{"rejected":>10}')
    tot = Counter()
    for seq, n, c, rej in summary:
        tot.update(c)
        tot['frames'] += n
        tot['rejected'] += len(rej)
        print(f'  {seq[:30]:<32}{n:>7}{c["player"]:>8}{c["goalkeeper"]:>5}'
              f'{c["referee"]:>5}{c["ball_excluded"]:>11}{len(rej):>10}')
        for f, pf, m0, ratio in rej:
            print(f'      REJECTED src {f} -> package {pf}: mad {m0:.2f}, '
                  f'neighbour ratio {ratio:.2f} (needs > {ALIGN_MARGIN})')
    print(f'  {"TOTAL":<32}{tot["frames"]:>7}{tot["player"]:>8}'
          f'{tot["goalkeeper"]:>5}{tot["referee"]:>5}{tot["ball_excluded"]:>11}'
          f'{tot["rejected"]:>10}')
    reusable = tot['player'] + tot['goalkeeper'] + tot['referee']
    print(f'\n{reusable} reference boxes on {tot["frames"]} of 1200 frames.')
    print('PROVENANCE UNCONFIRMED: reference and cross-check only, not GT.')
    print('Identity for all 1200 frames remains manual work.')
    if dry_run:
        print('\n(dry run -- nothing written)')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/tracking_val_gt')
    ap.add_argument('--data', default='data')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    build(Path(args.root), Path(args.data), args.dry_run)


if __name__ == '__main__':
    main()
