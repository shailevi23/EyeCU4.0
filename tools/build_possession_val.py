#!/usr/bin/env python
"""
P0 -- build and freeze the first possession benchmark, POSSESSION_VAL_V1.

Selects 6 contiguous windows of 10 frames (60 evaluated frames, the hard cap)
from the existing EyeCU-Tracking-Val-v1.1 sequences, which already carry VERIFIED
human identity GT. Selection is deterministic and content-neutral: fixed
fractional start positions inside each eligible sequence. It does NOT look at
PlayerBallAssigner output, at possession correctness, or at any ball detection.

Then renders annotation contact sheets carrying ONLY frame index, GT human boxes
and GT track ids. The model's possession prediction is never drawn, so the
annotator is blind to it.

Two stages, deliberately separate:
    --stage select    freeze the window/frame list and hash it, render sheets
    --stage finalise  attach the annotator's labels to the frozen list

No training, no tuning, no detector or selector involvement.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

SEQ_DIR = Path('data/tracking_val_gt/sequences')
MOT_DIR = Path('data/tracking_val_gt/mot/EyeCU-val')
OUT = Path('data/possession_val_v1')

# Deterministic, content-neutral, declared before selection. Same shape as the
# rule that built the frozen temporal benchmark.
START_FRACTIONS = (0.25, 0.65)
WINDOW_FRAMES = 10
LABELS = ('PLAYER', 'NO_CONTROL', 'NO_BALL', 'AMBIGUOUS')


def load_gt(seq):
    """MOT gt.txt -> {frame: [(id, x, y, w, h), ...]}. 1-based frame numbering."""
    p = MOT_DIR / seq / 'gt' / 'gt.txt'
    out = defaultdict(list)
    if not p.exists():
        return out
    for line in p.read_text(encoding='utf-8').splitlines():
        q = line.split(',')
        if len(q) < 6:
            continue
        out[int(q[0])].append((int(q[1]), float(q[2]), float(q[3]),
                               float(q[4]), float(q[5])))
    return out


def eligible():
    seqs = []
    for d in sorted(SEQ_DIR.iterdir()):
        if not (d / 'img1').is_dir():
            continue
        gt = load_gt(d.name)
        if not gt:
            continue          # no identity GT -> not eligible
        seqs.append((d.name, sorted((d / 'img1').glob('*.jpg')), gt))
    return seqs


def select():
    windows = []
    for seq, imgs, gt in eligible():
        n = len(imgs)
        for wi, frac in enumerate(START_FRACTIONS):
            start = int(round(frac * n))
            start = max(1, min(start, n - WINDOW_FRAMES + 1))
            frames = list(range(start, start + WINDOW_FRAMES))
            windows.append({
                'window_id': f'{seq}_w{wi}',
                'sequence': seq,
                'start_frame_1based': start,
                'frames_1based': frames,
                'frame_files': [imgs[f - 1].name for f in frames],
                'gt_boxes_per_frame': {str(f): len(gt.get(f, [])) for f in frames},
            })
    return windows


def render_sheets(windows, out_dir, per_sheet=5, scale=2):
    """Contact sheets: frame index + GT human boxes + GT ids. No predictions."""
    import cv2
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for w in windows:
        gt = load_gt(w['sequence'])
        imgs = SEQ_DIR / w['sequence'] / 'img1'
        chunks = [w['frames_1based'][i:i + per_sheet]
                  for i in range(0, len(w['frames_1based']), per_sheet)]
        for ci, chunk in enumerate(chunks):
            tiles = []
            for f in chunk:
                im = cv2.imdecode(np.fromfile(str(imgs / f'{f:06d}.jpg'),
                                              dtype=np.uint8), cv2.IMREAD_COLOR)
                im = cv2.resize(im, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)
                for tid, x, y, bw, bh in gt.get(f, []):
                    p1 = (int(x * scale), int(y * scale))
                    p2 = (int((x + bw) * scale), int((y + bh) * scale))
                    cv2.rectangle(im, p1, p2, (0, 255, 0), 1)
                    cv2.putText(im, str(tid), (p1[0], max(12, p1[1] - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.rectangle(im, (0, 0), (150, 26), (0, 0, 0), -1)
                cv2.putText(im, f'f{f}', (5, 19), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)
                tiles.append(im)
            sheet = np.hstack(tiles)
            name = f"{w['window_id']}_sheet{ci}.jpg"
            ok, buf = cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out_dir / name).write_bytes(buf.tobytes())
            made.append(name)
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['select', 'finalise'], required=True)
    ap.add_argument('--labels', help='finalise: JSON {frame_key: {...}}')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage == 'select':
        windows = select()
        n = sum(len(w['frames_1based']) for w in windows)
        assert n <= 60, f'cap exceeded: {n}'
        frozen = {
            'benchmark': 'POSSESSION_VAL_V1',
            'created_before_any_possession_tuning': True,
            'source': 'EyeCU-Tracking-Val-v1.1 sequences (identity GT VERIFIED)',
            'eligibility_rule': ('sequence has img1 frames AND non-empty MOT '
                                 'identity GT; austin_fc_vs__club_tijuana_284 is '
                                 'excluded because its gt.txt has 0 rows'),
            'selection_rule': {
                'deterministic': True,
                'start_fractions_of_sequence': list(START_FRACTIONS),
                'window_frames': WINDOW_FRAMES,
                'stride': 1,
                'seed': 'none required -- rule is closed-form, not sampled',
                'note': ('Content-neutral. Windows were NOT chosen by looking at '
                         'PlayerBallAssigner output, possession correctness, or '
                         'any ball detection.')},
            'label_states': list(LABELS),
            'n_windows': len(windows), 'n_frames': n,
            'windows': windows,
        }
        p = OUT / 'POSSESSION_VAL_V1_FROZEN.json'
        p.write_text(json.dumps(frozen, indent=1), encoding='utf-8')
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        (OUT / 'POSSESSION_VAL_V1_FROZEN.sha256').write_text(sha + '\n',
                                                             encoding='utf-8')
        sheets = render_sheets(windows, OUT / 'sheets')
        print(f'frozen {n} frames in {len(windows)} windows -> {p}')
        print(f'sha256 {sha}')
        print(f'sheets {len(sheets)}')
        for w in windows:
            print(f"  {w['window_id']:<34} frames "
                  f"{w['frames_1based'][0]}-{w['frames_1based'][-1]}")
        return 0

    # finalise
    frozen = json.loads((OUT / 'POSSESSION_VAL_V1_FROZEN.json').read_text(encoding='utf-8'))
    labels = json.loads(Path(args.labels).read_text(encoding='utf-8'))
    rows = []
    for w in frozen['windows']:
        for f in w['frames_1based']:
            key = f"{w['window_id']}:{f}"
            lab = labels.get(key)
            if lab is None:
                # Population is frozen at 60; the ANNOTATED SUBSET may be
                # smaller. Unlabelled frames are recorded as such rather than
                # dropped, so the gap is visible instead of implied.
                rows.append({'frame_id': key, 'window_id': w['window_id'],
                             'sequence': w['sequence'], 'frame_1based': f,
                             'label_state': 'UNLABELLED',
                             'gt_player_track_id': None, 'notes': ''})
                continue
            if lab['label_state'] not in LABELS:
                raise SystemExit(f'bad label state for {key}')
            rows.append({'frame_id': key, 'window_id': w['window_id'],
                         'sequence': w['sequence'], 'frame_1based': f,
                         'label_state': lab['label_state'],
                         'gt_player_track_id': lab.get('gt_player_track_id'),
                         'notes': lab.get('notes', '')})
    ann = {'benchmark': 'POSSESSION_VAL_V1',
           'frozen_list_sha256': (OUT / 'POSSESSION_VAL_V1_FROZEN.sha256')
           .read_text(encoding='utf-8').strip(),
           'n': len(rows),
           'distribution': {s: sum(1 for r in rows if r['label_state'] == s)
                            for s in list(LABELS) + ['UNLABELLED']},
           'annotated_subset_n': sum(1 for r in rows if r['label_state'] != 'UNLABELLED'),
           'annotations': rows}
    p = OUT / 'POSSESSION_VAL_V1_ANNOTATIONS.json'
    p.write_text(json.dumps(ann, indent=1), encoding='utf-8')
    print(f'annotations -> {p}')
    print('distribution', ann['distribution'])
    print('sha256', hashlib.sha256(p.read_bytes()).hexdigest())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
