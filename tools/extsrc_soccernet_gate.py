#!/usr/bin/env python
"""
SoccerNet-V3 (Voxel51) metadata gate: decide on 7 GB of images from 24 MB of JSON.

The gate exists so the decision is made on evidence rather than on the project's
reputation. SoccerNet as a whole is a large, well-known corpus; THIS Hugging Face
export is one FiftyOne snapshot of part of it, and the two are not the same thing.
Everything below is read out of the actual samples.json.

FiftyOne stores detections in RELATIVE coordinates -- bounding_box is
[x, y, w, h] as fractions of the image. Turning that into a pixel width needs the
image's real width, so metadata.width is used where present and the result is
labelled as what it is. Where the width is absent, the instance is counted but
excluded from pixel statistics rather than assigned a guessed dimension.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / 'EyeCU_external_data'
SRC = EXT / 'huggingface' / 'soccernet_v3'

BINS = [(0, 3, '<3'), (3, 5, '3-5'), (5, 8, '>5-8'), (8, 12, '>8-12'),
        (12, 20, '>12-20'), (20, 40, '>20-40'), (40, 1e9, '>40')]


def bin_of(w):
    for lo, hi, n in BINS:
        if lo == 0 and w < hi:
            return n
        if lo < w <= hi:
            return n
    return '>40'


def stats(a, nd=2):
    """nd=2 for pixels; fractions need more, or a real distribution looks quantised."""
    a = np.asarray(a, float)
    return {'n': int(a.size), 'median': round(float(np.median(a)), nd),
            'mean': round(float(a.mean()), nd),
            'p10': round(float(np.percentile(a, 10)), nd),
            'p25': round(float(np.percentile(a, 25)), nd),
            'p75': round(float(np.percentile(a, 75)), nd),
            'p90': round(float(np.percentile(a, 90)), nd),
            'min': round(float(a.min()), nd), 'max': round(float(a.max()), nd)}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    d = json.loads((SRC / 'metadata_only' / 'samples.json').read_text(encoding='utf-8'))
    samples = d['samples'] if isinstance(d, dict) else d
    print(f'{len(samples)} samples')

    label_fields = Counter()
    classes = Counter()
    per_class_boxes = defaultdict(list)
    dims = Counter()
    slices = Counter()
    games = Counter()
    no_dim = 0
    images_with_ball = set()
    sample_keys = Counter()

    for s in samples:
        for k in s:
            sample_keys[k] += 1
        slices[s.get('group', {}).get('name') if isinstance(s.get('group'), dict)
               else s.get('group')] += 1
        md = s.get('metadata') or {}
        W, H = md.get('width'), md.get('height')
        if W and H:
            dims[f'{W}x{H}'] += 1
        else:
            no_dim += 1
        fp = s.get('filepath', '')
        # data/data_44/2_0-52.png -> data_44 is the source group
        parts = [p for p in fp.replace('\\', '/').split('/') if p]
        if len(parts) >= 2:
            games[parts[-2]] += 1
        for field, val in s.items():
            if not isinstance(val, dict):
                continue
            dets = val.get('detections')
            if not isinstance(dets, list):
                continue
            label_fields[field] += 1
            for det in dets:
                lbl = det.get('label')
                classes[lbl] += 1
                bb = det.get('bounding_box')
                if bb and W and H:
                    per_class_boxes[lbl].append((bb[2] * W, bb[3] * H, W, H, fp))
                if lbl and 'ball' in str(lbl).lower():
                    images_with_ball.add(fp)

    print(f'\nsample fields: {dict(sample_keys.most_common(12))}')
    print(f'label fields carrying detections: {dict(label_fields)}')
    print(f'group slices: {dict(slices.most_common(12))}')
    print(f'image dimensions: {dict(dims.most_common(8))}  (no dimension: {no_dim})')
    print(f'source groups (dirs): {len(games)}')

    print(f'\nclasses ({len(classes)}):')
    for c, n in classes.most_common(40):
        print(f'   {str(c):<28} {n:>8}')

    report = {
        'export': 'Voxel51/SoccerNet-V3 (FiftyOne snapshot)',
        'samples': len(samples),
        'group_slices': dict(slices),
        'image_dimensions': dict(dims),
        'samples_without_dimensions': no_dim,
        'source_directories': len(games),
        'label_fields': dict(label_fields),
        'classes': dict(classes.most_common()),
        'total_annotations': sum(classes.values()),
    }

    ball_labels = [c for c in classes if c and 'ball' in str(c).lower()]
    report['ball_labels_present'] = ball_labels
    report['ball_present'] = bool(ball_labels)
    report['ball_annotations'] = sum(classes[c] for c in ball_labels)
    report['images_containing_ball'] = len(images_with_ball)

    if ball_labels:
        rows = [r for c in ball_labels for r in per_class_boxes[c]]
        if rows:
            w = np.array([r[0] for r in rows])
            h = np.array([r[1] for r in rows])
            report['ball_box_pixels'] = {
                'measured_on': 'stored image dimensions from FiftyOne metadata',
                'width': stats(w), 'height': stats(h),
                'width_bins': {b[2]: int(sum(bin_of(x) == b[2] for x in w))
                               for b in BINS},
                'le5': int((w <= 5).sum()), 'le8': int((w <= 8).sum()),
                'le12': int((w <= 12).sum()),
            }
            print(f'\nBALL: {report["ball_annotations"]} annotations in '
                  f'{report["images_containing_ball"]} images')
            print(f'   width px  {report["ball_box_pixels"]["width"]}')
            print(f'   bins      {report["ball_box_pixels"]["width_bins"]}')
            print(f'   <=5 {report["ball_box_pixels"]["le5"]}  '
                  f'<=8 {report["ball_box_pixels"]["le8"]}  '
                  f'<=12 {report["ball_box_pixels"]["le12"]}')
    else:
        print('\nBALL: no ball label exists in this export')

    # EyeCU-relevant classes. Written out explicitly rather than by substring:
    # 'Wall of players' contains "player" but is a GROUP box around a defensive
    # wall, not one person, and 'Referee flag' contains "referee" but is an
    # object. Both would be silently mismapped by a substring rule, and both
    # would poison a detector trained on the result.
    EXPLICIT = {
        'Player team left': 'player',
        'Player team right': 'player',
        'Goalkeeper team left': 'goalkeeper',
        'Goalkeeper team right': 'goalkeeper',
        'Main referee': 'referee',
        'Side referee': 'referee',
        'Ball': 'ball',
        'Staff members': 'EXCLUDE_not_an_eyecu_class',
        'Wall of players': 'EXCLUDE_group_box_not_one_person',
        'Referee flag': 'EXCLUDE_object_not_a_person',
        'Yellow card': 'EXCLUDE_object_not_a_person',
        'Red card': 'EXCLUDE_object_not_a_person',
    }
    eyecu_map = {c: EXPLICIT.get(c, 'UNMAPPED_REVIEW_REQUIRED') for c in classes}
    report['proposed_eyecu_mapping'] = eyecu_map
    report['unmapped_classes'] = sorted({c for c, v in eyecu_map.items()
                                         if v.startswith('UNMAPPED')})
    report['excluded_classes'] = {c: v for c, v in eyecu_map.items()
                                  if v.startswith('EXCLUDE')}
    report['annotations_by_eyecu_class'] = {
        cat: sum(classes[c] for c, v in eyecu_map.items() if v == cat)
        for cat in ('player', 'goalkeeper', 'referee', 'ball')}
    report['annotations_excluded'] = sum(
        classes[c] for c, v in eyecu_map.items() if v.startswith('EXCLUDE'))

    # Relative box size is available even though pixel dimensions are not, and
    # it is scale-invariant -- the honest thing to report before any image is
    # downloaded. Pixel sizes come later, from sampled real images.
    rel = defaultdict(list)
    jersey = Counter()
    for s in samples:
        for field, val in s.items():
            if isinstance(val, dict) and isinstance(val.get('detections'), list):
                for det in val['detections']:
                    bb = det.get('bounding_box')
                    if bb:
                        rel[eyecu_map.get(det.get('label'), '?')].append((bb[2], bb[3]))
                    j = det.get('jersey_number')
                    if j is not None:
                        jersey['present'] += 1
                        jersey['readable' if float(j) >= 0 else 'unset(-1)'] += 1
    report['relative_box_size_by_eyecu_class'] = {
        k: {'n': len(v),
            'width_fraction': stats([x[0] for x in v], 6),
            'height_fraction': stats([x[1] for x in v], 6)}
        for k, v in rel.items() if k in ('player', 'goalkeeper', 'referee', 'ball')}
    report['jersey_number_field'] = dict(jersey)
    print('\nrelative box size (fraction of image width), from metadata alone:')
    for k, v in report['relative_box_size_by_eyecu_class'].items():
        print(f"   {k:<11} n={v['n']:>6}  w median {v['width_fraction']['median']:.6f}  "
              f"p10 {v['width_fraction']['p10']:.6f}  p90 {v['width_fraction']['p90']:.6f}")
    print(f'   jersey_number field: {dict(jersey)}')
    report['goalkeeper_distinct'] = any(v == 'goalkeeper' for v in eyecu_map.values())
    report['referee_distinct'] = any(v == 'referee' for v in eyecu_map.values())
    for cat in ('player', 'goalkeeper', 'referee', 'ball'):
        rows = [r for c, v in eyecu_map.items() if v == cat for r in per_class_boxes[c]]
        if rows:
            report.setdefault('box_pixels_by_eyecu_class', {})[cat] = {
                'n': len(rows),
                'width': stats([r[0] for r in rows]),
                'height': stats([r[1] for r in rows])}

    (SRC / 'manifests').mkdir(parents=True, exist_ok=True)
    (SRC / 'manifests' / 'metadata_gate.json').write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'\nwrote soccernet_v3/manifests/metadata_gate.json')


if __name__ == '__main__':
    main()
