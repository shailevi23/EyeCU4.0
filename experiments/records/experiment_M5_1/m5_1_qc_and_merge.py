#!/usr/bin/env python
"""
M5.1 -- computational QC of the 20 blind Como re-annotations, then merge
with the unchanged manchester_city_v_liverpool/youth_2 GT (from the
original, frozen TEST_DETECTION_ANNOTATIONS.json) into a NEW corrected-GT
artifact. Original M4/M5 files are read-only here, never modified.
"""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
M4 = REPO / 'experiments/records/experiment_M4'
M5_1 = REPO / 'experiments/records/experiment_M5_1'

EXPECTED_FRAMES = [38, 113, 188, 263, 338, 413, 488, 563, 638, 713,
                  789, 864, 939, 1014, 1089, 1164, 1239, 1314, 1389, 1464]
DIMS = (640, 360)
CLASSES = {'player', 'goalkeeper', 'referee', 'ball'}


def qc_como(records):
    errors = []
    got_frames = sorted(r['frame_number_1based'] for r in records)
    if got_frames != sorted(EXPECTED_FRAMES):
        errors.append(f'frame set mismatch: expected {sorted(EXPECTED_FRAMES)}, got {got_frames}')
    seen = set()
    for r in records:
        if r['sequence'] != 'como_2-0_sassuolo':
            errors.append(f'unexpected sequence {r["sequence"]!r} in blind annotations')
        n = r['frame_number_1based']
        if n in seen:
            errors.append(f'duplicate frame record {n}')
        seen.add(n)
        w, h = DIMS
        seen_boxes = set()
        for i, o in enumerate(r.get('objects', [])):
            tag = f'frame {n} object {i}'
            cls = o.get('class')
            bbox = o.get('bbox')
            if cls not in CLASSES:
                errors.append(f'{tag}: invalid class {cls!r}')
                continue
            if not (isinstance(bbox, list) and len(bbox) == 4):
                errors.append(f'{tag}: malformed bbox {bbox!r}')
                continue
            x1, y1, x2, y2 = bbox
            if not (x2 > x1 and y2 > y1):
                errors.append(f'{tag}: non-positive-area box {bbox}')
            if not (0 <= x1 and 0 <= y1 and x2 <= w and y2 <= h):
                errors.append(f'{tag}: out of bounds for {w}x{h}: {bbox}')
            key = (cls, round(x1), round(y1), round(x2), round(y2))
            if key in seen_boxes:
                errors.append(f'{tag}: exact-duplicate box {bbox} (class {cls})')
            seen_boxes.add(key)
    return errors


def main():
    blind_path = M5_1 / 'local_annotator' / 'COMO_BLIND_ANNOTATIONS.json'
    blind = json.loads(blind_path.read_text(encoding='utf-8'))

    errors = qc_como(blind)
    n_objects = sum(len(r['objects']) for r in blind)
    per_class = {c: 0 for c in CLASSES}
    for r in blind:
        for o in r['objects']:
            if o['class'] in per_class:
                per_class[o['class']] += 1

    qc_result = {
        'n_frame_records': len(blind), 'n_expected': len(EXPECTED_FRAMES),
        'n_objects': n_objects, 'per_class_counts': per_class,
        'n_errors': len(errors), 'errors': errors, 'qc_pass': len(errors) == 0,
    }
    (M5_1 / 'M5_1_COMO_QC_RESULT.json').write_text(json.dumps(qc_result, indent=1), encoding='utf-8')
    print('Como blind-annotation QC:', json.dumps({k: v for k, v in qc_result.items() if k != 'errors'}, indent=1))
    if errors:
        print('ERRORS:')
        for e in errors:
            print(' -', e)
        return 1

    # ---- merge into corrected GT ----
    original = json.loads((M4 / 'TEST_DETECTION_ANNOTATIONS.json').read_text(encoding='utf-8'))
    unchanged = [r for r in original if r['sequence'] != 'como_2-0_sassuolo']
    assert len(unchanged) == 40, f'expected 40 non-Como records unchanged, got {len(unchanged)}'

    corrected_como = []
    for r in sorted(blind, key=lambda r: r['frame_number_1based']):
        fname = f"como_2-0_sassuolo_{r['frame_number_1based']:06d}.jpg"
        corrected_como.append({
            'sequence': 'como_2-0_sassuolo',
            'frame_number_1based': r['frame_number_1based'],
            'file': f'experiments/records/experiment_M4/candidates/como_2-0_sassuolo/{fname}',
            'objects': r['objects'],
            'notes': 'blind re-annotation, M5.1 -- local browser tool, no old GT/predictions/scores shown to the annotator',
        })

    corrected_gt = sorted(unchanged + corrected_como, key=lambda r: (r['sequence'], r['frame_number_1based']))
    assert len(corrected_gt) == 60

    out = M5_1 / 'TEST_DETECTION_ANNOTATIONS_CORRECTED.json'
    out.write_text(json.dumps(corrected_gt, indent=1), encoding='utf-8')
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (M5_1 / 'TEST_DETECTION_ANNOTATIONS_CORRECTED.sha256').write_text(sha + '\n', encoding='utf-8')

    print(f'\nmerged corrected GT: {len(corrected_gt)} frames '
         f'(40 unchanged manchester/youth_2 + 20 blind-reannotated como)')
    print('sha256:', sha)
    print('written:', out)

    orig_sha = hashlib.sha256((M4 / 'TEST_DETECTION_ANNOTATIONS.json').read_bytes()).hexdigest()
    print('\noriginal M4 TEST_DETECTION_ANNOTATIONS.json sha256 (unchanged, re-verified):', orig_sha)
    assert orig_sha == '4702b60fbdb173773e6bc7246d45587c1446093559c96792d3ae864e4d6896cb', 'original GT changed!'
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
