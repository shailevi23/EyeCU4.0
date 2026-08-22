#!/usr/bin/env python
"""
M4 Section 8 -- computational (non-visual) QC of the merged 60-frame
TEST_DETECTION_ANNOTATIONS set. Machine checks only: no image is opened, no
model runs, no metric/accuracy claim is computed against any prediction.
"""
import json
from pathlib import Path

DIMS = {'como_2-0_sassuolo': (640, 360),
        'manchester_city_v_liverpool': (640, 360),
        'youth_2': (1920, 1080)}
CLASSES = {'player', 'goalkeeper', 'referee', 'ball'}

FINAL_LIST = json.loads(Path('experiments/records/experiment_M4/FINAL_TEST_FRAME_LIST.json')
                        .read_text(encoding='utf-8'))
DRAFT = json.loads(Path('experiments/records/experiment_M4/ANNOTATIONS_DRAFT.json')
                   .read_text(encoding='utf-8'))


def main():
    errors = []
    warnings = []

    expected = set()
    for seq, info in FINAL_LIST['sequences'].items():
        for n in info['selected_frame_numbers_1based']:
            expected.add((seq, n))

    got = [(r['sequence'], r['frame_number_1based']) for r in DRAFT]
    got_set = set(got)

    if len(DRAFT) != 60:
        errors.append(f'expected exactly 60 frame records, found {len(DRAFT)}')
    if len(got) != len(got_set):
        dupes = [k for k in got_set if got.count(k) > 1]
        errors.append(f'duplicate (sequence, frame_number_1based) records: {dupes}')
    missing = expected - got_set
    extra = got_set - expected
    if missing:
        errors.append(f'{len(missing)} frames from FINAL_TEST_FRAME_LIST.json missing from annotations: {sorted(missing)}')
    if extra:
        errors.append(f'{len(extra)} annotated frames not in FINAL_TEST_FRAME_LIST.json (frame list violated): {sorted(extra)}')

    total_objects = 0
    per_class = {c: 0 for c in CLASSES}
    per_frame_counts = []
    multi_ball_frames = []

    for r in DRAFT:
        seq = r['sequence']
        w, h = DIMS[seq]
        objs = r.get('objects', [])
        per_frame_counts.append((seq, r['frame_number_1based'], len(objs)))
        seen_boxes = set()
        ball_n = 0
        for i, o in enumerate(objs):
            cls = o.get('class')
            bbox = o.get('bbox')
            tag = f'{seq} frame {r["frame_number_1based"]} object {i}'
            if cls not in CLASSES:
                errors.append(f'{tag}: invalid class {cls!r}')
                continue
            if cls == 'ball':
                ball_n += 1
            if not (isinstance(bbox, list) and len(bbox) == 4):
                errors.append(f'{tag}: malformed bbox {bbox!r}')
                continue
            x1, y1, x2, y2 = bbox
            if not (x2 > x1 and y2 > y1):
                errors.append(f'{tag}: non-positive-area box {bbox}')
            if not (0 <= x1 and 0 <= y1 and x2 <= w and y2 <= h):
                errors.append(f'{tag}: box {bbox} out of bounds for {w}x{h} frame')
            key = (cls, round(x1), round(y1), round(x2), round(y2))
            if key in seen_boxes:
                errors.append(f'{tag}: exact-duplicate box {bbox} (class {cls})')
            seen_boxes.add(key)
            per_class[cls] += 1
            total_objects += 1
        if ball_n > 1:
            multi_ball_frames.append((seq, r['frame_number_1based'], ball_n))
        if not r.get('notes', '').strip() and len(objs) == 0:
            warnings.append(f'{seq} frame {r["frame_number_1based"]}: zero objects and no note explaining it')

    report = {
        'n_frame_records': len(DRAFT),
        'n_expected_frames': len(expected),
        'population_matches_final_list': not missing and not extra and len(DRAFT) == 60,
        'no_duplicate_frame_records': len(got) == len(got_set),
        'total_objects': total_objects,
        'per_class_counts': per_class,
        'multi_ball_frames': multi_ball_frames,
        'n_errors': len(errors),
        'errors': errors,
        'n_warnings': len(warnings),
        'warnings': warnings,
        'qc_pass': len(errors) == 0,
    }
    out = Path('experiments/records/experiment_M4/QC_RESULT.json')
    out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('errors', 'warnings')}, indent=1))
    if errors:
        print('\nERRORS:')
        for e in errors:
            print(' -', e)
    if warnings:
        print('\nWARNINGS:')
        for w in warnings:
            print(' -', w)
    print('\nwritten:', out)


if __name__ == '__main__':
    main()
