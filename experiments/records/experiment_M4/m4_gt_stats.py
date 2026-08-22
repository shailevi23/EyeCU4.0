#!/usr/bin/env python
"""
M4 Section 9 -- GT-only descriptive statistics. No model, no prediction, no
comparison of any kind -- purely a description of what was annotated. Must
NOT be used to infer or modify any model/threshold/subset/ontology decision.
"""
import json
import statistics
from pathlib import Path

DRAFT = json.loads(Path('experiments/records/experiment_M4/ANNOTATIONS_DRAFT.json')
                   .read_text(encoding='utf-8'))
CLASSES = ['player', 'goalkeeper', 'referee', 'ball']


def main():
    per_match = {}
    per_class_total = {c: 0 for c in CLASSES}
    objects_per_frame = []
    ball_counts_per_frame = []
    frames_with_zero_objects = []
    frames_missing_gk = 0
    frames_missing_referee = 0
    frames_missing_ball = 0

    for r in DRAFT:
        seq = r['sequence']
        m = per_match.setdefault(seq, {'n_frames': 0, 'n_objects': 0,
                                       **{c: 0 for c in CLASSES}})
        m['n_frames'] += 1
        objs = r.get('objects', [])
        objects_per_frame.append(len(objs))
        if not objs:
            frames_with_zero_objects.append(f"{seq}#{r['frame_number_1based']}")
        counts = {c: 0 for c in CLASSES}
        for o in objs:
            counts[o['class']] += 1
            per_class_total[o['class']] += 1
            m[o['class']] += 1
            m['n_objects'] += 1
        ball_counts_per_frame.append(counts['ball'])
        if counts['goalkeeper'] == 0:
            frames_missing_gk += 1
        if counts['referee'] == 0:
            frames_missing_referee += 1
        if counts['ball'] == 0:
            frames_missing_ball += 1

    stats = {
        'n_frames': len(DRAFT),
        'n_objects_total': sum(per_class_total.values()),
        'per_class_total': per_class_total,
        'per_match': per_match,
        'objects_per_frame': {
            'min': min(objects_per_frame), 'max': max(objects_per_frame),
            'mean': round(statistics.mean(objects_per_frame), 2),
            'median': statistics.median(objects_per_frame),
        },
        'frames_with_zero_objects': frames_with_zero_objects,
        'frames_with_zero_objects_count': len(frames_with_zero_objects),
        'frames_missing_goalkeeper': frames_missing_gk,
        'frames_missing_referee': frames_missing_referee,
        'frames_missing_ball': frames_missing_ball,
        'multi_ball_frame_count': sum(1 for c in ball_counts_per_frame if c > 1),
        'ball_count_distribution': {
            str(k): ball_counts_per_frame.count(k) for k in sorted(set(ball_counts_per_frame))
        },
        'note': 'descriptive only -- GT composition, not compared against any model prediction; not used to alter any threshold, subset, or ontology decision',
    }
    out = Path('experiments/records/experiment_M4/GT_DESCRIPTIVE_STATS.json')
    out.write_text(json.dumps(stats, indent=1), encoding='utf-8')
    print(json.dumps(stats, indent=1))
    print('\nwritten:', out)


if __name__ == '__main__':
    main()
