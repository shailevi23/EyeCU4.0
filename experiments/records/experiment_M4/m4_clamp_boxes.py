#!/usr/bin/env python
"""
M4 Section 8 -- mechanical, content-blind correction of boxes that slightly
overshoot the frame boundary (mouse-drag overshoot at the image edge in the
local annotator). This clamps coordinates to [0, width] x [0, height] using
each sequence's known, fixed frame dimensions -- a pure geometric operation
that requires no image content and makes no visual judgement about what the
box should contain. It is not a re-annotation: no box is added, removed, or
moved except to pull an out-of-frame edge back to the frame boundary.
"""
import json
from pathlib import Path

DIMS = {'como_2-0_sassuolo': (640, 360),
        'manchester_city_v_liverpool': (640, 360),
        'youth_2': (1920, 1080)}

DRAFT_PATH = Path('experiments/records/experiment_M4/ANNOTATIONS_DRAFT.json')


def main():
    draft = json.loads(DRAFT_PATH.read_text(encoding='utf-8'))
    n_clamped = 0
    log = []
    for r in draft:
        w, h = DIMS[r['sequence']]
        for o in r.get('objects', []):
            x1, y1, x2, y2 = o['bbox']
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
                log.append({'sequence': r['sequence'], 'frame': r['frame_number_1based'],
                            'class': o['class'], 'before': [x1, y1, x2, y2],
                            'after': [cx1, cy1, cx2, cy2]})
                o['bbox'] = [cx1, cy1, cx2, cy2]
                n_clamped += 1

    DRAFT_PATH.write_text(json.dumps(draft, indent=1), encoding='utf-8')
    out = Path('experiments/records/experiment_M4/CLAMP_LOG.json')
    out.write_text(json.dumps(log, indent=1), encoding='utf-8')
    print(f'clamped {n_clamped} boxes to frame bounds')
    for e in log:
        print(f"  {e['sequence']} frame {e['frame']} [{e['class']}]: {e['before']} -> {e['after']}")
    print('written:', out)


if __name__ == '__main__':
    main()
