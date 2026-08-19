#!/usr/bin/env python
"""
M3 freeze-hygiene repair -- resolve the TEST-only thread-setting question.

The freeze defensively required torch.set_num_threads(1) / cv2.setNumThreads(1)
for TEST, not because measured nondeterminism required it (the actual CBIoU
blocker was shared mutable tracker state across sequences, already fixed).
This runs the frozen production configuration on a small non-TEST development
sample twice -- once at default thread settings, once single-threaded -- and
compares every prediction-affecting record. Timing is ignored.
"""

import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackers.detector import BALL_ACCEPT_CONF                          # noqa: E402
from trackers.football_tracker import FootballTracker                   # noqa: E402
from trackers.ball_temporal import BallTemporalSelector, FrameInput, detect_cuts  # noqa: E402

SEQ = Path('data/tracking_val_gt/sequences/bayern_munich_3-1_chelsea_228/img1')
N_FRAMES = 40


def run_once():
    frame_ids = list(range(1, N_FRAMES + 1))
    imgs = [cv2.imdecode(np.fromfile(str(SEQ / f'{f:06d}.jpg'), dtype=np.uint8),
                         cv2.IMREAD_COLOR) for f in frame_ids]

    tracker = FootballTracker(model_path='best_A_960.pt', imgsz=960,
                              confidence=BALL_ACCEPT_CONF, persist_cache=False,
                              ball_candidate_pool=True, ball_detector_backend='sn3d')
    tracker.detector.clear_cache()
    tracks = tracker.get_object_tracks(imgs)
    cands = tracker.ball_candidates

    human_records = []
    for key in ('players', 'goalkeepers', 'referees'):
        for i, frame in enumerate(tracks[key]):
            for tid, t in sorted(frame.items()):
                human_records.append({
                    'frame': i, 'key': key, 'track_id': int(tid),
                    'bbox': [round(float(v), 6) for v in t['bbox']],
                    'confidence': round(float(t['confidence']), 6),
                })

    ball_candidate_records = []
    for i, frame_cands in enumerate(cands):
        for c in frame_cands:
            ball_candidate_records.append({
                'frame': i, 'bbox': [round(float(v), 6) for v in c['bbox']],
                'confidence': round(float(c.get('confidence', 0.0)), 6),
                'state': c.get('state'),
            })

    thumbs = [cv2.cvtColor(cv2.resize(im, (64, 36)), cv2.COLOR_BGR2GRAY) for im in imgs]
    cuts = detect_cuts(thumbs)
    sel = BallTemporalSelector(frame_width=imgs[0].shape[1])
    fin = [FrameInput(candidates=[dict(c) for c in cands[i]], timestamp=i * 0.04,
                      dt=0.04, cut=cuts[i]) for i in range(len(imgs))]
    outs = sel.run(fin)
    selector_records = [{
        'frame': i, 'state': o.state,
        'bbox': None if o.bbox is None else [round(float(v), 6) for v in o.bbox],
    } for i, o in enumerate(outs)]

    return {
        'human_records': human_records,
        'ball_candidate_records': ball_candidate_records,
        'selector_records': selector_records,
    }


def main():
    import torch

    print('=== pass 1: default thread settings ===', flush=True)
    run_default = run_once()

    print('=== pass 2: single-threaded (torch.set_num_threads(1), cv2.setNumThreads(1)) ===',
         flush=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    run_single = run_once()

    diffs = {}
    for key in ('human_records', 'ball_candidate_records', 'selector_records'):
        a, b = run_default[key], run_single[key]
        diffs[key] = {
            'n_default': len(a), 'n_single_thread': len(b),
            'record_identical': a == b,
        }
        if a != b:
            first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
            diffs[key]['first_diff_index'] = first
            if first is not None:
                diffs[key]['default_at_first_diff'] = a[first]
                diffs[key]['single_thread_at_first_diff'] = b[first]

    result = {
        'sequence': 'bayern_munich_3-1_chelsea_228 (non-TEST)',
        'n_frames': N_FRAMES,
        'all_record_identical': all(v['record_identical'] for v in diffs.values()),
        'diffs': diffs,
    }
    out = Path('experiments/records/experiment_M3/thread_equivalence_result.json')
    out.write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
    print(json.dumps(result, indent=1, default=str))
    print('written:', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
