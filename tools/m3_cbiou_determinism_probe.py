#!/usr/bin/env python
"""
M3 -- minimum experiment to isolate the CBIoU nondeterminism P1.1 observed on
youth_premier_league_1133.

Step 1: run the human detector ONCE over the sequence and freeze the raw
        per-frame detection records to disk (boxes/class/confidence).
Step 2: feed that EXACT frozen stream into two fresh CBIoUTracker instances,
        frame by frame, replicating exactly what
        FootballTracker.get_object_tracks() does for the human path.
Step 3: compare the two tracker runs record-for-record (frame, track id,
        bbox, class).

If the two CBIoU-only runs (byte-identical input) differ, the nondeterminism
is inside tracking/post-processing. If they are identical, it is upstream
(most likely the detector).

READ-ONLY diagnostic. Does not change CBIoU, its thresholds, or the tracking
algorithm.
"""

import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackers.detector import BALL_ACCEPT_CONF, CLASS_IDS, HUMAN_CLASSES  # noqa: E402
from trackers.football_tracker import FootballTracker                    # noqa: E402
from rf_trackers import CBIoUTracker                                     # noqa: E402

SEQ = Path('data/tracking_val_gt/sequences/youth_premier_league_1133/img1')
FROZEN = Path('experiments/records/experiment_M3/cbiou_probe_detections.pkl')
FRAME_RATE = 25.0


def get_frozen_detections():
    if FROZEN.exists():
        print('reusing frozen detection stream:', FROZEN)
        with open(FROZEN, 'rb') as f:
            return pickle.load(f)

    frame_ids = list(range(1, 205))
    imgs = [cv2.imdecode(np.fromfile(str(SEQ / f'{f:06d}.jpg'), dtype=np.uint8),
                         cv2.IMREAD_COLOR) for f in frame_ids]
    tracker = FootballTracker(model_path='best_A_960.pt', imgsz=960,
                              confidence=BALL_ACCEPT_CONF, persist_cache=False,
                              ball_candidate_pool=True, ball_detector_backend='sn3d')
    tracker.detector.clear_cache()
    print(f'running detector once over {len(imgs)} frames...', flush=True)
    detections_list = tracker.detect_objects_in_frames(imgs)
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    with open(FROZEN, 'wb') as f:
        pickle.dump(detections_list, f)
    print('froze detection stream to', FROZEN)
    return detections_list


def run_cbiou_once(detections_list):
    """Exact replica of the human-association path in
    FootballTracker.get_object_tracks() (trackers/football_tracker.py),
    isolated from detection so the SAME frozen input can be replayed."""
    tracker = CBIoUTracker(frame_rate=FRAME_RATE)
    out = []
    for frame_detections in detections_list:
        boxes, class_ids, confidences = [], [], []
        for det in frame_detections:
            class_name = det.get('class')
            if class_name not in CLASS_IDS or class_name not in HUMAN_CLASSES:
                continue
            boxes.append(det['bbox'])
            class_ids.append(CLASS_IDS[class_name])
            confidences.append(det.get('confidence', 0.5))

        if boxes:
            det_obj = sv.Detections(xyxy=np.array(boxes), class_id=np.array(class_ids),
                                    confidence=np.array(confidences))
            tracked = tracker.update(det_obj)
            frame_out = []
            for i in range(len(tracked.xyxy)):
                raw_id = tracked.tracker_id[i]
                frame_out.append({
                    'track_id': None if raw_id is None else int(raw_id),
                    'bbox': [round(float(v), 6) for v in tracked.xyxy[i]],
                    'class_id': int(tracked.class_id[i]),
                })
            out.append(frame_out)
        else:
            # Matches trackers/football_tracker.py exactly: when a frame has
            # no human boxes, tracker.update() is never called at all (not
            # called with an empty Detections either) -- replicate that
            # precisely, since calling update() on an empty frame could by
            # itself perturb internal tracker state relative to production.
            out.append([])
    return out


def compare(run_a, run_b):
    assert len(run_a) == len(run_b), f'frame count differs: {len(run_a)} vs {len(run_b)}'
    diffs = []
    for i, (fa, fb) in enumerate(zip(run_a, run_b)):
        if fa != fb:
            diffs.append({'frame_idx': i, 'run_a': fa, 'run_b': fb})
    return diffs


def main():
    detections_list = get_frozen_detections()
    print('running CBIoU pass 1...')
    run_a = run_cbiou_once(detections_list)
    print('running CBIoU pass 2...')
    run_b = run_cbiou_once(detections_list)

    diffs = compare(run_a, run_b)
    result = {
        'sequence': 'youth_premier_league_1133',
        'n_frames': len(detections_list),
        'n_frames_differing': len(diffs),
        'record_identical': len(diffs) == 0,
        'first_5_diffs': diffs[:5],
    }
    out_path = Path('experiments/records/experiment_M3/cbiou_determinism_probe_result.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'first_5_diffs'}, indent=1))
    if diffs:
        print('FIRST DIFF:', json.dumps(diffs[0], indent=1, default=str)[:2000])
    print('written:', out_path)


if __name__ == '__main__':
    raise SystemExit(main())
