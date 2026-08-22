#!/usr/bin/env python
"""
M5 Section 4-5 -- ONE prediction pass per TEST sequence, raw detector output.

Runs the frozen TwoBranchDetector (best_A_960.pt for player/goalkeeper/
referee, SN3D yolo-sn-ball.pt for ball) directly on each of the 60 frozen
TEST_DETECTION_ANNOTATIONS frames, at the frozen accept_confidence=0.25
for both branches -- no BallTemporalSelector, no CBIoU tracking, no
threshold change. This is deliberately the RAW detector output the M4
contract (Section 3) requires for scoring: ball GT is
ALL_VISIBLE_PHYSICAL_FOOTBALLS, so scoring must happen before the selector
reduces candidates toward one canonical ball.

No sequence context is needed here: LocalDetector.detect() is a stateless
single-frame forward pass (see trackers/detector.py) -- neither branch has
any temporal memory, so this is not an approximation of "one pass", it is
the literal single inference each frame gets in the same one-pass
philosophy, just without the parts (CBIoU identity, selector) that operate
on results this script does not score.

CPU, single-threaded, deterministic runtime per the M4 contract's gate.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from trackers.detector import (LocalDetector, TwoBranchDetector,               # noqa: E402
                               resolve_sn3d_ball_path, verify_sn3d_ball_checkpoint)

M4 = REPO / 'experiments/records/experiment_M4'
GT = json.loads((M4 / 'TEST_DETECTION_ANNOTATIONS.json').read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)

    ball_path = resolve_sn3d_ball_path()
    verify_sn3d_ball_checkpoint(ball_path)

    human = LocalDetector(model_path=str(REPO / 'best_A_960.pt'), confidence=0.25, imgsz=960)
    ball = LocalDetector(model_path=ball_path, confidence=0.25, imgsz=1280)
    det = TwoBranchDetector(human, ball)

    records = []
    t0 = time.time()
    for r in GT:
        img_path = REPO / r['file']
        img = cv2.imdecode(__import__('numpy').fromfile(str(img_path), dtype='uint8'), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f'could not read {img_path}')
        preds = det.detect(img)
        records.append({
            'sequence': r['sequence'],
            'frame_number_1based': r['frame_number_1based'],
            'file': r['file'],
            'detections': [{'class': p['class'], 'bbox': [round(v, 2) for v in p['bbox']],
                           'confidence': round(p['confidence'], 4)} for p in preds],
        })
        print(f"{r['sequence']} {r['frame_number_1based']}: "
             f"{len(preds)} raw detections", flush=True)

    elapsed = time.time() - t0
    out = {
        'method': 'TwoBranchDetector (best_A_960.pt + SN3D yolo-sn-ball.pt), raw, '
                 'accept_confidence=0.25 both branches, no selector, no tracker, single-frame stateless inference',
        'human_checkpoint_sha256': __import__('hashlib').sha256((REPO / 'best_A_960.pt').read_bytes()).hexdigest(),
        'ball_checkpoint_sha256': __import__('hashlib').sha256(Path(ball_path).read_bytes()).hexdigest(),
        'n_frames': len(records),
        'elapsed_seconds': round(elapsed, 1),
        'detector_stats': det.stats(),
        'records': records,
    }
    out_path = Path('experiments/records/experiment_M5/RAW_PREDICTIONS.json')
    out_path.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'\n{len(records)} frames, {elapsed:.1f}s total')
    print('written:', out_path)


if __name__ == '__main__':
    main()
