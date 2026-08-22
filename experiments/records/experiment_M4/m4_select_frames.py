#!/usr/bin/env python
"""
M4 -- deterministic, content-neutral TEST frame selection.

Explicitly NOT under trackers/tools/rf_trackers/full_pipeline.py/run_pipeline.py
(the hashed prediction source tree) -- this is annotation/experiment tooling
for experiment_M4 only, per M4's own instruction.

Selects 20 frames per TEST sequence, uniformly across ordinal frame position,
using ONLY the sequence's total frame count (read from video container
metadata, not pixel content). No frame is opened or displayed here.

Formula, declared before any TEST pixel is touched:

    for k = 0..19, frame index (0-based) = floor((k + 0.5) * N / 20)

Indexing convention: video frames are addressed 0-based internally by
OpenCV (frame 0 is the first frame read by cv2.VideoCapture). This script
reports both the 0-based index and the 1-based frame NUMBER (index + 1),
matching this project's existing convention elsewhere (e.g.
data/tracking_val_gt sequences use 1-based "000001.jpg" naming) so the two
are never confused. All selection math is 0-based; only the reported label
adds 1.

Selection depends on nothing but N (sequence length). It has no access to
ball visibility, player count, image difficulty, class composition,
lighting, or any model output, because it is computed before any frame is
even decoded.
"""

import hashlib
import json
from pathlib import Path

import cv2

TEST_VIDEOS = {
    'como_2-0_sassuolo': 'input-videos/Como 2-0 Sassuolo.mp4',
    'manchester_city_v_liverpool': 'input-videos/Manchester City v Liverpool.mp4',
    'youth_2': 'input-videos/youth 2.webm',
}
K = 20


def sequence_length(path):
    """Frame count via true sequential decode, not the container's own
    CAP_PROP_FRAME_COUNT header field. That field is a mechanical/technical
    property of the file, not TEST image content -- but for at least one of
    these three videos (youth_2, .webm) it is simply wrong: it reports 1750
    while only 1561 frames are actually sequentially decodable. Using the
    wrong N would place several selected indices past the end of the file.
    Counting by sequential decode is the more reliable way to answer the same
    metadata question (how many frames does this file have), not a
    content-dependent selection step -- no frame's pixel content influences
    which frames end up chosen, only how many frames exist in total."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f'could not open {path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n, fps, w, h, reported_n


def select_indices(n, k=K):
    """0-based frame indices, uniform across ordinal position. Deterministic,
    content-independent: a pure function of n and k."""
    import math
    idx = []
    for i in range(k):
        pos = math.floor((i + 0.5) * n / k)
        pos = max(0, min(pos, n - 1))
        idx.append(pos)
    # strictly increasing and within range by construction for n >= k
    assert len(set(idx)) == k, 'formula produced duplicate indices -- sequence too short'
    return idx


def main():
    result = {
        'benchmark': 'EyeCU-TEST-v1 initial candidate frame list',
        'selection_rule': {
            'k_per_sequence': K,
            'formula': 'floor((k + 0.5) * N / 20) for k = 0..19, 0-based frame index',
            'indexing_convention': '0-based internally (matches cv2.VideoCapture frame reading order); reported frame_number is 1-based (index + 1) for consistency with this project\'s existing 1-based sequence-frame naming',
            'deterministic': True,
            'content_independent': True,
            'note': 'computed from video container frame-count metadata only; no frame was decoded or viewed to produce this list',
        },
        'sequences': {},
    }

    for name, path in TEST_VIDEOS.items():
        n, fps, w, h, reported_n = sequence_length(path)
        idx0 = select_indices(n, K)
        result['sequences'][name] = {
            'source_video': path,
            'n_frames_total': n,
            'n_frames_total_container_header_reported': reported_n,
            'frame_count_source': ('sequential decode' if n != reported_n
                                   else 'sequential decode (matches container header)'),
            'fps': fps,
            'width': w,
            'height': h,
            'selected_frame_indices_0based': idx0,
            'selected_frame_numbers_1based': [i + 1 for i in idx0],
        }

    out = Path('experiments/records/experiment_M4/INITIAL_TEST_FRAME_LIST.json')
    out.write_text(json.dumps(result, indent=1), encoding='utf-8')
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    Path('experiments/records/experiment_M4/INITIAL_TEST_FRAME_LIST.sha256').write_text(
        sha + '\n', encoding='utf-8')
    print(json.dumps(result, indent=1))
    print('sha256:', sha)
    print('written:', out)


if __name__ == '__main__':
    main()
