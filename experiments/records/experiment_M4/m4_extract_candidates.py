#!/usr/bin/env python
"""
M4 -- mechanically extract the 60 initial-candidate TEST frames named in
INITIAL_TEST_FRAME_LIST.json to individual image files, for leakage
screening. This does NOT constitute visual inspection: no image is
displayed, read by a human, or opened by any tool other than cv2 for
programmatic pixel decoding.
"""

import json
from pathlib import Path

import cv2

OUT = Path('experiments/records/experiment_M4/candidates')


def main():
    d = json.loads(Path('experiments/records/experiment_M4/INITIAL_TEST_FRAME_LIST.json')
                   .read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for seq, info in d['sequences'].items():
        seq_dir = OUT / seq
        seq_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(info['source_video'])
        for idx0, num1 in zip(info['selected_frame_indices_0based'],
                              info['selected_frame_numbers_1based']):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'could not read frame {idx0} of {seq}')
            fname = f'{seq}_{num1:06d}.jpg'
            fpath = seq_dir / fname
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            fpath.write_bytes(buf.tobytes())
            manifest.append({'sequence': seq, 'frame_index_0based': idx0,
                            'frame_number_1based': num1, 'file': str(fpath).replace('\\', '/')})
        cap.release()
        print(seq, 'extracted', len(info['selected_frame_indices_0based']), 'frames')
    Path('experiments/records/experiment_M4/candidates_manifest.json').write_text(
        json.dumps(manifest, indent=1), encoding='utf-8')
    print('total extracted:', len(manifest))


if __name__ == '__main__':
    main()
