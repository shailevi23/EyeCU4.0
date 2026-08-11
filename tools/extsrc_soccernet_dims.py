#!/usr/bin/env python
"""
Recover SoccerNet-V3 image dimensions without downloading the images.

The Voxel51 export stores boxes in relative coordinates and ships NO width or
height in its metadata, so ball size in pixels cannot be computed from the
metadata alone -- which is exactly what the download gate needs.

A PNG's dimensions are in bytes 16..24 of the file. An HTTP range request for
the first 32 bytes returns them, so the whole 7 GB payload stays untouched and
the gate is still decided on measured dimensions rather than an assumption.

Samples are spread across every source directory, because a single resolution
observed in one directory is not evidence about the rest.
"""

import argparse
import json
import os
import struct
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


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--per-dir', type=int, default=3)
    ap.add_argument('--revision', default=None)
    args = ap.parse_args()

    bundle = EXT / 'ca_bundle_system.pem'
    if bundle.exists():
        os.environ.setdefault('REQUESTS_CA_BUNDLE', str(bundle))
    import requests

    log = json.loads((EXT / 'huggingface' / 'download_logs' /
                      'Voxel51__SoccerNet-V3.json').read_text(encoding='utf-8'))
    revision = args.revision or log['revision']

    d = json.loads((SRC / 'metadata_only' / 'samples.json').read_text(encoding='utf-8'))
    samples = d['samples'] if isinstance(d, dict) else d
    by_dir = defaultdict(list)
    for s in samples:
        fp = s.get('filepath', '').replace('\\', '/')
        parts = [p for p in fp.split('/') if p]
        if len(parts) >= 2:
            by_dir[parts[-2]].append(fp)

    picks = []
    for dname, fps in sorted(by_dir.items()):
        idx = np.linspace(0, len(fps) - 1, min(args.per_dir, len(fps))).round().astype(int)
        picks += [fps[i] for i in dict.fromkeys(idx.tolist())]
    print(f'{len(by_dir)} source directories, probing {len(picks)} PNG headers')

    dims, failures = {}, []
    for i, fp in enumerate(picks, 1):
        rel = fp.split('data/', 1)[-1]
        url = (f'https://huggingface.co/datasets/Voxel51/SoccerNet-V3/'
               f'resolve/{revision}/data/{rel}')
        try:
            r = requests.get(url, headers={'Range': 'bytes=0-31'}, timeout=60)
            b = r.content
            if b[:8] != b'\x89PNG\r\n\x1a\n':
                failures.append((fp, f'not a PNG header: {b[:8]!r}'))
                continue
            w, h = struct.unpack('>II', b[16:24])
            dims[fp] = (int(w), int(h))
        except Exception as e:
            failures.append((fp, f'{type(e).__name__}: {e}'))
        if i % 40 == 0:
            print(f'  {i}/{len(picks)}')

    counts = Counter(dims.values())
    print(f'\ndimensions observed ({len(dims)} images, {len(failures)} failures):')
    for (w, h), n in counts.most_common():
        print(f'   {w}x{h}: {n}')
    per_dir_dim = {}
    for fp, wh in dims.items():
        per_dir_dim.setdefault(fp.replace('\\', '/').split('/')[-2], Counter())[wh] += 1
    mixed = {k: {f'{w}x{h}': n for (w, h), n in v.items()}
             for k, v in per_dir_dim.items() if len(v) > 1}

    # Ball sizes per SOURCE DIRECTORY resolution. The export mixes 1280x720 and
    # 1920x1080, so one dominant resolution applied to everything would misstate
    # every ball in the other half by 1.5x. Directories whose resolution was not
    # probed are excluded from pixel stats rather than assigned the dominant one.
    dom_w, dom_h = counts.most_common(1)[0][0]
    dir_res = {k: v.most_common(1)[0][0] for k, v in per_dir_dim.items()}
    ball, unknown_dir = [], 0
    ball_native = []
    for s in samples:
        fp = s.get('filepath', '').replace('\\', '/')
        parts = [p for p in fp.split('/') if p]
        dname = parts[-2] if len(parts) >= 2 else None
        res = dir_res.get(dname)
        for f, v in s.items():
            if isinstance(v, dict) and isinstance(v.get('detections'), list):
                for det in v['detections']:
                    if det.get('label') == 'Ball' and det.get('bounding_box'):
                        bb = det['bounding_box']
                        if res is None:
                            unknown_dir += 1
                            continue
                        ball.append((bb[2] * res[0], bb[3] * res[1]))
                        # 1920x1080-equivalent, so this is comparable with EyeCU
                        ball_native.append((bb[2] * 1920.0, bb[3] * 1080.0))
    w = np.array([b[0] for b in ball])
    h = np.array([b[1] for b in ball])
    rep = {
        'method': ('HTTP range request for the first 32 bytes of each PNG; '
                   'dimensions read from the IHDR chunk. No image payload was '
                   'downloaded.'),
        'revision': revision,
        'directories': len(by_dir),
        'headers_probed': len(dims),
        'failures': failures[:10],
        'dimensions_observed': {f'{k[0]}x{k[1]}': v for k, v in counts.items()},
        'directories_with_mixed_resolution': mixed,
        'dominant_resolution': [dom_w, dom_h],
        'ball_box_pixels_at_dominant_resolution': {
            'n': len(ball),
            'width': {'median': round(float(np.median(w)), 2),
                      'mean': round(float(w.mean()), 2),
                      'p10': round(float(np.percentile(w, 10)), 2),
                      'p25': round(float(np.percentile(w, 25)), 2),
                      'p75': round(float(np.percentile(w, 75)), 2),
                      'p90': round(float(np.percentile(w, 90)), 2),
                      'min': round(float(w.min()), 2), 'max': round(float(w.max()), 2)},
            'height_median': round(float(np.median(h)), 2),
            'width_bins': {b[2]: int(sum(bin_of(x) == b[2] for x in w)) for b in BINS},
            'le5': int((w <= 5).sum()), 'le8': int((w <= 8).sum()),
            'le12': int((w <= 12).sum()),
            'caveat': ('each ball measured at ITS OWN source directory resolution, '
                       'not at one dominant resolution'),
            'ball_boxes_in_directories_of_unknown_resolution': unknown_dir,
        },
        'directory_resolutions': {k: f'{v[0]}x{v[1]}' for k, v in dir_res.items()},
    }
    if ball_native:
        nw = np.array([b[0] for b in ball_native])
        rep['ball_box_pixels_1920x1080_equivalent'] = {
            'why': ('EyeCU frames are 16:9 broadcast; expressing every ball at '
                    '1920x1080 makes this directly comparable with EyeCU own '
                    'median of ~18 px native'),
            'n': len(nw),
            'median': round(float(np.median(nw)), 2),
            'p10': round(float(np.percentile(nw, 10)), 2),
            'p90': round(float(np.percentile(nw, 90)), 2),
            'width_bins': {b[2]: int(sum(bin_of(x) == b[2] for x in nw)) for b in BINS},
            'le5': int((nw <= 5).sum()), 'le8': int((nw <= 8).sum()),
            'le12': int((nw <= 12).sum()),
        }
    print(f'\nBALL at {dom_w}x{dom_h}: {len(ball)} boxes')
    print(f'   width px  {rep["ball_box_pixels_at_dominant_resolution"]["width"]}')
    print(f'   bins      {rep["ball_box_pixels_at_dominant_resolution"]["width_bins"]}')
    print(f'   <=5 {rep["ball_box_pixels_at_dominant_resolution"]["le5"]}  '
          f'<=8 {rep["ball_box_pixels_at_dominant_resolution"]["le8"]}  '
          f'<=12 {rep["ball_box_pixels_at_dominant_resolution"]["le12"]}')
    (SRC / 'manifests').mkdir(parents=True, exist_ok=True)
    (SRC / 'manifests' / 'image_dimensions.json').write_text(
        json.dumps(rep, indent=1), encoding='utf-8')
    print('wrote soccernet_v3/manifests/image_dimensions.json')


if __name__ == '__main__':
    main()
