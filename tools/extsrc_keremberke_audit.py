#!/usr/bin/env python
"""
Audit keremberke/football-object-detection for ball value.

This one matters because it is the first source seen in either audit whose
Roboflow export applied NO resize -- the card says "original-raw-images" and
"Auto-orientation" only. Everywhere else, a stored ball measured in pixels was a
ball in a downscaled image, and the tiny-ball counts evaporated on
back-projection. Here the stored pixels should be the real pixels, and the
audit's job is to confirm that from the images rather than trust the card.

COCO format, so boxes are already absolute pixels and image dimensions are
declared per image; both are cross-checked against the decoded files.
"""

import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / 'EyeCU_external_data'
SRC = EXT / 'huggingface' / 'keremberke_football_object_detection'
BINS = [(0, 3, '<3'), (3, 5, '3-5'), (5, 8, '>5-8'), (8, 12, '>8-12'),
        (12, 20, '>12-20'), (20, 40, '>20-40'), (40, 1e9, '>40')]


def bin_of(w):
    for lo, hi, n in BINS:
        if lo == 0 and w < hi:
            return n
        if lo < w <= hi:
            return n
    return '>40'


def stats(a, nd=2):
    a = np.asarray(a, float)
    return {'n': int(a.size), 'median': round(float(np.median(a)), nd),
            'mean': round(float(a.mean()), nd),
            'p10': round(float(np.percentile(a, 10)), nd),
            'p25': round(float(np.percentile(a, 25)), nd),
            'p75': round(float(np.percentile(a, 75)), nd),
            'p90': round(float(np.percentile(a, 90)), nd),
            'min': round(float(a.min()), nd), 'max': round(float(a.max()), nd)}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import cv2

    raw = SRC / 'raw' / 'data'
    ex = SRC / 'extracted'
    ex.mkdir(parents=True, exist_ok=True)
    for z in sorted(raw.glob('*.zip')):
        d = ex / z.stem
        if not d.exists():
            d.mkdir(parents=True)
            zipfile.ZipFile(z).extractall(d)
        print(f'{z.name} -> {d.name}')

    report = {'source': 'keremberke/football-object-detection',
              'underlying': 'roboflow augmented-startups/football-player-detection-kucab v3',
              'license_from_card': 'CC BY 4.0',
              'preprocessing_declared': 'auto-orientation only; NO resize',
              'augmentation_declared': 'none',
              'splits': {}}
    all_ball, all_cls = [], Counter()
    dims = Counter()
    per_image = []
    mismatched_dims = 0

    for split_dir in sorted(ex.iterdir()):
        if not split_dir.is_dir():
            continue
        anns = list(split_dir.rglob('_annotations.coco.json'))
        if not anns:
            print(f'  {split_dir.name}: no COCO json found')
            continue
        aj = json.loads(anns[0].read_text(encoding='utf-8'))
        cats = {c['id']: c['name'] for c in aj['categories']}
        imgs = {i['id']: i for i in aj['images']}
        base = anns[0].parent
        by_img = defaultdict(list)
        for a in aj['annotations']:
            by_img[a['image_id']].append(a)
        cls = Counter()
        ball_rows = []
        for iid, im in imgs.items():
            W, H = im['width'], im['height']
            dims[f'{W}x{H}'] += 1
            fp = base / im['file_name']
            if fp.exists():
                arr = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8),
                                   cv2.IMREAD_COLOR)
                if arr is not None and (arr.shape[1], arr.shape[0]) != (W, H):
                    mismatched_dims += 1
            counts = Counter()
            for a in by_img.get(iid, []):
                name = cats.get(a['category_id'], str(a['category_id']))
                cls[name] += 1
                counts[name] += 1
                if 'ball' in name.lower() or 'football' == name.lower():
                    x, y, w, h = a['bbox']
                    ball_rows.append({'file': im['file_name'], 'split': split_dir.name,
                                      'w': w, 'h': h, 'img_w': W, 'img_h': H,
                                      'w_native1920': w * 1920.0 / W})
            per_image.append({'split': split_dir.name, 'file': im['file_name'],
                              'w': W, 'h': H, 'counts': dict(counts),
                              'n': sum(counts.values())})
        all_cls.update(cls)
        all_ball += ball_rows
        report['splits'][split_dir.name] = {
            'images': len(imgs), 'annotations': sum(cls.values()),
            'classes': dict(cls), 'categories_declared': list(cats.values()),
            'images_with_zero_annotations': sum(
                1 for i in imgs if not by_img.get(i)),
        }
        print(f'  {split_dir.name}: {len(imgs)} images, {sum(cls.values())} anns, {dict(cls)}')

    report['classes_total'] = dict(all_cls)
    report['images_total'] = sum(v['images'] for v in report['splits'].values())
    report['annotations_total'] = sum(all_cls.values())
    report['image_dimensions'] = dict(dims.most_common())
    report['images_whose_decoded_size_differs_from_coco'] = mismatched_dims
    report['images_with_zero_annotations'] = sum(
        v['images_with_zero_annotations'] for v in report['splits'].values())

    if all_ball:
        w = np.array([b['w'] for b in all_ball])
        nw = np.array([b['w_native1920'] for b in all_ball])
        report['ball'] = {
            'instances': len(all_ball),
            'unique_images_with_ball': len({(b['split'], b['file']) for b in all_ball}),
            'stored_px': {
                'width': stats(w),
                'height': stats([b['h'] for b in all_ball]),
                'width_bins': {b[2]: int(sum(bin_of(x) == b[2] for x in w)) for b in BINS},
                'le5': int((w <= 5).sum()), 'le8': int((w <= 8).sum()),
                'le12': int((w <= 12).sum()),
            },
            'width_1920_equivalent': {
                'note': ('the export declares no resize, so stored == original; '
                         'this column only rescales for comparison with EyeCU'),
                'median': round(float(np.median(nw)), 2),
                'width_bins': {b[2]: int(sum(bin_of(x) == b[2] for x in nw)) for b in BINS},
                'le5': int((nw <= 5).sum()), 'le8': int((nw <= 8).sum()),
                'le12': int((nw <= 12).sum()),
            },
        }
        print(f"\nBALL: {len(all_ball)} instances in "
              f"{report['ball']['unique_images_with_ball']} images")
        print(f"   stored width {report['ball']['stored_px']['width']}")
        print(f"   bins {report['ball']['stored_px']['width_bins']}")
        print(f"   <=5 {report['ball']['stored_px']['le5']}  "
              f"<=8 {report['ball']['stored_px']['le8']}  "
              f"<=12 {report['ball']['stored_px']['le12']}")
    else:
        report['ball'] = {'instances': 0}
        print('\nBALL: none')

    (SRC / 'manifests').mkdir(parents=True, exist_ok=True)
    (SRC / 'manifests' / 'audit.json').write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')
    (SRC / 'manifests' / 'per_image.json').write_text(
        json.dumps(per_image, indent=0), encoding='utf-8')
    if all_ball:
        (SRC / 'manifests' / 'ball_instances.json').write_text(
            json.dumps(all_ball, indent=0), encoding='utf-8')
    print('\nwrote keremberke_football_object_detection/manifests/audit.json')


if __name__ == '__main__':
    main()
