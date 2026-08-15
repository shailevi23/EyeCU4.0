#!/usr/bin/env python
"""
Stage 2/3/5: inventory, class audit and ball geometry for the six external sets.

Nothing is repaired. Malformed rows are counted and kept as evidence; a dataset
that needs repair before it can be read is a fact about the dataset.

One measurement decision dominates this file. Roboflow applied a resize to five
of the six exports BEFORE writing the images, so the stored pixels are not the
original frame's pixels. A ball 12 px wide in a 1920x1080 broadcast frame is
4 px wide after a stretch to 640x640. Ball width is therefore reported twice:

    stored_px      measured in the image as exported -- what a model would see
    native_equiv   back-projected to 1920x1080 using the declared preprocessing

The second is an inference from declared metadata, not a measurement, and is
labelled as such everywhere. Neither is silently substituted for the other.

Two things the first pass of this audit got wrong, fixed here:

  * S1 mixes YOLO segmentation rows (class + polygon) into a detection export.
    Reading fields 1..4 of a polygon row as cx,cy,w,h produces box geometry that
    is not merely imprecise but meaningless. Polygons are converted to their
    axis-aligned bounds and counted separately.
  * S6's "1280x1280" images are 16:9 content letterboxed into a square canvas --
    measured content rows 275..1007, i.e. 1280x~724. Projecting from the canvas
    height would understate every S6 ball by ~44%. The content box is measured
    from the pixels, not assumed.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / 'experiments' / 'external_data_audit'
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# EyeCU's four classes, in EyeCU's frozen order.
EYECU = ['player', 'goalkeeper', 'referee', 'ball']

BINS = [(0, 3, '<3'), (3, 5, '3-5'), (5, 8, '>5-8'), (8, 12, '>8-12'),
        (12, 20, '>12-20'), (20, 40, '>20-40'), (40, 1e9, '>40')]


def imread(p: Path):
    """cv2.imread cannot open a path containing Hebrew; this repo's path does."""
    import cv2
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def dims(p: Path):
    """Header-only read. Falls back to a full decode, then reports corruption."""
    from PIL import Image
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            return im.size, None            # (w, h)
    except Exception as e:
        a = imread(p)
        if a is None:
            return None, f'undecodable: {type(e).__name__}'
        return (a.shape[1], a.shape[0]), f'PIL failed, cv2 decoded: {type(e).__name__}'


def bin_of(w):
    for lo, hi, name in BINS:
        if lo == 0 and w < hi:
            return name
        if lo < w <= hi:
            return name
    return BINS[-1][2]


def content_box(root: Path, n=40):
    """Measure the non-letterboxed content region from the pixels.

    S6 pads 16:9 frames into a square canvas, so its declared 1280x1280 is not
    the image. Rather than trust the canvas, sample images and find the rows and
    columns that are not black. Returns (x0, y0, w, h) in stored pixels, or None
    if the content fills the canvas.
    """
    import cv2
    imgs = []
    for split in ('train', 'valid', 'test'):
        d = root / split / 'images'
        if d.exists():
            imgs += [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT]
    if not imgs:
        return None
    sel = [imgs[i] for i in np.linspace(0, len(imgs) - 1, min(n, len(imgs)))
           .round().astype(int)]
    y0s, y1s, x0s, x1s, shape = [], [], [], [], None
    for p in sel:
        a = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if a is None:
            continue
        shape = a.shape
        r = np.where(a.max(axis=(1, 2)) > 12)[0]
        c = np.where(a.max(axis=(0, 2)) > 12)[0]
        if len(r) and len(c):
            y0s.append(r.min()); y1s.append(r.max())
            x0s.append(c.min()); x1s.append(c.max())
    if shape is None or not y0s:
        return None
    # median, so one dark night-match frame cannot define the content box
    y0, y1 = int(np.median(y0s)), int(np.median(y1s))
    x0, x1 = int(np.median(x0s)), int(np.median(x1s))
    h, w = shape[0], shape[1]
    if y0 <= 2 and x0 <= 2 and y1 >= h - 3 and x1 >= w - 3:
        return None
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def scale_to_native(src, w_px, h_px, img_w, img_h, content=None):
    """Back-project a stored-pixel box to a 1920x1080 broadcast equivalent.

    'Stretch' scales the axes independently. 'Fill (with center crop)' preserves
    aspect, so one uniform factor applies -- but it must be taken against the
    CONTENT height, not the padded canvas. Where preprocessing is absent the
    image is already native and no projection is needed.
    """
    pre = ' '.join(src['preprocessing']).lower()
    if 'stretch' in pre:
        return w_px * 1920.0 / img_w, h_px * 1080.0 / img_h, 'stretch->1920x1080'
    if 'center crop' in pre or 'fill' in pre:
        ch = content[3] if content else img_h
        f = 1080.0 / ch
        how = (f'uniform x{f:.3f} against measured content height {ch}px'
               if content else 'uniform (aspect preserved by fill)')
        return w_px * f, h_px * f, how
    return w_px, h_px, 'none (exported at original size)'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    srcs = json.loads((AUDIT / 'raw' / 'SOURCES.json').read_text(encoding='utf-8'))['sources']
    cmap = json.loads((AUDIT / 'reports' / 'class_map.json').read_text(encoding='utf-8'))
    report = {'note': __doc__.strip(), 'sources': {}}
    all_ball = []

    for sid, src in srcs.items():
        root = REPO / src['extracted_to']
        names = src['declared_classes']
        # the PROPOSED mapping, used only so that ball geometry can be measured
        # on sources whose classes are bare integers. No label is rewritten.
        to_eyecu = {int(k): v for k, v in cmap[sid]['mapping'].items()}
        content = content_box(root)
        inv = {'source': sid,
               'project': f"{src['workspace']}/{src['project']} v{src['version']}",
               'declared_classes': names,
               'proposed_eyecu_mapping': cmap[sid]['mapping'],
               'declared_image_count': src['declared_image_count'],
               'preprocessing': src['preprocessing'],
               'augmentation': src['augmentation'],
               'measured_content_box_xywh': content,
               'splits': {}, 'dimensions': Counter(), 'problems': defaultdict(int),
               'problem_examples': defaultdict(list)}
        if content:
            inv['letterbox_note'] = (
                f'canvas is padded: real content is {content[2]}x{content[3]} px '
                f'inside the stored canvas; the padding is not image data')
        boxes = Counter()
        empty_label_files = 0
        images_total = 0
        per_image = []          # one row per image, reused by later stages
        ball_rows = []

        for split in ('train', 'valid', 'test'):
            idir, ldir = root / split / 'images', root / split / 'labels'
            if not idir.exists():
                continue
            imgs = sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT)
            lbls = {p.stem for p in ldir.iterdir()} if ldir.exists() else set()
            inv['splits'][split] = {'images': len(imgs),
                                    'label_files': len(lbls),
                                    'images_without_label': 0,
                                    'labels_without_image': len(lbls - {p.stem for p in imgs})}
            for p in imgs:
                images_total += 1
                wh, err = dims(p)
                if wh is None:
                    inv['problems']['corrupt_images'] += 1
                    inv['problem_examples']['corrupt_images'].append(p.name)
                    continue
                if err:
                    inv['problems']['image_header_warnings'] += 1
                iw, ih = wh
                inv['dimensions'][f'{iw}x{ih}'] += 1

                lp = ldir / f'{p.stem}.txt'
                if not lp.exists():
                    inv['splits'][split]['images_without_label'] += 1
                    inv['problems']['missing_label_file'] += 1
                    per_image.append({'sid': sid, 'split': split, 'file': p.name,
                                      'w': iw, 'h': ih, 'counts': {}, 'n': 0,
                                      'label_file': False})
                    continue
                rows = [l for l in lp.read_text(encoding='utf-8',
                                                errors='replace').splitlines() if l.strip()]
                if not rows:
                    empty_label_files += 1
                counts = Counter()
                for r in rows:
                    f = r.split()
                    if len(f) < 5:
                        inv['problems']['malformed_annotation'] += 1
                        inv['problem_examples']['malformed_annotation'].append(
                            f'{p.stem}.txt: {r[:40]}')
                        continue
                    try:
                        ci = int(float(f[0]))
                        vals = [float(x) for x in f[1:]]
                    except ValueError:
                        inv['problems']['malformed_annotation'] += 1
                        continue
                    if len(vals) == 4:
                        cx, cy, bw, bh = vals
                    elif len(vals) >= 6 and len(vals) % 2 == 0:
                        # YOLO segmentation polygon in a detection export
                        inv['problems']['segmentation_polygon_rows'] += 1
                        xs, ys = vals[0::2], vals[1::2]
                        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
                    else:
                        inv['problems']['malformed_annotation'] += 1
                        inv['problem_examples']['malformed_annotation'].append(
                            f'{p.stem}.txt: {len(f)} fields')
                        continue
                    if not 0 <= ci < len(names):
                        inv['problems']['class_index_out_of_range'] += 1
                        continue
                    if bw <= 0 or bh <= 0:
                        inv['problems']['zero_or_negative_area'] += 1
                        continue
                    x1, y1 = cx - bw / 2, cy - bh / 2
                    x2, y2 = cx + bw / 2, cy + bh / 2
                    if x1 < -1e-6 or y1 < -1e-6 or x2 > 1 + 1e-6 or y2 > 1 + 1e-6:
                        inv['problems']['out_of_bounds'] += 1
                    cname = names[ci]
                    counts[cname] += 1
                    boxes[cname] += 1
                    if to_eyecu.get(ci) == 'ball':
                        w_px, h_px = bw * iw, bh * ih
                        nw, nh, how = scale_to_native(src, w_px, h_px, iw, ih, content)
                        ball_rows.append({'file': p.name, 'split': split,
                                          'img_w': iw, 'img_h': ih,
                                          'w_px': w_px, 'h_px': h_px,
                                          'area_px': w_px * h_px,
                                          'w_frac': bw, 'h_frac': bh,
                                          'native_w': nw, 'native_h': nh,
                                          'projection': how})
                per_image.append({'sid': sid, 'split': split, 'file': p.name,
                                  'w': iw, 'h': ih, 'counts': dict(counts),
                                  'eyecu': {to_eyecu.get(names.index(k), '?'): v
                                            for k, v in counts.items()},
                                  'n': sum(counts.values()), 'label_file': True})

        inv['images_total'] = images_total
        inv['boxes_per_class'] = dict(boxes)
        mapped = Counter()
        for i, nm in enumerate(names):
            if nm in boxes:
                mapped[to_eyecu.get(i, 'UNMAPPED')] += boxes[nm]
        inv['boxes_per_eyecu_class_IF_MAPPED'] = {c: mapped.get(c, 0) for c in EYECU}
        inv['boxes_total'] = sum(boxes.values())
        inv['empty_label_files'] = empty_label_files
        inv['images_with_zero_boxes'] = sum(1 for r in per_image if r['n'] == 0)
        inv['problems'] = dict(inv['problems'])
        inv['problem_examples'] = {k: v[:5] for k, v in inv['problem_examples'].items()}
        inv['dimensions'] = dict(inv['dimensions'])
        if src['declared_image_count'] != images_total:
            inv['declared_vs_actual'] = (
                f"README declares {src['declared_image_count']} images, archive "
                f"contains {images_total}")

        # ---- ball geometry -------------------------------------------------
        if ball_rows:
            w = np.array([r['w_px'] for r in ball_rows])
            nw = np.array([r['native_w'] for r in ball_rows])
            uniq = len({r['file'] for r in ball_rows})

            def stats(a):
                return {'median': round(float(np.median(a)), 2),
                        'mean': round(float(a.mean()), 2),
                        'p10': round(float(np.percentile(a, 10)), 2),
                        'p25': round(float(np.percentile(a, 25)), 2),
                        'p75': round(float(np.percentile(a, 75)), 2),
                        'p90': round(float(np.percentile(a, 90)), 2)}
            inv['ball'] = {
                'instances': len(ball_rows),
                'unique_images_with_ball': uniq,
                'projection': ball_rows[0]['projection'],
                'stored_px': {
                    'width': stats(w),
                    'height': stats(np.array([r['h_px'] for r in ball_rows])),
                    'area': stats(np.array([r['area_px'] for r in ball_rows])),
                    'width_bins': {b[2]: int(sum(bin_of(x) == b[2] for x in w))
                                   for b in BINS},
                    'le5': int((w <= 5).sum()), 'le8': int((w <= 8).sum()),
                    'le12': int((w <= 12).sum()),
                    'w_frac': stats(np.array([r['w_frac'] for r in ball_rows])),
                    'h_frac': stats(np.array([r['h_frac'] for r in ball_rows])),
                },
                'native_equivalent_1920x1080_INFERRED': {
                    'width': stats(nw),
                    'width_bins': {b[2]: int(sum(bin_of(x) == b[2] for x in nw))
                                   for b in BINS},
                    'le5': int((nw <= 5).sum()), 'le8': int((nw <= 8).sum()),
                    'le12': int((nw <= 12).sum()),
                    'caveat': ('derived from the DECLARED Roboflow preprocessing, '
                               'not measured; the original frames are not in the export'),
                },
            }
            all_ball += [dict(r, sid=sid) for r in ball_rows]
        else:
            inv['ball'] = {'instances': 0, 'unique_images_with_ball': 0}

        report['sources'][sid] = inv
        (AUDIT / 'reports' / f'{sid}_per_image.json').write_text(
            json.dumps(per_image, indent=0), encoding='utf-8')

        print(f'{sid}  {inv["images_total"]:>5} img  {inv["boxes_total"]:>6} boxes  '
              f'{dict(boxes)}')
        if inv['problems']:
            print(f'      problems: {inv["problems"]}')

    # ---- combined ball ------------------------------------------------------
    if all_ball:
        w = np.array([r['w_px'] for r in all_ball])
        nw = np.array([r['native_w'] for r in all_ball])
        report['combined_ball'] = {
            'instances': len(all_ball),
            'unique_images_with_ball': len({(r['sid'], r['file']) for r in all_ball}),
            'stored_px': {
                'median': round(float(np.median(w)), 2),
                'mean': round(float(w.mean()), 2),
                'p10': round(float(np.percentile(w, 10)), 2),
                'p25': round(float(np.percentile(w, 25)), 2),
                'p75': round(float(np.percentile(w, 75)), 2),
                'p90': round(float(np.percentile(w, 90)), 2),
                'width_bins': {b[2]: int(sum(bin_of(x) == b[2] for x in w)) for b in BINS},
                'le5': int((w <= 5).sum()), 'le8': int((w <= 8).sum()),
                'le12': int((w <= 12).sum()),
            },
            'native_equivalent_1920x1080_INFERRED': {
                'median': round(float(np.median(nw)), 2),
                'width_bins': {b[2]: int(sum(bin_of(x) == b[2] for x in nw)) for b in BINS},
                'le5': int((nw <= 5).sum()), 'le8': int((nw <= 8).sum()),
                'le12': int((nw <= 12).sum()),
            },
        }
        (AUDIT / 'reports' / 'ball_instances.json').write_text(
            json.dumps(all_ball, indent=0), encoding='utf-8')

    (AUDIT / 'reports' / 'inventory.json').write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'\ncombined ball instances: {report.get("combined_ball", {}).get("instances", 0)}')
    print(f'wrote reports/inventory.json')


if __name__ == '__main__':
    main()
