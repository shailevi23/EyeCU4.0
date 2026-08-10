#!/usr/bin/env python
"""
Visual audit of GT box tightness. Read-only; alters no annotation.

The question this answers is "are these boxes drawn to the EyeCU convention --
tight around the visible person, minimal background, no shadow, no horizontal
safety margin" -- and that question is settled by looking at the image, not by
agreeing with a detector.

So two contact sheets are produced from the SAME 40 observations:

    _gt_only    GT box alone. Judge tightness here first.
    _with_det   GT plus the frozen detector box where a reasonable match
                exists, as context for why the question came up at all.

The order matters. Seeing the detector box first anchors the eye to it, and
the detector is not the authority -- it is one more opinion, from a model whose
own box convention is what is under discussion.

Sampling is deterministic and stratified across:

    time        first / middle / last third of the sequence
    size        small / medium / large, by GT box height terciles
    role        player / goalkeeper / referee
    occlusion   as the ANNOTATOR marked it in CVAT (occluded="1"), read from
                the export -- not guessed from box overlap

Occlusion comes from the CVAT export because the canonical importer does not
currently carry `occluded` through. That is worth deciding about separately;
here it is simply read at source.
"""

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N_SAMPLES = 40
COLS, CELL_W, CELL_H = 8, 240, 300
CROP_PAD = 1.9          # crop this many box-widths/heights around the box
SEED = 20260810         # fixed: the audit must be reproducible
GT_COLOUR = (80, 240, 80)
DET_COLOUR = (60, 170, 255)
MATCH_IOU = 0.2


def imread(p: Path):
    if not p.exists():
        return None
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(p: Path, img):
    """cv2.imwrite cannot handle the non-ASCII path this repo lives under."""
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(buf.tobytes())
    return ok


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def occluded_flags(xml_path: Path):
    """{(package_frame, identity): True} straight from the annotator's marks."""
    out = {}
    root = ET.parse(xml_path).getroot()
    for tr in root.findall('.//track'):
        ident = int(tr.get('id')) + 1
        for b in tr.findall('box'):
            if b.get('outside') == '1':
                continue
            out[(int(b.get('frame')) + 1, ident)] = b.get('occluded') == '1'
    return out


def load_detector(det_path: Path):
    per_frame = defaultdict(list)
    for line in det_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        p = line.split(',')
        f = int(p[0])
        x, y, w, h = (float(v) for v in p[2:6])
        per_frame[f].append([x, y, x + w, y + h])
    return per_frame


def sample(boxes, occ, n_frames, k=N_SAMPLES):
    """
    Deterministic stratified sample.

    Round-robin over the strata rather than sampling proportionally, so the
    rare cells -- goalkeeper, occluded, large -- actually appear. A sample that
    mirrors the population would be almost entirely medium unoccluded players
    and would tell the reviewer nothing they do not already see.
    """
    heights = sorted(b['bbox'][3] - b['bbox'][1] for b in boxes)
    t1, t2 = heights[len(heights) // 3], heights[2 * len(heights) // 3]

    def size_of(b):
        h = b['bbox'][3] - b['bbox'][1]
        return 'small' if h < t1 else ('medium' if h < t2 else 'large')

    def time_of(b):
        return ('begin' if b['frame'] <= n_frames // 3 else
                'middle' if b['frame'] <= 2 * n_frames // 3 else 'end')

    strata = defaultdict(list)
    for b in boxes:
        key = (time_of(b), size_of(b), b['role'],
               'occluded' if occ.get((b['frame'], b['id'])) else 'normal')
        strata[key].append(b)

    rng = random.Random(SEED)
    for v in strata.values():
        v.sort(key=lambda b: (b['frame'], b['id']))     # stable before shuffle
        rng.shuffle(v)

    keys = sorted(strata)
    rng.shuffle(keys)
    picked, i = [], 0
    while len(picked) < k and any(strata[key] for key in keys):
        key = keys[i % len(keys)]
        if strata[key]:
            b = dict(strata[key].pop())
            b['stratum'] = {'time': key[0], 'size': key[1], 'role': key[2],
                            'occlusion': key[3]}
            picked.append(b)
        i += 1
    picked.sort(key=lambda b: (b['frame'], b['id']))
    return picked, (t1, t2)


def cell(img, box, det_box, label, show_det):
    """One crop, upscaled, with the box(es) drawn at display resolution."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max(w * CROP_PAD, 26) / 2
    half_h = max(h * CROP_PAD, 34) / 2
    # keep the crop's aspect equal to the cell's so nothing is stretched
    if half_w / half_h > CELL_W / CELL_H:
        half_h = half_w * CELL_H / CELL_W
    else:
        half_w = half_h * CELL_W / CELL_H

    cx1, cy1 = int(round(cx - half_w)), int(round(cy - half_h))
    cx2, cy2 = int(round(cx + half_w)), int(round(cy + half_h))
    pad_l, pad_t = max(0, -cx1), max(0, -cy1)
    pad_r = max(0, cx2 - img.shape[1])
    pad_b = max(0, cy2 - img.shape[0])
    crop = img[max(0, cy1):min(img.shape[0], cy2),
               max(0, cx1):min(img.shape[1], cx2)]
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                  cv2.BORDER_CONSTANT, value=(30, 30, 30))
    if crop.size == 0:
        crop = np.full((CELL_H, CELL_W, 3), 30, np.uint8)

    sx = CELL_W / crop.shape[1]
    sy = CELL_H / crop.shape[0]
    # nearest neighbour: the reviewer is judging where the box edge sits
    # relative to real pixels, and interpolation would blur exactly that
    out = cv2.resize(crop, (CELL_W, CELL_H), interpolation=cv2.INTER_NEAREST)

    def draw(b, colour, thick):
        p1 = (int(round((b[0] - cx1) * sx)), int(round((b[1] - cy1) * sy)))
        p2 = (int(round((b[2] - cx1) * sx)), int(round((b[3] - cy1) * sy)))
        cv2.rectangle(out, p1, p2, colour, thick)

    if show_det and det_box is not None:
        draw(det_box, DET_COLOUR, 1)
    draw(box, GT_COLOUR, 1)

    bar = np.full((22, CELL_W, 3), 20, np.uint8)
    cv2.putText(bar, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([out, bar])


def sheet(cells, cols=COLS):
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        while len(row) < cols:
            row.append(np.full_like(cells[0], 20))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/tracking_val_gt')
    ap.add_argument('--sequence', default='women_1_239')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    root = Path(args.root)
    seq = args.sequence
    man = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    s = next(x for x in man['sequences'] if x['sequence'] == seq)
    out_dir = Path(args.out) if args.out else root / 'audit' / seq
    out_dir.mkdir(parents=True, exist_ok=True)

    ann = json.loads((root / s['annotation_file_expected']).read_text(encoding='utf-8'))
    occ = occluded_flags(root / 'cvat_exports' / f'{seq}.xml')
    det = load_detector(root / s['preannotation_det'])

    picked, (t1, t2) = sample(ann['boxes'], occ, s['frame_count'])
    print(f'{len(picked)} observations sampled (seed {SEED}); '
          f'height terciles {t1:.1f} / {t2:.1f} px')

    img1 = root / 'sequences' / seq / 'img1'
    cells_gt, cells_det, records = [], [], []
    for b in picked:
        img = imread(img1 / f"{b['frame']:06d}.jpg")
        if img is None:
            continue
        best, bd = 0.0, None
        for d in det.get(b['frame'], []):
            v = iou(b['bbox'], d)
            if v > best:
                best, bd = v, d
        matched = bd if best >= MATCH_IOU else None
        st = b['stratum']
        tag = 'occ' if st['occlusion'] == 'occluded' else '   '
        label = (f"f{b['frame']} id{b['id']} {st['role'][:3]} "
                 f"{st['size'][:3]} {tag}")
        cells_gt.append(cell(img, b['bbox'], None, label, show_det=False))
        cells_det.append(cell(img, b['bbox'], matched,
                              label + (f" IoU{best:.2f}" if matched else " nodet"),
                              show_det=True))
        gw, gh = b['bbox'][2] - b['bbox'][0], b['bbox'][3] - b['bbox'][1]
        records.append({
            'frame': b['frame'], 'id': b['id'], 'role': b['role'],
            'stratum': st, 'gt_bbox': [round(v, 2) for v in b['bbox']],
            'gt_w': round(gw, 2), 'gt_h': round(gh, 2),
            'gt_aspect_h_over_w': round(gh / gw, 3) if gw else None,
            'detector_bbox': [round(v, 2) for v in matched] if matched else None,
            'iou_with_detector': round(best, 3) if matched else None,
            'annotator_marked_occluded': bool(occ.get((b['frame'], b['id']))),
        })

    gt_only = out_dir / f'{seq}_box_audit_gt_only.jpg'
    with_det = out_dir / f'{seq}_box_audit_with_detector.jpg'
    imwrite(gt_only, sheet(cells_gt))
    imwrite(with_det, sheet(cells_det))
    (out_dir / f'{seq}_box_audit_sample.json').write_text(json.dumps({
        'sequence': seq,
        'purpose': 'visual audit of GT box tightness',
        'authoritative': False,
        'alters_annotations': False,
        'sampling': {'seed': SEED, 'n': len(records),
                     'strata': ['time', 'size', 'role', 'occlusion'],
                     'height_terciles_px': [round(t1, 2), round(t2, 2)],
                     'occlusion_source': 'annotator occluded="1" in the CVAT '
                                         'export, not inferred'},
        'detector_box_status': 'comparison context only; the frozen detector is '
                               'not the authority on box convention',
        'observations': records,
    }, indent=1), encoding='utf-8')

    from collections import Counter
    for k in ('time', 'size', 'role', 'occlusion'):
        print(f'  {k:<10}{dict(Counter(r["stratum"][k] for r in records))}')
    print(f'\nGT only     : {gt_only}')
    print(f'With detector: {with_det}')
    print('Judge tightness on the GT-only sheet first.')


if __name__ == '__main__':
    main()
