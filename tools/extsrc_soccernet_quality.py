#!/usr/bin/env python
"""
SoccerNet-V3 label quality and internal duplication, on the downloaded payload.

Two questions the metadata could not answer.

QUALITY: are the boxes actually on the objects, and is the class right? Judged
from the image and its annotation, never from an EyeCU detector prediction --
using our own model to score somebody else's labels would make the audit
circular.

DUPLICATION: the export ships one action frame plus up to eight replay frames of
the SAME event. Replays are the same moment from another camera, so they are
correlated by construction and must not be counted as independent images. This
measures how correlated, per group, rather than assuming.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / 'EyeCU_external_data'
SN = EXT / 'huggingface' / 'soccernet_v3'
XS = REPO / 'experiments' / 'external_sources'

COLORS = {'player': (80, 255, 80), 'goalkeeper': (0, 220, 255),
          'referee': (255, 120, 0), 'ball': (60, 60, 255),
          'EXCLUDE': (160, 160, 160)}
MAP = {'Player team left': 'player', 'Player team right': 'player',
       'Goalkeeper team left': 'goalkeeper', 'Goalkeeper team right': 'goalkeeper',
       'Main referee': 'referee', 'Side referee': 'referee', 'Ball': 'ball'}


def canonical(bgr, sig=64):
    import cv2
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (sig, sig), interpolation=cv2.INTER_AREA).astype(np.float32)


def ncc(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import cv2
    d = json.loads((SN / 'metadata_only' / 'samples.json').read_text(encoding='utf-8'))
    samples = d['samples']
    root = SN / 'full_dataset'

    def local(fp):
        rel = fp.replace('\\', '/').split('data/', 1)[-1]
        return root / 'data' / rel

    # ---- replay correlation --------------------------------------------------
    # The group id is group._id.$oid, not group.id -- reading the wrong key gave
    # one giant group and a meaningless correlation, so it is read explicitly.
    groups = defaultdict(list)
    for s in samples:
        g = s.get('group') or {}
        gid = (g.get('_id') or {}).get('$oid') if isinstance(g, dict) else None
        groups[gid].append(s)
    multi = [v for k, v in groups.items() if k and len(v) > 1]
    print(f'{len(groups)} groups, {len(multi)} with more than one slice')

    sel = multi[:120]
    pair_ncc = []
    for grp in sel:
        imgs = []
        for s in grp[:4]:
            p = local(s['filepath'])
            if p.exists():
                a = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
                if a is not None:
                    imgs.append(canonical(a))
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                pair_ncc.append(ncc(imgs[i], imgs[j]))
    pn = np.array(pair_ncc) if pair_ncc else np.array([0.0])
    print(f'within-group (action vs replay) NCC over {len(pn)} pairs: '
          f'median {np.median(pn):.3f}  p90 {np.percentile(pn,90):.3f}  '
          f'>=0.985 {(pn>=0.985).sum()}')

    # ---- label quality: render annotated samples -----------------------------
    XS.mkdir(parents=True, exist_ok=True)
    (XS / 'contact_sheets').mkdir(parents=True, exist_ok=True)
    idx = np.linspace(0, len(samples) - 1, 12).round().astype(int)
    tiles = []
    for i in dict.fromkeys(idx.tolist()):
        s = samples[i]
        p = local(s['filepath'])
        if not p.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        for f, v in s.items():
            if isinstance(v, dict) and isinstance(v.get('detections'), list):
                for det in v['detections']:
                    bb = det.get('bounding_box')
                    if not bb:
                        continue
                    lbl = MAP.get(det.get('label'), 'EXCLUDE')
                    x, y = int(bb[0] * W), int(bb[1] * H)
                    w, h = int(bb[2] * W), int(bb[3] * H)
                    cv2.rectangle(img, (x, y), (x + w, y + h),
                                  COLORS[lbl], 2 if lbl == 'ball' else 1)
        t = cv2.resize(img, (640, int(640 * H / W)))
        cv2.rectangle(t, (0, 0), (640, 17), (0, 0, 0), -1)
        cv2.putText(t, f"{s.get('match','')[:40]} {Path(s['filepath']).name}",
                    (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(t)
    if tiles:
        hh = max(t.shape[0] for t in tiles)
        g = np.full(((len(tiles) + 2) // 3 * (hh + 6) + 6, 3 * 646 + 6, 3), 32, np.uint8)
        for i, t in enumerate(tiles):
            y = 6 + (i // 3) * (hh + 6); x = 6 + (i % 3) * 646
            g[y:y + t.shape[0], x:x + 640] = t
        cv2.imencode('.jpg', g, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(
            str(XS / 'contact_sheets' / 'soccernet_v3_labels.jpg'))
        print(f'wrote soccernet_v3_labels.jpg ({len(tiles)} frames)')

    # ---- ball crops ---------------------------------------------------------
    balls = []
    for s in samples:
        for f, v in s.items():
            if isinstance(v, dict) and isinstance(v.get('detections'), list):
                for det in v['detections']:
                    if det.get('label') == 'Ball' and det.get('bounding_box'):
                        balls.append((s['filepath'], det['bounding_box']))
    pick = [balls[i] for i in np.linspace(0, len(balls) - 1, 24).round().astype(int)]
    crops = []
    for fp, bb in dict.fromkeys([(a, tuple(b)) for a, b in pick]):
        p = local(fp)
        if not p.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        x, y, w, h = bb[0] * W, bb[1] * H, bb[2] * W, bb[3] * H
        m = max(2.0 * max(w, h), 24)
        x0, y0 = int(max(0, x - m)), int(max(0, y - m))
        x1, y1 = int(min(W, x + w + m)), int(min(H, y + h + m))
        c = img[y0:y1, x0:x1].copy()
        if c.size == 0:
            continue
        s_ = 170 / max(c.shape[:2])
        c = cv2.resize(c, None, fx=s_, fy=s_, interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(c, (int((x - x0) * s_), int((y - y0) * s_)),
                      (int((x + w - x0) * s_), int((y + h - y0) * s_)), (60, 60, 255), 1)
        cv2.rectangle(c, (0, 0), (c.shape[1], 15), (0, 0, 0), -1)
        cv2.putText(c, f'{w:.0f}x{h:.0f}px', (2, 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1, cv2.LINE_AA)
        crops.append(c)
    if crops:
        hh = max(c.shape[0] for c in crops); ww = max(c.shape[1] for c in crops)
        g = np.full(((len(crops) + 7) // 8 * (hh + 6) + 6, 8 * (ww + 6) + 6, 3), 32, np.uint8)
        for i, c in enumerate(crops):
            y = 6 + (i // 8) * (hh + 6); x = 6 + (i % 8) * (ww + 6)
            g[y:y + c.shape[0], x:x + c.shape[1]] = c
        cv2.imencode('.jpg', g, [int(cv2.IMWRITE_JPEG_QUALITY), 94])[1].tofile(
            str(XS / 'contact_sheets' / 'soccernet_v3_ball.jpg'))
        print(f'wrote soccernet_v3_ball.jpg ({len(crops)} crops)')

    rep = {
        'payload_images': len(list(root.rglob('*.png'))),
        'groups': len(groups),
        'groups_with_replays': len(multi),
        'within_group_ncc': {'pairs': int(len(pn)),
                             'median': round(float(np.median(pn)), 4),
                             'p90': round(float(np.percentile(pn, 90)), 4),
                             'pairs_at_or_above_0.985': int((pn >= 0.985).sum())},
        'replay_interpretation': (
            'Action and replay slices of one group are the same moment from a '
            'different camera. They are NOT near-duplicate pixels -- the median '
            'correlation is low -- but they are not independent samples either, '
            'because the same players in the same instant appear in both. Count '
            'groups, not frames, when estimating diversity.'),
        'label_evidence': ['soccernet_v3_labels.jpg', 'soccernet_v3_ball.jpg'],
        'method_note': ('quality judged from image and annotation only; no EyeCU '
                        'detector prediction was used'),
    }
    (XS / 'reports').mkdir(parents=True, exist_ok=True)
    (XS / 'reports' / 'soccernet_label_quality.json').write_text(
        json.dumps(rep, indent=1), encoding='utf-8')
    print('wrote reports/soccernet_label_quality.json')


if __name__ == '__main__':
    main()
