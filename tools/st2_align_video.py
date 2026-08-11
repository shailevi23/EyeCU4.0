#!/usr/bin/env python
"""
Prove -- or disprove -- that the one downloaded video aligns with its annotations.

Filenames claim the video is match 128058. That is not evidence. Three separate
things have to line up before a single GSR box can be drawn on a video frame:

  1. GEOMETRY. GSR declares 3840x1504 images; the downloaded panorama is
     4096x1080. Those are not the same picture, and the ratio is not a simple
     scale (3840/4096 = 0.9375 but 1504/1080 = 1.3926).
  2. TIME.  GSR frames are 1..69251; the video has 68975. RAW metadata says the
     half spans recording frames 251..69501 at 25 fps.
  3. CONTENT. Even with 1 and 2 agreed, the boxes have to land on players.

Alignment is therefore SEARCHED, not assumed: for a set of probe frames the
tool scores every candidate frame offset by how much of the annotated box area
lands on non-grass pixels, and reports the profile. A flat profile means no
alignment was found, and that is a reportable result rather than a failure to
hide.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'experiments' / 'soccertrack_audit'

OBJ_START = b'\n        {\n'


def annotations_for_frames(path: Path, frames, seq_length):
    """Pull annotations for specific frame numbers out of a 2.7 GB file.

    A proportional seek into the annotations array was tried first and was not
    reliable: it silently returned 15 of 22 boxes for one frame and none at all
    for another, because an object boundary found by pattern is not guaranteed
    to be an object boundary. This reads sequentially from the start of the
    array instead -- slower, but every object is seen exactly once. Probe frames
    should therefore be chosen early in the half.
    """
    want = sorted(set(frames))
    got = {f: [] for f in want}
    hi = max(want)
    sys.path.insert(0, str(REPO / 'tools'))
    from st2_gsr_scan import iter_objects
    for a in iter_objects(path, b'"annotations"'):
        iid = str(a.get('image_id', ''))
        if not iid:
            continue
        fr = int(iid[-6:])
        if fr in got and a.get('category_id') in (1, 2, 3, 4):
            got[fr].append(a)
        if fr > hi:
            break
    return got


def read_frame(video: Path, index):
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def grass_mask(bgr):
    b, g, r = bgr[:, :, 0].astype(np.int16), bgr[:, :, 1].astype(np.int16), \
        bgr[:, :, 2].astype(np.int16)
    return (g > b + 12) & (g > r + 12)


def score_boxes(img, boxes):
    """Fraction of annotated box area that is NOT grass -- i.e. is a person."""
    m = grass_mask(img)
    h, w = m.shape
    tot, nong = 0, 0
    for x, y, bw, bh in boxes:
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
        if x1 <= x0 or y1 <= y0:
            continue
        sub = m[y0:y1, x0:x1]
        tot += sub.size
        nong += int((~sub).sum())
    return (nong / tot) if tot else 0.0


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gsr', default='EyeCU_external_data/SoccerTrackV2/gsr/128058_1st-015.json')
    ap.add_argument('--video', default='EyeCU_external_data/SoccerTrackV2/videos/128058_panorama_1st_half-002.mp4')
    ap.add_argument('--match', default='128058')
    ap.add_argument('--probe-frames', default='20000,35000,50000')
    ap.add_argument('--offsets', default='-300,-250,-200,-150,-100,-50,-10,0,10,50,100,150,200,250,300')
    args = ap.parse_args()
    import cv2

    gsr = REPO / args.gsr
    video = REPO / args.video
    raw = REPO / 'experiments/soccertrack_audit/extracted/raw' / args.match
    H = np.load(raw / f'{args.match}_homography.npy')
    mapx = np.asarray(np.load(raw / f'{args.match}_mapx.npy'))
    mh, mw = mapx.shape[:2]

    info = json.loads(open(gsr, 'rb').read(2000).split(b'"images"')[0]
                      .rstrip().rstrip(b',').decode() + '}')['info']
    seq = int(info['seq_length'])
    cap = cv2.VideoCapture(str(video))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    report = {'gsr': gsr.name, 'video': video.name,
              'gsr_declared_image_size': [3840, 1504],
              'video_size': [vw, vh], 'video_frames': vn,
              'gsr_seq_length': seq,
              'geometry_note': ('GSR declares 3840x1504, video is '
                                f'{vw}x{vh}; ratios differ per axis '
                                f'({vw/3840:.4f} vs {vh/1504:.4f}), so the GSR '
                                'image is not this video rescaled'),
              'probes': []}
    print(json.dumps({k: v for k, v in report.items() if k != 'probes'}, indent=1))

    probes = [int(x) for x in args.probe_frames.split(',')]
    offsets = [int(x) for x in args.offsets.split(',')]
    anns = annotations_for_frames(gsr, probes, seq)

    for pf in probes:
        A = anns.get(pf) or []
        if not A:
            print(f'frame {pf}: no annotations recovered')
            continue
        # route 1: the declared bbox_image, rescaled per axis to the video
        sx, sy = vw / 3840.0, vh / 1504.0
        b_img = [(a['bbox_image']['x'] * sx, a['bbox_image']['y'] * sy,
                  a['bbox_image']['w'] * sx, a['bbox_image']['h'] * sy)
                 for a in A if 'bbox_image' in a]
        # route 2: bbox_pitch (metres, centred origin) -> H -> undistort map
        pts = []
        for a in A:
            bp = a.get('bbox_pitch') or {}
            if 'x_bottom_middle' not in bp:
                continue
            X = bp['x_bottom_middle'] + 52.5
            Y = bp['y_bottom_middle'] + 34.0
            p = H @ np.array([X, Y, 1.0])
            u, v = p[0] / p[2], p[1] / p[2]
            if 0 <= u < mw and 0 <= v < mh:
                pts.append(tuple(mapx[int(v), int(u)].astype(float)))
        rec = {'frame': pf, 'annotations': len(A),
               'boxes_from_bbox_image': len(b_img),
               'points_from_bbox_pitch': len(pts), 'offset_scores': {},
               'offset_scores_pitch_route': {}}
        # A small box around each projected pitch point, so the two routes are
        # scored the same way and can be compared directly.
        b_pitch = [(u - 20, v - 45, 40, 50) for u, v in pts]
        for off in offsets:
            idx = pf + off
            if not (0 <= idx < vn):
                continue
            img = read_frame(video, idx)
            if img is None:
                continue
            rec['offset_scores'][str(off)] = round(score_boxes(img, b_img), 4)
            rec['offset_scores_pitch_route'][str(off)] = round(
                score_boxes(img, b_pitch), 4)
        for key, label in (('offset_scores', 'bbox_image'),
                           ('offset_scores_pitch_route', 'bbox_pitch->H->map')):
            s = rec[key]
            if not s:
                continue
            best = max(s, key=s.get)
            rec[f'best_offset_{label}'] = best
            rec[f'best_score_{label}'] = s[best]
            rec[f'spread_{label}'] = round(max(s.values()) - min(s.values()), 4)
            print(f'frame {pf} [{label}]: {len(A)} anns, best offset {best} '
                  f'score {s[best]} spread {rec[f"spread_{label}"]}')
            print('   ', s)
        best = rec.get('best_offset_bbox_pitch->H->map') or \
            rec.get('best_offset_bbox_image') or '0'
        report['probes'].append(rec)

        # render evidence at the best offset
        img = read_frame(video, pf + int(best or 0))
        if img is not None:
            vis = img.copy()
            for x, y, w_, h_ in b_img:
                cv2.rectangle(vis, (int(x), int(y)), (int(x + w_), int(y + h_)),
                              (0, 255, 255), 2)
            for u, v in pts:
                cv2.circle(vis, (int(u), int(v)), 6, (0, 0, 255), 2)
            (OUT / 'contact_sheets').mkdir(parents=True, exist_ok=True)
            dst = OUT / 'contact_sheets' / f'align_{args.match}_f{pf}.jpg'
            cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(str(dst))
            # and a zoomed crop where the boxes are
            if b_img:
                cx = int(np.median([x + w_ / 2 for x, _, w_, _ in b_img]))
                cy = int(np.median([y + h_ / 2 for _, y, _, h_ in b_img]))
                x0, y0 = max(0, cx - 500), max(0, cy - 200)
                crop = vis[y0:y0 + 400, x0:x0 + 1000]
                if crop.size:
                    d2 = OUT / 'contact_sheets' / f'align_{args.match}_f{pf}_zoom.jpg'
                    cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(str(d2))

    (OUT / 'reports').mkdir(parents=True, exist_ok=True)
    (OUT / 'reports' / f'alignment_{args.match}.json').write_text(
        json.dumps(report, indent=1), encoding='utf-8')
    print(f'\nwrote reports/alignment_{args.match}.json')


if __name__ == '__main__':
    main()
