#!/usr/bin/env python
"""
Demonstrate the SoccerTrack pixel <-> metre transform on real annotation points.

This is a demonstration, not an integration: EyeCU's calibration pipeline is not
touched. The point is to show that the transform in the downloaded RAW package
is complete and correct enough to be worth having, and to put a number on how
correct.

Two independent routes are drawn on the same frame:

    pitch metres --homography--> undistorted canvas --mapx--> video pixel
    pitch metres --fisheye projectPoints(K, D, rvec, tvec)--> video pixel

They come from different files and agree to a few pixels, which is the strongest
available evidence that the shipped calibration is internally consistent.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / 'experiments' / 'soccertrack_audit'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--match', default='128058')
    ap.add_argument('--video', default='EyeCU_external_data/SoccerTrackV2/videos/128058_panorama_1st_half-002.mp4')
    ap.add_argument('--frame', type=int, default=30000)
    args = ap.parse_args()
    import cv2

    D = AUDIT / 'extracted' / 'raw' / args.match
    H = np.load(D / f'{args.match}_homography.npy')
    mapx = np.asarray(np.load(D / f'{args.match}_mapx.npy'))
    mh, mw = mapx.shape[:2]
    z = np.load(D / f'{args.match}_camera_intrinsics.npz', allow_pickle=True)
    K, dist = z['K'], z['D']
    rvec = np.array(z['rvecs'][0], float).reshape(3, 1)
    tvec = np.array(z['tvecs'][0], float).reshape(3, 1)
    kp = json.loads((D / f'{args.match}_keypoints.json').read_text(encoding='utf-8'))

    def via_homography(pts):
        out = []
        for X, Y in pts:
            q = H @ np.array([X, Y, 1.0])
            u, v = q[0] / q[2], q[1] / q[2]
            out.append(tuple(mapx[int(v), int(u)].astype(float))
                       if 0 <= u < mw and 0 <= v < mh else (np.nan, np.nan))
        return np.array(out)

    def via_camera(pts):
        obj = np.c_[np.array(pts, float), np.zeros(len(pts))].reshape(-1, 1, 3)
        return cv2.fisheye.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2)

    cap = cv2.VideoCapture(str(REPO / args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit('could not read frame')
    vis = img.copy()

    # pitch grid every 5 m, drawn through the homography route
    for X in np.arange(0, 105.1, 5):
        pts = via_homography([(X, y) for y in np.arange(0, 68.1, 1.0)])
        good = pts[~np.isnan(pts).any(1)].astype(np.int32)
        if len(good) > 1:
            cv2.polylines(vis, [good], False, (255, 200, 0), 1, cv2.LINE_AA)
    for Y in np.arange(0, 68.1, 5):
        pts = via_homography([(x, Y) for x in np.arange(0, 105.1, 1.0)])
        good = pts[~np.isnan(pts).any(1)].astype(np.int32)
        if len(good) > 1:
            cv2.polylines(vis, [good], False, (255, 200, 0), 1, cv2.LINE_AA)

    # the 65 shipped correspondences: clicked position vs both predictions
    P = np.array([ast.literal_eval(k) for k in kp], float)
    I = np.array(list(kp.values()), float)
    ph = via_homography(P)
    pc = via_camera(P)
    for (u, v), (a, b), (c, d) in zip(I, ph, pc):
        cv2.circle(vis, (int(u), int(v)), 9, (255, 255, 255), 2)     # shipped
        if not np.isnan(a):
            cv2.drawMarker(vis, (int(a), int(b)), (0, 0, 255),
                           cv2.MARKER_CROSS, 14, 2)                  # homography
        cv2.drawMarker(vis, (int(c), int(d)), (0, 255, 0),
                       cv2.MARKER_TILTED_CROSS, 12, 2)               # camera model

    eh = np.hypot(*(ph - I).T)
    ec = np.hypot(*(pc - I).T)
    agree = np.hypot(*(ph - pc).T)
    stats = {
        'match': args.match, 'frame': args.frame,
        'keypoints': len(P),
        'homography_route_error_px': {'median': float(np.nanmedian(eh)),
                                      'p90': float(np.nanpercentile(eh, 90))},
        'camera_model_route_error_px': {'median': float(np.median(ec)),
                                        'p90': float(np.percentile(ec, 90))},
        'routes_agree_px': {'median': float(np.nanmedian(agree)),
                            'p90': float(np.nanpercentile(agree, 90))},
        'legend': {'white circle': 'shipped keypoint (clicked)',
                   'red cross': 'homography -> mapx',
                   'green cross': 'fisheye projectPoints(K, D, rvec, tvec)'},
    }
    print(json.dumps(stats, indent=1))

    cv2.rectangle(vis, (0, 0), (1200, 96), (0, 0, 0), -1)
    cv2.putText(vis, f'{args.match} f{args.frame}  white=shipped keypoint  '
                     f'red=homography  green=camera model',
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f'median error {np.nanmedian(eh):.1f} px / '
                     f'{np.median(ec):.1f} px   routes agree to '
                     f'{np.nanmedian(agree):.1f} px',
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    (AUDIT / 'contact_sheets').mkdir(parents=True, exist_ok=True)
    dst = AUDIT / 'contact_sheets' / f'calibration_{args.match}_f{args.frame}.jpg'
    cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(str(dst))
    (AUDIT / 'reports' / f'calibration_{args.match}.json').write_text(
        json.dumps(stats, indent=1), encoding='utf-8')
    print(f'wrote {dst.name}')


if __name__ == '__main__':
    main()
