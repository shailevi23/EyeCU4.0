#!/usr/bin/env python
"""
M2 -- fit and validate an image->pitch homography from the circle-derived
correspondences, and write the frozen calibration artifact.

Fit set (4 points, non-collinear, well-distributed around the circle):
    axis_a(+9.15), axis_a(-9.15), axis_b(+9.15), axis_b(-9.15)

Validation (never used in fitting):
    1. held-out landmark: the pitch centre (0,0), known exactly by
       construction (see tools/m2_circle_correspondences.py) -- reprojected
       through the fitted H and compared in METRES.
    2. known-geometry reconstruction check: every other detected boundary
       pixel of the same circle (NOT among the 4 fit points) is mapped
       through H^-1 into pitch space and its distance from the fitted centre
       is compared against the true radius, 9.15 m. This uses dozens of
       independent constraints, not just one point.

VALIDATION ONLY -- geometric, never player speed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

R = 9.15


def fit_and_validate(corr_path, out_path, segment_meta):
    d = json.loads(Path(corr_path).read_text(encoding='utf-8'))
    fp = d['fit_points_image']
    image_pts = np.array([fp['axis_a(+9.15)'], fp['axis_a(-9.15)'],
                          fp['axis_b(+9.15)'], fp['axis_b(-9.15)']], dtype=np.float64)
    pitch_pts = np.array([[R, 0], [-R, 0], [0, R], [0, -R]], dtype=np.float64)

    H, mask = cv2.findHomography(image_pts, pitch_pts, method=0)  # exact 4-pt solve
    if H is None:
        raise SystemExit('homography fit failed (degenerate points)')
    Hinv = np.linalg.inv(H)

    def to_pitch(img_xy):
        p = np.array([img_xy[0], img_xy[1], 1.0])
        q = H @ p
        return q[:2] / q[2]

    # ---- validation 1: held-out centre landmark
    centre_img = d['held_out_landmark_image']['(0,0)']
    centre_pitch = to_pitch(centre_img)
    centre_error_m = float(np.hypot(*centre_pitch))  # true value is (0,0)

    # ---- validation 2: known-geometry reconstruction (boundary pixels)
    boundary = np.array(d['reconstruction_check_pixels'])
    fit_set = set(tuple(p) for p in image_pts.tolist())
    holdout_boundary = np.array([p for p in boundary.tolist() if tuple(p) not in fit_set])
    radii = []
    for x, y in holdout_boundary:
        px, py = to_pitch((x, y))
        radii.append(float(np.hypot(px, py)))
    radii = np.array(radii)
    radius_error = np.abs(radii - R)

    # ---- numerical stability
    cond = float(np.linalg.cond(H))
    det = float(np.linalg.det(H))

    artifact = {
        **segment_meta,
        'coordinate_convention': (
            'pitch coordinates in metres, local frame centred on the centre '
            'circle; axis_a / axis_b are an arbitrary orthogonal local basis '
            '(NOT globally along/across the pitch -- see reference_line_mode '
            'in the correspondences file), origin = image of the pitch '
            'centre spot as reconstructed from the circle conic'),
        'player_ground_point_convention': 'bottom-centre (ground contact point) of the player bbox',
        'fit_points_image': fp,
        'fit_points_pitch_m': {k: v.tolist() for k, v in
                               zip(fp.keys(), pitch_pts)},
        'homography_image_to_pitch': H.tolist(),
        'homography_pitch_to_image': Hinv.tolist(),
        'validation': {
            'held_out_landmark': {
                'point': '(0,0) pitch centre',
                'image_px': centre_img,
                'reprojected_pitch_m': centre_pitch.tolist(),
                'error_m': round(centre_error_m, 4),
            },
            'known_geometry_reconstruction': {
                'description': ('every detected circle-boundary pixel not used '
                                'in fitting, reprojected to pitch space; error is '
                                'deviation of its distance-from-centre from the '
                                'true radius 9.15 m'),
                'n_points': int(len(holdout_boundary)),
                'radius_error_m_mean': round(float(radius_error.mean()), 4),
                'radius_error_m_median': round(float(np.median(radius_error)), 4),
                'radius_error_m_p95': round(float(np.percentile(radius_error, 95)), 4),
                'radius_error_m_max': round(float(radius_error.max()), 4),
            },
            'numerical_stability': {
                'condition_number': cond,
                'determinant': det,
                'degenerate': bool(cond > 1e8 or abs(det) < 1e-12),
            },
        },
    }
    art_bytes = json.dumps(artifact, indent=1, sort_keys=True).encode('utf-8')
    artifact['calibration_artifact_sha256'] = hashlib.sha256(art_bytes).hexdigest()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(artifact, indent=1), encoding='utf-8')
    print('written:', out_path)
    print('held-out centre error (m):', artifact['validation']['held_out_landmark']['error_m'])
    print('reconstruction radius error (m):',
          artifact['validation']['known_geometry_reconstruction'])
    print('numerical stability:', artifact['validation']['numerical_stability'])
    return artifact


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--correspondences', required=True)
    ap.add_argument('--sequence', required=True)
    ap.add_argument('--frame-range', required=True, help='start,end (1-based, inclusive)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    start, end = (int(v) for v in args.frame_range.split(','))
    meta = {'sequence': args.sequence, 'frame_range_1based': [start, end]}
    fit_and_validate(args.correspondences, args.out, meta)
