#!/usr/bin/env python
"""
M2 -- extract image<->pitch correspondences from a broadcast-visible centre
circle, using only IFAB-fixed geometry (circle radius 9.15 m) that does not
require knowing this stadium's pitch length or width.

Method (exact, not an approximation):
  1. Detect the circle's white boundary pixels inside a hand-specified rough
     ROI and fit a conic (ellipse) to them.
  2. Detect the halfway-line's white pixels inside a hand-specified rough band
     and fit a straight line to them.
  3. Intersect the line with the conic -> 2 EXACT image points, corresponding
     to pitch (+9.15, 0) and (-9.15, 0) (pitch-local frame centred on the
     circle's own centre, x-axis along the halfway line).
  4. Compute the POLE of the halfway line with respect to the conic -> the
     EXACT image of the pitch centre (0, 0). This is a standard projective
     construction (pole-polar), not an approximation.
  5. Compute the direction CONJUGATE to the halfway line's direction with
     respect to the conic's quadratic form, construct the line through the
     pole in that direction, and intersect it with the conic -> 2 more EXACT
     image points, corresponding to pitch (0, +9.15) and (0, -9.15). Any two
     perpendicular diameters of a circle are conjugate diameters of the
     circle, and conjugacy is a projectively invariant relation, so this
     pair is exact under a full homography, not just an affine one.

Output: 4 non-collinear FIT points (the two diameters' 4 endpoints) + 1
independent HELD-OUT landmark (the centre) + many additional boundary pixels
usable for a "known-geometry reconstruction error" check (each must lie at
pitch-distance 9.15 m from the fitted centre once projected back).

Manual input is limited to two rough rectangular regions (circle ROI, line
band) -- no per-point clicking.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

R_CENTRE_CIRCLE_M = 9.15  # IFAB Laws of the Game, fixed regardless of pitch size


def white_pixels(img, roi, score_min=35, max_component_fill=0.55, min_component_span=8):
    """Pitch-line pixels within roi.

    Grass and line pixels separate cleanly on V-S ("whiteness": bright and
    desaturated), but players' kit is even whiter than a compressed, thin
    anti-aliased line, so a pure colour threshold also captures players.
    Connected-component shape filtering removes them: a line segment is a
    long, thin, sparsely-filled arc, a player's visible kit is a small, dense
    blob. `max_component_fill` rejects components whose pixels don't stretch
    thinly across their own bounding box (fill = pixel_count / bbox_area);
    `min_component_span` additionally drops components too small to be a
    piece of the boundary at all.
    """
    x0, y0, x1, y1 = roi
    sub = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    score = v - s
    mask = (score >= score_min).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        bw, bh, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        span = max(bw, bh)
        fill = area / max(bw * bh, 1)
        if span >= min_component_span and fill <= max_component_fill:
            keep[labels == i] = 1

    ys, xs = np.nonzero(keep)
    return np.stack([xs + x0, ys + y0], axis=1).astype(np.float64)


def _fit_ellipse_once(points):
    pts32 = points.astype(np.float32).reshape(-1, 1, 2)
    return cv2.fitEllipse(pts32)


def fit_conic(points, n_refine=4, keep_frac=0.7):
    """Algebraic conic Ax^2+Bxy+Cy^2+Dx+Ey+F=0 via cv2.fitEllipse -> 3x3 matrix.

    The candidate pixel set still contains some non-boundary contamination
    (player-kit edges pass the colour+shape filter in white_pixels() too).
    cv2.fitEllipse is not robust to that, so this refits iteratively: fit,
    score every point by its perpendicular distance to the fitted ellipse,
    keep the closest `keep_frac`, refit. A few rounds converge because the
    true boundary points vastly outnumber the scattered contamination.
    """
    pts = points
    for _ in range(n_refine):
        (cx, cy), (MA, ma), angle = _fit_ellipse_once(pts)
        a, b = MA / 2.0, ma / 2.0
        theta = np.deg2rad(angle)
        ct, st = np.cos(theta), np.sin(theta)
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        xr = ct * dx + st * dy
        yr = -st * dx + ct * dy
        # normalised radial residual: 0 exactly on the ellipse
        resid = np.abs((xr / a) ** 2 + (yr / b) ** 2 - 1.0)
        n_keep = max(20, int(len(pts) * keep_frac))
        order = np.argsort(resid)[:n_keep]
        pts = pts[order]
    (cx, cy), (MA, ma), angle = _fit_ellipse_once(pts)
    a, b = MA / 2.0, ma / 2.0
    theta = np.deg2rad(angle)
    ct, st = np.cos(theta), np.sin(theta)
    # conic in the ellipse's own centred/rotated frame: x'^2/a^2 + y'^2/b^2 = 1
    # frame change: x' = ct*(x-cx) + st*(y-cy); y' = -st*(x-cx) + ct*(y-cy)
    A = (ct / a) ** 2 + (st / b) ** 2
    B = 2 * ct * st * (1 / a ** 2 - 1 / b ** 2)
    C = (st / a) ** 2 + (ct / b) ** 2
    D = -2 * A * cx - B * cy
    E = -B * cx - 2 * C * cy
    F = A * cx ** 2 + B * cx * cy + C * cy ** 2 - 1
    conic = np.array([[A, B / 2, D / 2],
                      [B / 2, C, E / 2],
                      [D / 2, E / 2, F]])
    return conic, (cx, cy, MA, ma, angle), pts


def fit_line(points, n_refine=4, keep_frac=0.7):
    """Robust total-least-squares line -> homogeneous (a,b,c), a*x+b*y+c=0.

    Same iterative-trim rationale as fit_conic(): the candidate pixel set can
    include a few non-line points, so this refits and drops the largest
    perpendicular residuals a few times before returning.
    """
    pts = points
    for _ in range(n_refine):
        mean = pts.mean(axis=0)
        centred = pts - mean
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        direction = vt[0]
        normal = np.array([direction[1], -direction[0]])
        normal = normal / np.linalg.norm(normal)
        resid = np.abs((points - mean) @ normal)
        n_keep = max(15, int(len(points) * keep_frac))
        order = np.argsort(resid)[:n_keep]
        pts = points[order]
    mean = pts.mean(axis=0)
    centred = pts - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0]
    normal = np.array([direction[1], -direction[0]])
    normal = normal / np.linalg.norm(normal)
    c = -normal @ mean
    return np.array([normal[0], normal[1], c]), direction, mean


def line_conic_intersections(line, conic):
    """Solve for the (up to 2) real points where a line meets a conic."""
    a, b, c = line
    # Parametrise the line: P(t) = p0 + t*d, with p0 any point on it.
    if abs(b) > abs(a):
        p0 = np.array([0.0, -c / b])
        d = np.array([1.0, -a / b])
    else:
        p0 = np.array([-c / a, 0.0])
        d = np.array([-b / a, 1.0])
    d = d / np.linalg.norm(d)
    P0 = np.array([p0[0], p0[1], 1.0])
    D = np.array([d[0], d[1], 0.0])
    A = D @ conic @ D
    B = 2 * (P0 @ conic @ D)
    C = P0 @ conic @ P0
    disc = B * B - 4 * A * C
    if disc < 0:
        raise ValueError('line does not meet the conic in real points')
    sq = np.sqrt(disc)
    t1 = (-B + sq) / (2 * A)
    t2 = (-B - sq) / (2 * A)
    return p0 + t1 * d, p0 + t2 * d


def pole_of_line(line, conic):
    """Pole of a line w.r.t. a conic: pole (homogeneous) = C^-1 @ L."""
    L = np.array(line)
    pole_h = np.linalg.solve(conic, L)
    return pole_h[:2] / pole_h[2]


def conic_centre(conic):
    """Image of the pitch centre = pole of the line at infinity (0,0,1) w.r.t.
    the conic -- NOT the pole of the halfway line. A line through a central
    conic's centre is a diameter, and the pole of any diameter is a point at
    infinity (the two are pole-polar duals of each other: centre <-> line at
    infinity), so computing "pole of the halfway line" would itself return a
    point at infinity, not the centre.
    """
    return pole_of_line((0.0, 0.0, 1.0), conic)


def conjugate_direction(direction, conic):
    Q = conic[:2, :2]
    v = Q @ direction
    conj = np.array([v[1], -v[0]])
    return conj / np.linalg.norm(conj)


def line_through_point(point, direction):
    normal = np.array([direction[1], -direction[0]])
    normal = normal / np.linalg.norm(normal)
    c = -normal @ point
    return np.array([normal[0], normal[1], c])


def extract(image_path, circle_roi, line_band, out_path, order_top_is_positive=True,
           reference_line='diameter'):
    """reference_line:
      'diameter'  -- the reference line (e.g. the halfway line) passes through
                     the circle itself; its two intersections with the conic
                     are used directly, as pitch (+-9.15, 0).
      'direction' -- the reference line does NOT pass through the circle (e.g.
                     a touchline, parallel to the missing halfway line's
                     perpendicular axis); only its DIRECTION is used. Both
                     axes are then constructed as lines through the
                     independently-computed conic centre (which needs no line
                     at all -- see conic_centre()), one along that direction,
                     one along its conjugate.
    """
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)

    circle_pts = white_pixels(img, circle_roi)
    if len(circle_pts) < 30:
        raise SystemExit(f'too few circle boundary pixels: {len(circle_pts)}')
    conic, ellipse_params, clean_circle_pts = fit_conic(circle_pts)
    centre = conic_centre(conic)

    line_pts = white_pixels(img, line_band)
    if len(line_pts) < 20:
        raise SystemExit(f'too few reference-line pixels: {len(line_pts)}')
    line, direction, _ = fit_line(line_pts)

    if reference_line == 'diameter':
        p1, p2 = line_conic_intersections(line, conic)
        axis_a_direction = direction
    else:
        axis_a_line = line_through_point(centre, direction)
        p1, p2 = line_conic_intersections(axis_a_line, conic)
        axis_a_direction = direction

    # order by y: smaller y (higher in image) = "far" side = +9.15 by convention
    if order_top_is_positive:
        (p_pos, p_neg) = (p1, p2) if p1[1] < p2[1] else (p2, p1)
    else:
        (p_pos, p_neg) = (p1, p2) if p1[1] > p2[1] else (p2, p1)

    conj_dir = conjugate_direction(axis_a_direction, conic)
    conj_line = line_through_point(centre, conj_dir)
    p3, p4 = line_conic_intersections(conj_line, conic)
    # order by x: smaller x (left in image) as the second axis's "+" side
    (p3_pos, p3_neg) = (p3, p4) if p3[0] < p4[0] else (p4, p3)

    result = {
        'image': str(image_path),
        'circle_roi': circle_roi,
        'line_band': line_band,
        'n_circle_boundary_px': int(len(circle_pts)),
        'n_line_px': int(len(line_pts)),
        'ellipse_fit': {'cx': ellipse_params[0], 'cy': ellipse_params[1],
                        'major_axis': ellipse_params[2], 'minor_axis': ellipse_params[3],
                        'angle_deg': ellipse_params[4]},
        'reference_line_mode': reference_line,
        'reference_line': {'a': line[0], 'b': line[1], 'c': line[2]},
        # local pitch frame centred on the circle: axis_a is along the
        # reference line's own direction (the halfway line itself in
        # 'diameter' mode, or the direction it supplies in 'direction' mode);
        # axis_b is its conic-conjugate (== perpendicular, for a true circle).
        # Which physical pitch axis is "a" vs "b" is a local labelling choice
        # and does not affect the homography fit.
        'fit_points_image': {
            'axis_a(+9.15)': [float(p_pos[0]), float(p_pos[1])],
            'axis_a(-9.15)': [float(p_neg[0]), float(p_neg[1])],
            'axis_b(+9.15)': [float(p3_pos[0]), float(p3_pos[1])],
            'axis_b(-9.15)': [float(p3_neg[0]), float(p3_neg[1])],
        },
        'held_out_landmark_image': {'(0,0)': [float(centre[0]), float(centre[1])]},
        # In 'diameter' mode the centre is derived from the conic ALONE and
        # never touches the fit points, so it is a genuine held-out check. In
        # 'direction' mode the centre is used to CONSTRUCT the fit points
        # (both axis lines pass through it), so it is not independent of them
        # and must not be reported as if it were.
        'held_out_landmark_independent': reference_line == 'diameter',
        'reconstruction_check_pixels_raw_n': int(len(circle_pts)),
        'reconstruction_check_pixels': clean_circle_pts.tolist(),
        'conic_matrix': conic.tolist(),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=1), encoding='utf-8')
    print('written:', out_path)
    print('fit points (image px):', result['fit_points_image'])
    print('held-out centre (image px):', result['held_out_landmark_image'])
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', required=True)
    ap.add_argument('--circle-roi', required=True, help='x0,y0,x1,y1')
    ap.add_argument('--line-band', required=True, help='x0,y0,x1,y1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference-line', choices=['diameter', 'direction'], default='diameter')
    args = ap.parse_args()
    croi = tuple(int(v) for v in args.circle_roi.split(','))
    lband = tuple(int(v) for v in args.line_band.split(','))
    extract(args.image, croi, lband, args.out, reference_line=args.reference_line)
