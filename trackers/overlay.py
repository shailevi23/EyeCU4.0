"""
Rendering-only helpers for the annotated output video.

Presentation layer exclusively: every function here consumes already-computed
tracks/state (bboxes, team colors, ball selector state, possession flags) and
draws pixels. Nothing in this module reads a model, changes a threshold, or
alters detection/tracking/selection/possession results -- it only decides how
existing results are drawn, and how that drawing scales across resolutions.

Two independent knobs live outside this module:
  - overlay_mode ('viewer' | 'debug'), threaded through the callers in
    trackers/football_tracker.py and trackers/camera_movement.py.
  - the UI scale below, derived only from frame_width/frame_height.
"""

import cv2
import numpy as np

REFERENCE_HEIGHT = 720.0
MIN_SCALE = 0.55
MAX_SCALE = 2.25
SAFE_MARGIN_FRAC = 0.028  # ~2.8% of frame height, per spec's 2.5-3% band
FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_ui_scale(frame_width, frame_height):
    """
    One small style object all drawing dimensions derive from, instead of
    scattered magic numbers. Scale is frame_height / 720, clamped so 360p,
    720p, 1080p and higher all stay readable without any resolution-specific
    branch.
    """
    raw_scale = frame_height / REFERENCE_HEIGHT
    scale = max(MIN_SCALE, min(MAX_SCALE, raw_scale))
    margin = max(4, int(round(frame_height * SAFE_MARGIN_FRAC)))
    return {
        'scale': scale,
        'frame_width': frame_width,
        'frame_height': frame_height,
        # Polish pass v3: pill/marker/ball coefficients cut ~11-15% from the
        # v2 values (0.42/4/17/6/9) so 360p in particular reads lighter;
        # readability floors (max(...)) are unchanged.
        'font_scale': round(0.38 * scale, 3),
        'font_thickness': max(1, round(1 * scale)),
        'stroke': max(1, round(2 * scale)),
        'marker_radius': max(2, round(3.5 * scale)),
        'ball_ring_radius': max(4, round(7.8 * scale)),
        'pill_height': max(11, round(15 * scale)),
        'pill_pad_x': max(4, round(5 * scale)),
        'corner_radius': max(2, round(5 * scale)),
        # Short connector between the feet marker and its ID pill, not a long
        # stem -- the pill sits close to the feet, below by default.
        'label_gap': max(2, round(4 * scale)),
        'margin': margin,
        'hud_line_h': max(12, round(19 * scale)),
        'hud_pad': max(4, round(8 * scale)),
        'hud_max_w_frac': 0.24,
    }


def draw_text_with_outline(frame, text, org, font_scale, color, thickness,
                            outline_color=(0, 0, 0)):
    """Light text + dark outline: readable over a moving green pitch."""
    cv2.putText(frame, text, org, FONT, font_scale, outline_color,
                thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, font_scale, color, thickness,
                cv2.LINE_AA)


def draw_rounded_rect(frame, pt1, pt2, color, radius, alpha=1.0):
    """Filled rounded rectangle; alpha<1 composites onto frame in place."""
    x1, y1 = pt1
    x2, y2 = pt2
    if x2 <= x1 or y2 <= y1:
        return
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    target = frame if alpha >= 1.0 else frame.copy()
    if radius <= 0:
        cv2.rectangle(target, (x1, y1), (x2, y2), color, -1)
    else:
        cv2.rectangle(target, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(target, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(target, (cx, cy), radius, color, -1)
    if alpha < 1.0:
        cv2.addWeighted(target, alpha, frame, 1 - alpha, 0, frame)


def draw_dashed_circle(frame, center, radius, color, thickness, n_dashes=10):
    """Broken-ring style, used for recovered_low_conf ball state."""
    step = 360.0 / n_dashes
    for i in range(n_dashes):
        if i % 2 == 0:
            a0 = i * step
            a1 = a0 + step
            cv2.ellipse(frame, center, (radius, radius), 0, a0, a1, color,
                        thickness)


def _boxes_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _clamp_box_to_frame(x1, y1, w, h, style):
    """
    Clamp a w x h box so it lies fully within [margin, dim - margin] on both
    axes. This is the only thing that may move a label -- markers and
    bbox/track coordinates are never touched.
    """
    fw, fh = style['frame_width'], style['frame_height']
    margin = style['margin']
    lo_x, hi_x = margin, fw - margin - w
    lo_y, hi_y = margin, fh - margin - h
    # If the safe area is narrower than the label (tiny frame), fall back to
    # centering in-frame rather than producing an inverted range.
    x1 = min(max(x1, lo_x), hi_x) if hi_x >= lo_x else (fw - w) / 2
    y1 = min(max(y1, lo_y), hi_y) if hi_y >= lo_y else (fh - h) / 2
    return int(round(x1)), int(round(y1)), int(round(x1 + w)), int(round(y1 + h))


def _label_anchor_and_direction(foot_xy, w, h, style):
    """
    Choose where a label starts and which way it may fall back for collision
    avoidance. Default is BELOW the feet (a small gap under the marker) so
    the pill sits away from the player's body rather than over it; ABOVE
    (a small gap over the feet, not the old long stem) is used only when
    below would clip outside the frame's safe margin.
    """
    x, y = foot_xy
    gap = style['label_gap']
    margin, fh = style['margin'], style['frame_height']
    below_y = y + gap
    fits_below = below_y + h <= fh - margin
    if fits_below:
        return (x, below_y), 1
    return (x, y - gap - h), -1


def place_label(anchor_xy, text_size_wh, occupied, style, direction=1, max_tries=3):
    """
    Bounded, deterministic collision avoidance that also guarantees the
    result is fully in-frame: given a starting anchor and a fallback
    direction (see _label_anchor_and_direction), up to `max_tries` offsets
    are attempted further in that direction, and every candidate is clamped
    to the safe margin on both axes before the collision check. Only the
    label may move -- the marker stays at the real anchor. Not a layout
    engine -- three fixed, clamped tries, then suppress the text.
    """
    x, y = anchor_xy
    w, h = text_size_wh
    step = h + 2
    for i in range(max_tries):
        cand_y = y + direction * i * step
        box = _clamp_box_to_frame(x - w / 2, cand_y, w, h, style)
        if not any(_boxes_overlap(box, o) for o in occupied):
            return box
    return None


def _measure_label(text, style):
    font_scale = style['font_scale']
    thickness = style['font_thickness']
    (tw, th), _ = cv2.getTextSize(str(text), FONT, font_scale, thickness)
    w = tw + 2 * style['pill_pad_x']
    h = style['pill_height']
    return w, h, tw, th


def draw_rounded_label_in_box(frame, box, text, bg_color, style, tw, th,
                               border_color=None, border_thickness=0, alpha=0.83):
    """
    Draw the pill background/border/text inside an ALREADY-PLACED box
    (x1, y1, x2, y2) -- callers decide placement (and clamp it) before this
    draws pixels, so drawing never re-derives (and potentially un-clamps) a
    position from a center point.
    """
    x1, y1, x2, y2 = box
    draw_rounded_rect(frame, (x1, y1), (x2, y2), bg_color,
                       style['corner_radius'], alpha=alpha)
    if border_color is not None and border_thickness > 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color,
                       border_thickness)
    cx = (x1 + x2) / 2
    text_x = int(cx - tw / 2)
    text_y = int(y1 + (y2 - y1) / 2 + th / 2 - 1)
    draw_text_with_outline(frame, str(text), (text_x, text_y),
                            style['font_scale'], (255, 255, 255),
                            style['font_thickness'])
    return box


def draw_rounded_label(frame, anchor_xy, text, bg_color, style,
                        border_color=None, border_thickness=0, alpha=0.83):
    """
    Compact rounded ID pill anchored at (x, top_y), clamped to stay fully
    inside the frame's safe margin. Returns the drawn bbox.
    """
    x, top_y = anchor_xy
    w, h, tw, th = _measure_label(text, style)
    box = _clamp_box_to_frame(x - w / 2, top_y, w, h, style)
    return draw_rounded_label_in_box(frame, box, text, bg_color, style, tw, th,
                                      border_color=border_color,
                                      border_thickness=border_thickness,
                                      alpha=alpha)


ROLE_PRIORITY = {'possessor': 0, 'goalkeeper': 1, 'player': 2, 'referee': 3}


def draw_player_marker(frame, foot_xy, role, fill_color, style, track_id=None,
                        is_possessor=False, occupied=None):
    """
    Lightweight tactical-style marker: small ring/dot at the feet, a short
    connector to a compact ID pill placed close to the feet (below by
    default, so the pill does not sit over the player's body -- see
    _label_anchor_and_direction), and a subtle possessor halo. `occupied` is
    a list of already-placed label boxes this frame for collision avoidance
    (mutated in place when a label is placed); pass None to skip labeling.

    Returns the drawn label bbox, or None if no label was placed (marker is
    still drawn either way -- detection/tracking is never hidden, only text).
    """
    x, y = int(foot_xy[0]), int(foot_xy[1])
    r = style['marker_radius']

    if is_possessor:
        # Subtle, neutral white/grey ring -- yellow is reserved for the ball
        # marker only, so the possessor cue can never be confused with it.
        halo_r = r + max(2, int(round(style['scale'] * 4)))
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), halo_r, (235, 235, 235), max(1, style['stroke'] - 1))
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if role == 'goalkeeper':
        cv2.circle(frame, (x, y), r + 2, (255, 255, 255), max(1, style['stroke']))
        cv2.circle(frame, (x, y), r, fill_color, -1)
    elif role == 'referee':
        cv2.circle(frame, (x, y), r, fill_color, -1)
        cv2.circle(frame, (x, y), r, (30, 30, 30), max(1, style['stroke']))
    else:
        cv2.circle(frame, (x, y), r, fill_color, -1)
        cv2.circle(frame, (x, y), r, (20, 20, 20), 1)

    if track_id is None:
        return None

    border_color = (255, 255, 255) if role == 'goalkeeper' else None
    border_thickness = max(1, style['stroke'] - 1) if role == 'goalkeeper' else 0

    w, h, tw, th = _measure_label(track_id, style)
    anchor_xy, direction = _label_anchor_and_direction((x, y), w, h, style)

    def _finish(box):
        drawn = draw_rounded_label_in_box(frame, box, track_id, fill_color, style,
                                           tw, th, border_color=border_color,
                                           border_thickness=border_thickness)
        # Short connector from the marker to wherever the pill actually landed.
        stem_y = box[1] if direction > 0 else box[3]
        cv2.line(frame, (x, y), (x, stem_y), (25, 25, 25), max(1, style['stroke'] - 1))
        return drawn

    if occupied is None:
        box = _clamp_box_to_frame(anchor_xy[0] - w / 2, anchor_xy[1], w, h, style)
        return _finish(box)

    box = place_label(anchor_xy, (w, h), occupied, style, direction=direction)
    if box is None:
        return None
    drawn = _finish(box)
    occupied.append(drawn)
    return drawn


def draw_ball_marker(frame, center_xy, state, style):
    """
    Small high-visibility ring, styled by BallTemporalSelector provenance.
    Callers must simply not invoke this for a frame with no ball entry
    (state == 'unknown' frames carry no ball dict entry already) -- this
    function never guesses a position.
    """
    cx, cy = int(center_xy[0]), int(center_xy[1])
    r = style['ball_ring_radius']
    ring_color = (0, 255, 255)  # bright yellow (BGR), high visibility on turf
    thickness = max(1, style['stroke'])

    if state == 'recovered_low_conf':
        cv2.circle(frame, (cx, cy), r + 1, (0, 0, 0), thickness + 1)
        draw_dashed_circle(frame, (cx, cy), r, ring_color, thickness)
    elif state == 'interpolated_short_gap':
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r, ring_color, max(1, thickness - 1))
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    else:  # 'observed' (and any future default) gets the solid ring
        cv2.circle(frame, (cx, cy), r + 1, (0, 0, 0), thickness + 1)
        cv2.circle(frame, (cx, cy), r, ring_color, thickness)


def fit_text_to_width(text, max_width, style, min_font_scale=0.30):
    """
    Deterministic fit: measure at the HUD's base font scale; if too wide,
    shrink in fixed steps down to `min_font_scale`; if still too wide at the
    floor, ellipsize with '...' (plain ASCII -- Hershey fonts have no
    typographic ellipsis glyph). Never returns a size wider than max_width
    unless even a single ellipsis character doesn't fit (pathological/near-
    zero width), in which case it returns the smallest measured candidate.
    """
    thickness = style['font_thickness']
    base_scale = style['font_scale'] * 0.95
    font_scale = base_scale
    (tw, th) = cv2.getTextSize(text, FONT, font_scale, thickness)[0]
    while tw > max_width and font_scale > min_font_scale:
        font_scale = max(min_font_scale, round(font_scale * 0.88, 3))
        (tw, th) = cv2.getTextSize(text, FONT, font_scale, thickness)[0]

    if tw <= max_width:
        return text, font_scale, (tw, th)

    # Still too wide at the minimum font scale: shorten with an ellipsis.
    trimmed = text
    candidate = trimmed + "..."
    (tw, th) = cv2.getTextSize(candidate, FONT, font_scale, thickness)[0]
    while len(trimmed) > 1 and tw > max_width:
        trimmed = trimmed[:-1]
        candidate = trimmed + "..."
        (tw, th) = cv2.getTextSize(candidate, FONT, font_scale, thickness)[0]
    return candidate, font_scale, (tw, th)


def draw_viewer_hud(frame, style, lines):
    """
    Compact top-left HUD, capped to hud_max_w_frac of frame width so it can
    never grow off-screen. Caller is responsible for keeping `lines` short
    (spec: ~3-5 concise, viewer-facing lines only); each line is measured and
    fit to the panel's inner width via fit_text_to_width so text never draws
    outside the HUD panel.
    """
    fw, fh = style['frame_width'], style['frame_height']
    max_w = int(fw * style['hud_max_w_frac'])
    pad = style['hud_pad']
    line_h = style['hud_line_h']
    x1 = style['margin']
    y1 = style['margin']
    h = pad * 2 + line_h * max(1, len(lines))
    x2 = min(fw - style['margin'], x1 + max_w)
    y2 = min(fh - style['margin'], y1 + h)
    inner_w = max(1, (x2 - x1) - 2 * pad)
    draw_rounded_rect(frame, (x1, y1), (x2, y2), (15, 15, 15),
                       style['corner_radius'], alpha=0.55)
    for i, line in enumerate(lines):
        ty = y1 + pad + line_h * (i + 1) - int(line_h * 0.3)
        if ty > y2 - 2:
            break
        fitted, font_scale, _ = fit_text_to_width(line, inner_w, style)
        draw_text_with_outline(frame, fitted, (x1 + pad, ty), font_scale,
                                (255, 255, 255), style['font_thickness'])
    return (x1, y1, x2, y2)
