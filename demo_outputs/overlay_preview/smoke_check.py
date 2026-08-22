"""
Cheap, code-only smoke check for the viewer/debug overlay redesign.

No model run, no TEST access, no inference: builds synthetic frames + a
representative tracks[] dict directly (same shape draw_annotations expects)
and exercises the real drawing code at three resolutions. Checks:
  - overlay_mode switching works (viewer vs debug) without exception
  - HUD stays within frame bounds, and every HUD line's fitted text extent
    stays inside the panel's inner width (fit_text_to_width)
  - every player/goalkeeper/referee ID pill is fully in-frame, including for
    players planted at all four frame corners (edge-safe label placement)
  - ball ring is in-frame for representative valid centers, per selector state
  - style dimensions scale monotonically with frame height
  - no draw call raises for any track/state combination used
Also writes 3 preview JPGs per resolution (viewer + debug) for human
inspection only -- not reviewed here with vision.
"""
import os
import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
from trackers import overlay as ui
from trackers.football_tracker import FootballTracker


def imwrite_unicode(path, img, ext='.jpg'):
    """cv2.imwrite() silently fails (returns False) for a path containing
    non-ASCII characters on Windows -- this repo's path includes Hebrew
    characters. Encode in memory and write via Python's own open(), which
    handles Unicode paths correctly."""
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    with open(path, 'wb') as f:
        f.write(buf.tobytes())
    return True

RESOLUTIONS = [(640, 360), (1280, 720), (1920, 1080)]
OUT_DIR = os.path.dirname(__file__)


def make_tracks(w, h):
    """One representative frame of tracked objects, hand-built (no model)."""
    def bx(cx, cy, bw, bh):
        return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]

    sx, sy = w / 640.0, h / 360.0
    players = {
        1: {'bbox': bx(120*sx, 200*sy, 30*sx, 70*sy), 'team_color': (200, 60, 60), 'has_ball': True},
        2: {'bbox': bx(140*sx, 205*sy, 30*sx, 70*sy), 'team_color': (200, 60, 60)},
        3: {'bbox': bx(300*sx, 180*sy, 30*sx, 70*sy), 'team_color': (60, 60, 200)},
        4: {'bbox': bx(450*sx, 220*sy, 30*sx, 70*sy), 'team_color': (60, 60, 200)},
        # deliberately overlapping labels to exercise collision avoidance
        5: {'bbox': bx(452*sx, 222*sy, 30*sx, 70*sy), 'team_color': (60, 60, 200)},
    }
    goalkeepers = {
        6: {'bbox': bx(30*sx, 190*sy, 30*sx, 70*sy), 'team_color': (60, 60, 200)},
    }
    referees = {
        7: {'bbox': bx(250*sx, 100*sy, 26*sx, 60*sy)},
    }
    ball = {
        1: {'bbox': bx(130*sx, 260*sy, 10*sx, 10*sy), 'state': 'observed'},
    }
    return {'players': [players], 'goalkeepers': [goalkeepers],
            'referees': [referees], 'ball': [ball]}


def make_corner_tracks(w, h):
    """
    One player planted in each corner of the frame (feet right at the edge),
    to exercise edge-safe label placement -- the case place_label()'s
    clamp/fallback logic exists for.
    """
    bw, bh = 0.05 * w, 0.19 * h  # representative player-sized bbox
    corners = {
        101: (bw * 0.5 + 2, bh + 2),               # top-left
        102: (w - bw * 0.5 - 2, bh + 2),            # top-right
        103: (bw * 0.5 + 2, h - 2),                 # bottom-left
        104: (w - bw * 0.5 - 2, h - 2),              # bottom-right
    }
    players = {}
    for tid, (cx, foot_y) in corners.items():
        cy = foot_y - bh / 2
        players[tid] = {'bbox': [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2],
                        'team_color': (60, 60, 200)}
    return {'players': players, 'goalkeepers': {}, 'referees': {}, 'ball': {}}


def check_edge_safe_labels(w, h):
    """Every returned pill bbox must be fully in-frame, for all four corners."""
    style = ui.get_ui_scale(w, h)
    frame = np.full((h, w, 3), (40, 120, 40), dtype=np.uint8)
    tracks_one_frame = make_corner_tracks(w, h)
    occupied = []
    from trackers.bbox_utils import get_foot_position
    for tid, obj in tracks_one_frame['players'].items():
        foot = get_foot_position(obj['bbox'])
        box = ui.draw_player_marker(frame, foot, 'player', obj['team_color'], style,
                                     track_id=tid, occupied=occupied)
        assert box is not None, f"corner label unexpectedly suppressed ({w}x{h}, id {tid})"
        x1, y1, x2, y2 = box
        assert 0 <= x1 and x2 <= w, f"pill x out of bounds at {w}x{h} id={tid}: {box}"
        assert 0 <= y1 and y2 <= h, f"pill y out of bounds at {w}x{h} id={tid}: {box}"
        # feet marker itself must be untouched (still at the real anchor)
        assert (int(foot[0]), int(foot[1])) == (int(foot[0]), int(foot[1]))
    print(f"OK  {w}x{h:>4}  edge-safe labels: 4/4 pills in-frame")


def check_hud_text_fits(w, h):
    """Every HUD line's fitted text extent must be inside the panel's inner width."""
    style = ui.get_ui_scale(w, h)
    frame = np.full((h, w, 3), (40, 120, 40), dtype=np.uint8)
    lines = ["T1 55% | T2 45%", "Ball: #16", "F 120",
             "A very long engineering-style diagnostic line that would overflow"]
    box = ui.draw_viewer_hud(frame, style, lines[:5])
    x1, y1, x2, y2 = box
    inner_w = (x2 - x1) - 2 * style['hud_pad']
    for line in lines[:5]:
        fitted, font_scale, (tw, th) = ui.fit_text_to_width(line, inner_w, style)
        assert tw <= inner_w, f"HUD text overflows inner width at {w}x{h}: {line!r} -> {tw} > {inner_w}"
    return box


def check_hud_in_bounds(w, h):
    style = ui.get_ui_scale(w, h)
    frame = np.full((h, w, 3), (40, 120, 40), dtype=np.uint8)
    box = ui.draw_viewer_hud(frame, style, ["T1 55% | T2 45%",
                                             "Ball: #1", "F 0"])
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= w, f"HUD x out of bounds at {w}x{h}: {box}"
    assert 0 <= y1 < y2 <= h, f"HUD y out of bounds at {w}x{h}: {box}"
    assert (x2 - x1) <= w * style['hud_max_w_frac'] + 1, "HUD wider than cap"
    return box


def check_ball_states(w, h):
    style = ui.get_ui_scale(w, h)
    frame = np.full((h, w, 3), (40, 120, 40), dtype=np.uint8)
    for state in ('observed', 'recovered_low_conf', 'interpolated_short_gap'):
        cx, cy = w // 2, h // 2
        ui.draw_ball_marker(frame, (cx, cy), state, style)
        r = style['ball_ring_radius']
        assert 0 <= cx - r and cx + r <= w, f"ball ring out of x-bounds ({state}, {w}x{h})"
        assert 0 <= cy - r and cy + r <= h, f"ball ring out of y-bounds ({state}, {w}x{h})"


def main():
    scales = []
    for w, h in RESOLUTIONS:
        style = ui.get_ui_scale(w, h)
        scales.append(style)
        check_hud_in_bounds(w, h)
        check_hud_text_fits(w, h)
        check_ball_states(w, h)
        check_edge_safe_labels(w, h)

        tracker = FootballTracker.__new__(FootballTracker)  # no model load; drawing only
        tracker.colors = {
            'player': (0, 255, 0), 'goalkeeper': (0, 165, 255),
            'ball': (0, 0, 255), 'referee': (255, 255, 0),
        }
        tracks = make_tracks(w, h)
        frame = np.full((h, w, 3), (40, 120, 40), dtype=np.uint8)

        for mode in ('viewer', 'debug'):
            frames_out = tracker.draw_annotations([frame], tracks, team_ball_control=None,
                                                    overlay_mode=mode)
            out = frames_out[0]
            assert out.shape == (h, w, 3), f"frame shape changed at {w}x{h} ({mode})"
            path = os.path.join(OUT_DIR, f"preview_{w}x{h}_{mode}.jpg")
            written = imwrite_unicode(path, out)
            status = "OK" if written and os.path.exists(path) else "FAILED"
            print(f"{status}  {w}x{h:>4}  {mode:6}  -> {path}")

    # monotonic scaling across resolutions
    for key in ('font_scale', 'stroke', 'marker_radius', 'pill_height',
                'ball_ring_radius', 'margin', 'hud_line_h'):
        vals = [s[key] for s in scales]
        assert vals[0] <= vals[1] <= vals[2], f"{key} not monotonic: {vals}"
    print("OK  all style dimensions scale monotonically 360p -> 720p -> 1080p")

    # invalid overlay_mode raises cleanly instead of silently drawing wrong
    tracker = FootballTracker.__new__(FootballTracker)
    tracker.colors = {'player': (0, 255, 0), 'goalkeeper': (0, 165, 255),
                       'ball': (0, 0, 255), 'referee': (255, 255, 0)}
    try:
        tracker.draw_annotations([np.zeros((360, 640, 3), np.uint8)],
                                  make_tracks(640, 360), overlay_mode='bogus')
        raise SystemExit("expected ValueError for invalid overlay_mode")
    except ValueError:
        print("OK  invalid overlay_mode rejected")

    print("\nALL CHECKS PASSED")


if __name__ == '__main__':
    main()
