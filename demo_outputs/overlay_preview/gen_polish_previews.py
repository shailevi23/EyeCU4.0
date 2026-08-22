"""
Generate 4 real Viewer V2 preview stills from the EXISTING cached tracks
(demo_outputs/final_e2e_demo/cache/tracks_*.pkl) after the polish pass.

No model inference: tracks/ball_candidates come from the frozen cache; pixel
content for each still comes from decoding the same source video file that
produced that cache (plain video read, not detection -- the pipeline does
this same read step before any model runs). Nothing here calls YOLO, SN3D,
or CBIoU, and nothing here touches the cache file.
"""
import os
import sys
import pickle

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
from trackers.football_tracker import FootballTracker


def imwrite_unicode(path, img, ext='.jpg'):
    """
    cv2.imwrite() silently returns False (no exception) for a path containing
    non-ASCII characters on Windows -- this repo's path includes Hebrew
    characters, so plain cv2.imwrite() here writes nothing. Encode in memory
    and write through Python's own open(), which handles Unicode paths
    correctly on Windows.
    """
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    with open(path, 'wb') as f:
        f.write(buf.tobytes())
    return True

CACHE_PATH = os.path.join('demo_outputs', 'final_e2e_demo', 'cache',
                           'tracks_b3bacf1707645184.pkl')
VIDEO_PATH = os.path.join('input-videos', 'Bayern Munich 3-1 Chelsea.mp4')
SKIP_FRAMES = 2
OUT_DIR = os.path.dirname(__file__)

# processed-frame indices chosen from the cached tracks' own density profile:
# 0 = sequence start (sparse-ish), 297/298 = the least-populated frames in
# this cache, 370 = the most-populated frame (crowded tactical scene).
PREVIEW_FRAME_IDXS = [0, 150, 297, 370]


def main():
    if not os.path.exists(CACHE_PATH):
        print(f"MISSING: {CACHE_PATH}")
        sys.exit(1)
    with open(CACHE_PATH, 'rb') as f:
        blob = pickle.load(f)
    tracks = blob['tracks']
    n_cached = len(tracks['players'])
    print(f"Loaded cache: cache_format={blob.get('cache_format')}, frames={n_cached}")

    tracker = FootballTracker.__new__(FootballTracker)  # drawing only, no model load
    tracker.colors = {
        'player': (0, 255, 0), 'goalkeeper': (0, 165, 255),
        'ball': (0, 0, 255), 'referee': (255, 255, 0),
    }

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"MISSING: could not open {VIDEO_PATH}")
        sys.exit(1)

    for idx in PREVIEW_FRAME_IDXS:
        if idx >= n_cached:
            print(f"SKIP idx {idx}: beyond cached frame count {n_cached}")
            continue
        raw_idx = idx * SKIP_FRAMES
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"SKIP idx {idx}: could not read raw frame {raw_idx}")
            continue

        single_tracks = {k: [tracks[k][idx]] for k in ('players', 'goalkeepers', 'referees', 'ball')}
        density = (len(single_tracks['players'][0]) + len(single_tracks['goalkeepers'][0])
                   + len(single_tracks['referees'][0]))
        out_frames = tracker.draw_annotations([frame], single_tracks, team_ball_control=None,
                                               overlay_mode='viewer')
        out_path = os.path.join(OUT_DIR, f"polish_preview_frame{idx:04d}.jpg")
        written = imwrite_unicode(out_path, out_frames[0])
        status = "OK" if written and os.path.exists(out_path) else "FAILED"
        print(f"{status}  frame {idx:4d} (raw {raw_idx:4d}, density {density:2d}) -> {out_path}")

    cap.release()


if __name__ == '__main__':
    main()
