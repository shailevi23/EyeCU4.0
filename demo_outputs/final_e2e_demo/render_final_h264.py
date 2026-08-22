"""
Final native-resolution (640x360) Viewer V2 render, from the ORIGINAL source
video frames + the EXISTING tracks cache -- zero model inference.

Guardrail: this script verifies the exact tracks cache the pipeline would
need is present and well-formed (cache_format=2, tracks, ball_candidates)
BEFORE calling process_video(). If it is missing or invalid, it aborts
without ever touching the detector/tracker/selector models.

fps is written at 12.5 (source 25 fps / skip_frames 2) instead of the CLI
default of 15, so the 375 processed frames play back at their real timing
(~30s) instead of sped up. codec is fixed to real H.264 ('avc1') in
full_pipeline.py's save_output_video -- this script just calls it.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pickle
from trackers.cache_utils import compute_cache_key, cache_path_for
from trackers.detector import (BALL_ACCEPT_CONF, BALL_CANDIDATE_CONF,
                               HUMAN_ACCEPT_CONF, HUMAN_CANDIDATE_CONF,
                               SN3D_BALL_IMGSZ, resolve_sn3d_ball_path,
                               verify_sn3d_ball_checkpoint)
from full_pipeline import FootballAnalysisPipeline

VIDEO_PATH = os.path.join('input-videos', 'Bayern Munich 3-1 Chelsea.mp4')
OUTPUT_DIR = os.path.join('demo_outputs', 'final_e2e_demo')
OUTPUT_NAME = 'tracked_output_viewer_v2_final_h264.mp4'
MAX_FRAMES = 750
SKIP_FRAMES = 2
YOLO_MODEL = 'best_A_960.pt'
IMGSZ = 960
CONFIDENCE = 0.25
TRACKER_BACKEND = 'cbiou'
BALL_DETECTOR_BACKEND = 'sn3d'  # SN3D_BASE, yolo-sn-ball.pt @1280 -- production default
BALL_CANDIDATE_POOL = True
MAX_BALL_GAP = 15


def verify_cache_or_stop():
    """
    Reproduce the exact cache key the pipeline will look up, and confirm that
    file already exists and is well-formed, before running anything.
    """
    ball_model_path = resolve_sn3d_ball_path(None)
    ball_identity = {'backend': 'sn3d',
                      'sha256': verify_sn3d_ball_checkpoint(ball_model_path),
                      'imgsz': SN3D_BALL_IMGSZ,
                      'accept_conf': BALL_ACCEPT_CONF,
                      'candidate_conf': BALL_CANDIDATE_CONF if BALL_CANDIDATE_POOL else None}

    cache_key = compute_cache_key(
        video_path=VIDEO_PATH,
        model_path=YOLO_MODEL,
        detector_settings={'imgsz': IMGSZ, 'confidence': CONFIDENCE,
                           'use_roboflow': False,
                           'human_candidate_pool': False,
                           'human_candidate_conf': None,
                           'human_accept_conf': max(CONFIDENCE, HUMAN_ACCEPT_CONF),
                           'ball': ball_identity},
        tracker_settings={'max_ball_gap': MAX_BALL_GAP,
                          'tracker': TRACKER_BACKEND,
                          'ball_candidate_pool': BALL_CANDIDATE_POOL,
                          'ball_temporal_selector': 'v1'},
        skip_frames=SKIP_FRAMES,
        max_frames=MAX_FRAMES,
    )
    cache_path = cache_path_for(os.path.join(OUTPUT_DIR, 'cache'), 'tracks', cache_key)
    print(f"Expected tracks cache: {cache_path}")

    if not os.path.exists(cache_path):
        print(f"STOP: tracks cache not found at {cache_path}")
        print("Refusing to fall back to inference. No render performed.")
        sys.exit(1)

    with open(cache_path, 'rb') as f:
        blob = pickle.load(f)
    if not isinstance(blob, dict) or blob.get('cache_format') != 2:
        print(f"STOP: cache at {cache_path} is not cache_format=2 (got {blob.get('cache_format') if isinstance(blob, dict) else type(blob)}).")
        sys.exit(1)
    if 'tracks' not in blob or 'ball_candidates' not in blob:
        print(f"STOP: cache at {cache_path} is missing 'tracks' or 'ball_candidates'.")
        sys.exit(1)

    print(f"Cache verified: cache_format={blob['cache_format']}, "
          f"tracks keys={list(blob['tracks'].keys())}, "
          f"ball_candidates frames={len(blob['ball_candidates'])}")
    return cache_key


def main():
    cache_key = verify_cache_or_stop()

    pipeline = FootballAnalysisPipeline(
        yolo_model=YOLO_MODEL,
        output_dir=OUTPUT_DIR,
        use_cache=True,          # read_from_cache=True -> must hit, or this script's guard above already stopped
        imgsz=IMGSZ,
        confidence=CONFIDENCE,
        max_ball_gap=MAX_BALL_GAP,
        ball_candidate_pool=BALL_CANDIDATE_POOL,
        tracker_backend=TRACKER_BACKEND,
        ball_detector_backend=BALL_DETECTOR_BACKEND,
        overlay_mode='viewer',
    )

    stats = pipeline.process_video(
        video_path=VIDEO_PATH,
        skip_frames=SKIP_FRAMES,
        max_frames=MAX_FRAMES,
        display_results=False,
    )
    if stats.get('error'):
        print(f"STOP: pipeline reported an error: {stats['error']}")
        sys.exit(1)
    if pipeline.cache_key != cache_key:
        print(f"STOP: pipeline computed a different cache key ({pipeline.cache_key}) "
              f"than expected ({cache_key}) -- refusing to proceed on a mismatch.")
        sys.exit(1)

    # effective_fps = source_fps / skip_frames = 25 / 2 = 12.5 -- correct
    # playback timing for the existing processed frames, no interpolation.
    pipeline.save_output_video(OUTPUT_NAME, fps=pipeline.effective_fps)
    pipeline.generate_final_report()
    pipeline.cleanup()

    print(f"\nDone. Output: {os.path.join(OUTPUT_DIR, OUTPUT_NAME)}")


if __name__ == '__main__':
    main()
