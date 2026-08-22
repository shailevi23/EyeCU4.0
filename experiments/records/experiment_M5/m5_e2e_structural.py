#!/usr/bin/env python
"""
M5 Section 7 -- structural E2E acceptance, one fresh production-pipeline run
per TEST sequence. Adapted directly from tools/m3_e2e_acceptance.py's own
checks (that file is unmodified; this is a new M5-only script), pointed at
each TEST video instead of dev footage.

Structural only: proves the real entry point (FootballAnalysisPipeline)
completes, CBIoU never absorbs a ball box into a human track, the ball
branch is active, BallTemporalSelector states are all legal, possession
executes without crashing, output/provenance files are written, and
calibration/speed/distance stay marked unsupported. No TEST accuracy claim
of any kind is made or computed here -- this file does not touch
TEST_DETECTION_ANNOTATIONS.json at all.
"""
import json
import sys
from pathlib import Path

import torch
import cv2

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from full_pipeline import FootballAnalysisPipeline  # noqa: E402
from trackers.ball_temporal import OBSERVED, RECOVERED, INTERPOLATED, UNKNOWN  # noqa: E402

VIDEOS = {
    'como_2-0_sassuolo': 'input-videos/Como 2-0 Sassuolo.mp4',
    'manchester_city_v_liverpool': 'input-videos/Manchester City v Liverpool.mp4',
    'youth_2': 'input-videos/youth 2.webm',
}
MAX_FRAMES = 40
SKIP_FRAMES = 2


def run_one(seq, video):
    out_dir = f'experiments/records/experiment_M5/e2e_output/{seq}'
    pipeline = FootballAnalysisPipeline(
        output_dir=out_dir, match_id=1, use_cache=False,
        show_speed=False, show_distance=False,
        # everything else at the frozen production defaults
    )
    pipeline.process_video(video_path=video, skip_frames=SKIP_FRAMES,
                           max_frames=MAX_FRAMES, display_results=False)
    report = pipeline.generate_final_report()
    tracks = pipeline.adv_tracker.tracks
    checks = {'sequence': seq, 'video': video}

    checks['frames_processed'] = pipeline.frame_count
    checks['frames_processed_gt_0'] = pipeline.frame_count > 0

    n_player_dets = sum(len(f) for f in tracks.get('players', []))
    checks['human_tracks_exist'] = n_player_dets > 0
    checks['n_player_detections_total'] = n_player_dets

    ball_frames = tracks.get('ball', [])
    checks['ball_branch_active'] = len(ball_frames) > 0
    checks['n_ball_present_frames'] = sum(1 for f in ball_frames if f)
    checks['n_ball_total_frames'] = len(ball_frames)

    legal_states = {OBSERVED, RECOVERED, INTERPOLATED, UNKNOWN}
    seen_states = set()
    for f in ball_frames:
        for v in f.values():
            seen_states.add(v.get('state'))
    checks['selector_states_seen'] = sorted(str(s) for s in seen_states)
    checks['selector_states_all_legal'] = seen_states <= (legal_states | {None})

    human_key_leak = False
    for i, ball_frame in enumerate(ball_frames):
        ball_boxes = {tuple(v['bbox']) for v in ball_frame.values()}
        if not ball_boxes:
            continue
        for key in ('players', 'goalkeepers', 'referees'):
            for t in tracks.get(key, [{}] * len(ball_frames))[i].values():
                if tuple(t['bbox']) in ball_boxes:
                    human_key_leak = True
    checks['ball_excluded_from_human_tracks'] = not human_key_leak

    possession_ok = False
    possession_error = None
    try:
        team_ball_control = pipeline.ball_assigner.compute_team_ball_control(tracks)
        possession_ok = True
        checks['possession_frames_scored'] = int(len(team_ball_control))
    except Exception as e:  # noqa: BLE001 -- acceptance check, must not crash the run
        possession_error = f'{type(e).__name__}: {e}'
    checks['possession_path_executes'] = possession_ok
    checks['possession_error'] = possession_error

    stats_path = Path(out_dir) / 'processing_stats.json'
    checks['processing_stats_written'] = stats_path.exists()
    report_path = Path(out_dir) / 'final_report.json'
    checks['final_report_written'] = report_path.exists()

    checks['speed_distance_calibrated_field'] = report.get('speed_distance_calibrated')
    checks['speed_distance_marked_uncalibrated'] = report.get('speed_distance_calibrated') is False
    player_stats_path = Path(out_dir) / 'reports' / 'player_statistics.json'
    uncalibrated_fields_present = False
    if player_stats_path.exists():
        ps = json.loads(player_stats_path.read_text(encoding='utf-8'))
        if ps:
            sample = next(iter(ps.values()))
            uncalibrated_fields_present = (
                'max_speed_kmh_UNCALIBRATED' in sample
                and 'total_distance_m_UNCALIBRATED' in sample)
    checks['uncalibrated_fields_correctly_labelled'] = uncalibrated_fields_present

    checks['cache_key_present'] = bool(report.get('cache_key'))
    checks['source_fps_present'] = report.get('source_fps') not in (None, 0)
    checks['effective_fps_present'] = report.get('effective_fps') not in (None, 0)

    checks['ALL_PASS'] = all([
        checks['frames_processed_gt_0'], checks['human_tracks_exist'],
        checks['ball_branch_active'], checks['selector_states_all_legal'],
        checks['ball_excluded_from_human_tracks'], checks['possession_path_executes'],
        checks['processing_stats_written'], checks['final_report_written'],
        checks['speed_distance_marked_uncalibrated'],
        checks['uncalibrated_fields_correctly_labelled'],
        checks['cache_key_present'], checks['source_fps_present'],
        checks['effective_fps_present'],
    ])
    return checks


def main():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)

    results = {}
    for seq, video in VIDEOS.items():
        print(f'=== {seq} ===', flush=True)
        results[seq] = run_one(seq, video)
        print(json.dumps(results[seq], indent=1, default=str))

    overall = all(r['ALL_PASS'] for r in results.values())
    out = {'per_sequence': results, 'overall_all_pass': overall,
          'max_frames_per_sequence': MAX_FRAMES, 'skip_frames': SKIP_FRAMES}
    out_path = Path('experiments/records/experiment_M5/E2E_ACCEPTANCE.json')
    out_path.write_text(json.dumps(out, indent=1, default=str), encoding='utf-8')
    print('\noverall_all_pass:', overall)
    print('written:', out_path)
    return 0 if overall else 1


if __name__ == '__main__':
    raise SystemExit(main())
