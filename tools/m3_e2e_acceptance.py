#!/usr/bin/env python
"""
M3 -- ONE development end-to-end acceptance run through the real production
entry point (FootballAnalysisPipeline / full_pipeline.py), on non-TEST
footage, using the now-corrected production defaults.

This is a STRUCTURAL acceptance check, not an accuracy benchmark: it proves
the pipeline runs start-to-finish on the real entry point, exercises every
closed component, and that unsupported metrics stay clearly labelled.
Nothing here is tuned from the result.
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from full_pipeline import FootballAnalysisPipeline  # noqa: E402

VIDEO = 'input-videos/Bayern Munich 3-1 Chelsea.mp4'
OUT_DIR = 'experiments/records/experiment_M3/e2e_acceptance_output'


def main():
    pipeline = FootballAnalysisPipeline(
        output_dir=OUT_DIR,
        match_id=1,
        use_cache=False,
        show_speed=False,
        show_distance=False,
        # everything else left at the (now-corrected) production defaults:
        # yolo_model=best_A_960.pt, imgsz=960, confidence=0.25,
        # ball_candidate_pool=True, tracker_backend='cbiou',
        # ball_detector_backend='sn3d'
    )

    stats = pipeline.process_video(
        video_path=VIDEO,
        skip_frames=2,
        max_frames=80,
        display_results=False,
    )

    report = pipeline.generate_final_report()

    tracks = pipeline.adv_tracker.tracks
    checks = {}

    checks['frames_processed'] = pipeline.frame_count
    checks['frames_processed_gt_0'] = pipeline.frame_count > 0

    n_player_dets = sum(len(f) for f in tracks.get('players', []))
    checks['human_tracks_exist'] = n_player_dets > 0
    checks['n_player_detections_total'] = n_player_dets

    ball_frames = tracks.get('ball', [])
    n_ball_present = sum(1 for f in ball_frames if f)
    checks['ball_branch_active'] = len(ball_frames) > 0
    checks['n_ball_present_frames'] = n_ball_present
    checks['n_ball_total_frames'] = len(ball_frames)

    # selector states valid: every ball dict should carry one of the four
    # legal BallTemporalSelector states, nothing else
    from trackers.ball_temporal import OBSERVED, RECOVERED, INTERPOLATED, UNKNOWN
    legal_states = {OBSERVED, RECOVERED, INTERPOLATED, UNKNOWN}
    seen_states = set()
    for f in ball_frames:
        for v in f.values():
            seen_states.add(v.get('state'))
    checks['selector_states_seen'] = sorted(str(s) for s in seen_states)
    checks['selector_states_all_legal'] = seen_states <= (legal_states | {None})

    # ball excluded from CBIoU: no human-track bbox should ever equal a
    # same-frame ball bbox. Compares bboxes, not track-id numbers -- track
    # ids and class ids are unrelated small-integer namespaces that collide
    # by pure coincidence, so an id-based check would be meaningless (this
    # is the same bbox-comparison approach as the dedicated unit test,
    # tests/test_cbiou_integration.py::TestEyeCuSemantics::
    # test_ball_never_enters_human_association, which already proves this
    # invariant under a controlled synthetic detector; this just confirms it
    # holds on the real end-to-end run too).
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

    # possession path executes without crash
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

    # statistics/output generation completes
    stats_path = Path(OUT_DIR) / 'processing_stats.json'
    checks['processing_stats_written'] = stats_path.exists()
    report_path = Path(OUT_DIR) / 'final_report.json'
    checks['final_report_written'] = report_path.exists()

    # unsupported calibrated metrics NOT presented as validated
    checks['speed_distance_calibrated_field'] = report.get('speed_distance_calibrated')
    checks['speed_distance_marked_uncalibrated'] = report.get('speed_distance_calibrated') is False
    player_stats_path = Path(OUT_DIR) / 'reports' / 'player_statistics.json'
    uncalibrated_fields_present = False
    if player_stats_path.exists():
        ps = json.loads(player_stats_path.read_text(encoding='utf-8'))
        if ps:
            sample = next(iter(ps.values()))
            uncalibrated_fields_present = (
                'max_speed_kmh_UNCALIBRATED' in sample
                and 'total_distance_m_UNCALIBRATED' in sample)
    checks['uncalibrated_fields_correctly_labelled'] = uncalibrated_fields_present

    # provenance metadata emitted
    checks['cache_key_present'] = bool(report.get('cache_key'))
    checks['source_fps_present'] = report.get('source_fps') not in (None, 0)
    checks['effective_fps_present'] = report.get('effective_fps') not in (None, 0)

    checks['ALL_PASS'] = all([
        checks['frames_processed_gt_0'],
        checks['human_tracks_exist'],
        checks['ball_branch_active'],
        checks['selector_states_all_legal'],
        checks['ball_excluded_from_human_tracks'],
        checks['possession_path_executes'],
        checks['processing_stats_written'],
        checks['final_report_written'],
        checks['speed_distance_marked_uncalibrated'],
        checks['uncalibrated_fields_correctly_labelled'],
        checks['cache_key_present'],
        checks['source_fps_present'],
        checks['effective_fps_present'],
    ])

    out = Path('experiments/records/experiment_M3/e2e_acceptance_result.json')
    out.write_text(json.dumps(checks, indent=1, default=str), encoding='utf-8')
    print(json.dumps(checks, indent=1, default=str))
    print('written:', out)
    return 0 if checks['ALL_PASS'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
