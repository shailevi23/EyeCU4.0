"""
TODO.md section 6 / section 8: speed and distance must not change when
skip_frames changes.

Only every skip_frames-th frame reaches the estimator, so the gap between the
frames it sees is skip_frames times longer in real time. If the estimator is
handed the source FPS instead of source_fps / skip_frames, every speed scales
by skip_frames — a player at 25 km/h reads 50 km/h at skip=2.
"""

import numpy as np
import pytest

from trackers.speed_distance import SpeedDistanceEstimator

SOURCE_FPS = 30.0
PIXELS_PER_METER = 12.0
PX_PER_SOURCE_FRAME = 4.0  # constant velocity


def tracks_for(skip_frames, n_processed=30):
    """
    One player moving at a constant velocity, sampled every skip_frames-th
    source frame. Larger skip means the same motion appears as bigger jumps
    between fewer samples.
    """
    players = []
    for i in range(n_processed):
        x = i * PX_PER_SOURCE_FRAME * skip_frames
        players.append({1: {'bbox': [x, 0.0, x + 10.0, 20.0],
                            'position': (x, 0.0)}})
    return {'players': players, 'goalkeepers': [], 'referees': [], 'ball': []}


def max_speed(tracks):
    return max(info['speed']
               for frame in tracks['players']
               for info in frame.values()
               if 'speed' in info)


def total_distance(tracks):
    return max(info['distance']
               for frame in tracks['players']
               for info in frame.values()
               if 'distance' in info)


@pytest.mark.parametrize('skip_frames', [1, 2, 3, 5])
def test_speed_is_consistent_across_skip_frames(skip_frames):
    """Same physical motion -> same km/h, whatever skip_frames is."""
    baseline = tracks_for(1)
    SpeedDistanceEstimator(frame_rate=SOURCE_FPS,
                           pixels_per_meter=PIXELS_PER_METER
                           ).add_speed_and_distance_to_tracks(baseline)
    expected = max_speed(baseline)

    tracks = tracks_for(skip_frames)
    effective_fps = SOURCE_FPS / skip_frames
    SpeedDistanceEstimator(frame_rate=effective_fps,
                           pixels_per_meter=PIXELS_PER_METER
                           ).add_speed_and_distance_to_tracks(tracks)

    assert max_speed(tracks) == pytest.approx(expected, rel=1e-6)


def test_speed_is_wrong_if_skip_frames_is_ignored():
    """
    Guards the regression itself: feeding the raw source FPS at skip=2 inflates
    speed by exactly 2x. If this ever stops holding, the test above has become
    vacuous.
    """
    tracks = tracks_for(2)
    SpeedDistanceEstimator(frame_rate=SOURCE_FPS,          # bug: not divided
                           pixels_per_meter=PIXELS_PER_METER
                           ).add_speed_and_distance_to_tracks(tracks)

    correct = tracks_for(2)
    SpeedDistanceEstimator(frame_rate=SOURCE_FPS / 2,
                           pixels_per_meter=PIXELS_PER_METER
                           ).add_speed_and_distance_to_tracks(correct)

    assert max_speed(tracks) == pytest.approx(2 * max_speed(correct), rel=1e-6)


@pytest.mark.parametrize('skip_frames', [2, 3])
def test_distance_covered_is_consistent_across_skip_frames(skip_frames):
    """
    Distance is frame-rate independent, but a coarser sample covers less of the
    clip: n samples at skip=k span k times more source frames. Compare over the
    same span of real time instead of the same sample count.
    """
    n = 30
    fine = tracks_for(1, n_processed=n * skip_frames)
    SpeedDistanceEstimator(frame_rate=SOURCE_FPS,
                           pixels_per_meter=PIXELS_PER_METER
                           ).add_speed_and_distance_to_tracks(fine)

    coarse = tracks_for(skip_frames, n_processed=n)
    SpeedDistanceEstimator(frame_rate=SOURCE_FPS / skip_frames,
                           pixels_per_meter=PIXELS_PER_METER
                           ).add_speed_and_distance_to_tracks(coarse)

    # Windowing granularity differs, so allow a few percent rather than exact.
    assert total_distance(coarse) == pytest.approx(total_distance(fine), rel=0.10)


def test_effective_fps_formula():
    """source_fps / skip_frames, and skip_frames < 1 must not divide by zero."""
    for source, skip, expected in [(30.0, 1, 30.0), (30.0, 2, 15.0),
                                   (25.0, 5, 5.0), (30.0, 0, 30.0)]:
        assert source / max(1, skip) == pytest.approx(expected)
