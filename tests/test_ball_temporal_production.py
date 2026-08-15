"""The production ball path: BallTemporalSelector, not interpolate+bfill+ffill.

These are synthetic unit tests. Nothing here loads YOLO weights or touches a
GPU -- the tracker is driven through a stub detector, and the selector is
driven directly.

The property under test throughout is that a reported ball is either a real
detection or an explicitly-labelled estimate, and that "no ball" survives all
the way to possession and output rather than being filled in.
"""

import inspect
import json

import numpy as np
import pytest

from trackers.ball_temporal import (INTERPOLATED, OBSERVED, RECOVERED, UNKNOWN,
                                    BallTemporalSelector, FrameInput)
from trackers.football_tracker import FootballTracker
from trackers.player_ball_assigner import PlayerBallAssigner


def box(cx, cy, s=10.0):
    return [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]


def frames(cands, dt=0.2, cuts=None):
    return [FrameInput(candidates=c, timestamp=i * dt, dt=dt,
                       cut=bool(cuts[i]) if cuts else False)
            for i, c in enumerate(cands)]


def obs(cx, cy, conf=0.9):
    return {'bbox': box(cx, cy), 'confidence': conf}


# ------------------------------------------------------------- 1. baseline


def test_consecutive_detections_are_all_observed():
    out = BallTemporalSelector().run(
        frames([[obs(100, 100)], [obs(110, 100)], [obs(120, 100)]]))
    assert [o.state for o in out] == [OBSERVED] * 3
    assert all(o.confidence == 0.9 for o in out)
    # geometry is passed through untouched, not smoothed
    assert out[1].bbox == box(110, 100)


# ------------------------------------------------- 2. short recoverable gap


def test_short_gap_between_anchors_is_interpolated_and_labelled():
    out = BallTemporalSelector().run(
        frames([[obs(100, 100)], [], [obs(120, 100)]]))
    assert [o.state for o in out] == [OBSERVED, INTERPOLATED, OBSERVED]
    assert out[1].bbox == pytest.approx(box(110, 100))
    # an estimate has no detector confidence and must not claim one
    assert out[1].confidence is None


# ----------------------------------------------------- 3. gap too long


def test_long_gap_stays_unknown():
    # 6 empty frames at dt=0.2 is 1.2s, far beyond MAX_INTERP_GAP_SECONDS=0.4
    out = BallTemporalSelector().run(
        frames([[obs(100, 100)]] + [[]] * 6 + [[obs(400, 300)]]))
    assert out[0].state == OBSERVED and out[-1].state == OBSERVED
    assert all(o.state == UNKNOWN for o in out[1:-1])
    assert all(o.bbox is None for o in out[1:-1])


def test_camera_cut_blocks_interpolation_across_it():
    out = BallTemporalSelector().run(
        frames([[obs(100, 100)], [], [obs(120, 100)]], cuts=[False, False, True]))
    assert out[1].state == UNKNOWN, 'a cut invalidates the trajectory'


# ----------------------------------------------- 4/5. clip start and end


def test_no_ball_at_the_start_is_never_back_filled():
    out = BallTemporalSelector().run(
        frames([[], [], [obs(100, 100)], [obs(110, 100)]]))
    assert [o.state for o in out[:2]] == [UNKNOWN, UNKNOWN]
    assert all(o.bbox is None for o in out[:2]), 'no ball before the first sighting'


def test_no_ball_at_the_end_is_never_forward_filled():
    out = BallTemporalSelector().run(
        frames([[obs(100, 100)], [obs(110, 100)], [], []]))
    assert [o.state for o in out[2:]] == [UNKNOWN, UNKNOWN]
    assert all(o.bbox is None for o in out[2:]), 'no ball held after the last'


def test_a_clip_with_no_detections_at_all_reports_nothing():
    out = BallTemporalSelector().run(frames([[], [], []]))
    assert all(o.state == UNKNOWN and o.bbox is None for o in out)


# ------------------------------------------- 6. multiple candidates


def test_multiple_observations_pick_the_most_confident():
    out = BallTemporalSelector().run(
        frames([[obs(100, 100, 0.4), obs(500, 400, 0.95)]]))
    assert out[0].state == OBSERVED
    assert out[0].bbox == box(500, 400)


def test_rescue_picks_the_candidate_nearest_the_prediction():
    """Two low-confidence candidates; motion says the ball is near (120,100).
    The more confident one is nowhere near it and must lose."""
    out = BallTemporalSelector().run(frames([
        [obs(100, 100)],
        [obs(110, 100)],
        [{'bbox': box(400, 300), 'confidence': 0.22},     # closer to accept
         {'bbox': box(120, 100), 'confidence': 0.12}],    # on the trajectory
    ]))
    assert out[2].state == RECOVERED
    assert out[2].bbox == box(120, 100), 'geometry beat raw confidence'


# --------------------------------------------- 7. low-confidence handling


def test_rescue_candidate_outside_the_gate_is_refused():
    out = BallTemporalSelector().run(frames([
        [obs(100, 100)], [obs(110, 100)],
        [{'bbox': box(600, 450), 'confidence': 0.20}],
    ]))
    assert out[2].state == UNKNOWN, 'far candidate is not the ball'


def test_rescue_requires_a_history_anchor():
    """The close-up case: candidates with no prior trajectory are refused."""
    out = BallTemporalSelector().run(
        frames([[{'bbox': box(100, 100), 'confidence': 0.15}]]))
    assert out[0].state == UNKNOWN


def test_below_the_rescue_floor_is_ignored_entirely():
    out = BallTemporalSelector().run(frames([
        [obs(100, 100)], [obs(110, 100)],
        [{'bbox': box(120, 100), 'confidence': 0.05}],
    ]))
    assert out[2].state == UNKNOWN, '0.05 is below CANDIDATE_CONF=0.10'


# ------------------------------------ 8. possession tolerates missing balls


def _tracks_with_ball(ball_frames):
    players = [{1: {'bbox': [90, 90, 110, 130], 'team': 1},
                2: {'bbox': [300, 90, 320, 130], 'team': 2}}
               for _ in ball_frames]
    return {'players': players, 'ball': ball_frames}


def test_possession_handles_measured_recovered_and_missing():
    tracks = _tracks_with_ball([
        {1: {'bbox': box(100, 120), 'state': OBSERVED, 'confidence': 0.9}},
        {},                                                     # unknown
        {1: {'bbox': box(100, 120), 'state': INTERPOLATED}},     # estimate
        {},
    ])
    control = PlayerBallAssigner(max_distance=70).compute_team_ball_control(tracks)
    assert len(control) == 4, 'one entry per frame, including the empty ones'


def test_possession_with_no_ball_at_all_does_not_crash_or_invent_control():
    tracks = _tracks_with_ball([{}, {}, {}])
    control = PlayerBallAssigner(max_distance=70).compute_team_ball_control(tracks)
    assert len(control) == 3
    assert set(np.asarray(control).tolist()) == {0}, 'no ball -> no team control'


def test_unknown_frames_are_not_credited_to_the_last_holder():
    """A gap must not be scored as possession for whoever held it last on the
    strength of a ball that was never observed.

    The previous version of this test asserted only that consecutive OBSERVED
    frames agreed, and its comment called carry-forward "a documented
    convention". It was not documented anywhere, and the test passed while the
    trailing gap was silently credited to team 1.
    """
    seen = _tracks_with_ball(
        [{1: {'bbox': box(100, 120), 'state': OBSERVED}}] * 2 + [{}] * 2)
    control = PlayerBallAssigner(max_distance=70).compute_team_ball_control(seen)
    assert list(control) == [1, 1, 0, 0], (
        'the two unknown frames must be 0, not an inherited team id')


# --------------------------------- 9. output/visualisation tolerate gaps


def test_draw_annotations_survives_a_missing_ball(monkeypatch):
    t = FootballTracker(detector=object(), persist_cache=False)
    frames_ = [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(3)]
    tracks = {
        'players': [{1: {'bbox': [10, 10, 30, 60], 'team': 1,
                         'team_color': (255, 0, 0)}} for _ in range(3)],
        'referees': [{} for _ in range(3)],
        'goalkeepers': [{} for _ in range(3)],
        'ball': [{1: {'bbox': box(50, 40), 'state': OBSERVED}}, {},
                 {1: {'bbox': box(60, 40), 'state': INTERPOLATED}}],
    }
    out = t.draw_annotations(frames_, tracks, np.array([1, 1, 1]))
    assert len(out) == 3


def test_provenance_survives_json_serialisation():
    entry = {1: {'bbox': box(50, 40), 'state': RECOVERED, 'confidence': 0.14}}
    back = json.loads(json.dumps({'ball': [entry]}))
    assert back['ball'][0]['1']['state'] == RECOVERED
    assert back['ball'][0]['1']['confidence'] == pytest.approx(0.14)


# -------------------------- 10. the production path uses the selector


def test_legacy_interpolator_is_gone():
    assert not hasattr(FootballTracker, 'interpolate_ball_positions'), (
        'interpolate_ball_positions() was removed; it filled missing balls with '
        '[0,0,0,0] and then bfill/ffill-ed a ball onto every frame')


def test_production_pipeline_calls_the_selector():
    import full_pipeline
    src = inspect.getsource(full_pipeline)
    assert 'apply_ball_temporal_selection' in src
    assert 'interpolate_ball_positions' not in src


def test_apply_ball_temporal_selection_tags_every_reported_ball():
    t = FootballTracker(detector=object(), persist_cache=False)
    ball = [{1: {'bbox': box(100, 100), 'confidence': 0.9}},
            {},
            {1: {'bbox': box(120, 100), 'confidence': 0.9}}]
    cands = [[obs(100, 100)], [], [obs(120, 100)]]
    out = t.apply_ball_temporal_selection(ball, candidates=cands, fps=5.0,
                                          frame_width=640)
    assert len(out) == 3
    states = [next(iter(f.values()))['state'] if f else UNKNOWN for f in out]
    assert states == [OBSERVED, INTERPOLATED, OBSERVED]


def test_apply_ball_temporal_selection_leaves_unrecoverable_frames_empty():
    t = FootballTracker(detector=object(), persist_cache=False)
    ball = [{1: {'bbox': box(100, 100), 'confidence': 0.9}}] + [{}] * 6
    cands = [[obs(100, 100)]] + [[]] * 6
    out = t.apply_ball_temporal_selection(ball, candidates=cands, fps=5.0,
                                          frame_width=640)
    assert out[0] and all(f == {} for f in out[1:]), (
        'no anchor on the right-hand side, so nothing may be filled')


def test_apply_ball_temporal_selection_refuses_missing_candidates():
    """Superseded behaviour, kept as a regression guard.

    This test used to assert that a missing candidate record was tolerated by
    rebuilding candidates from `ball_positions`. That fallback was the cache
    bug: `ball_positions` is tracks['ball'], which contains boxes the tracker
    copied forward, and feeding them back relabelled a held box as `observed`.
    Refusing is now the correct behaviour -- see tests/test_ball_cache_equivalence.py.
    """
    t = FootballTracker(detector=object(), persist_cache=False)
    ball = [{1: {'bbox': box(100, 100), 'confidence': 0.9}},
            {},
            {1: {'bbox': box(120, 100), 'confidence': 0.9}}]
    with pytest.raises(ValueError, match='requires detector candidates'):
        t.apply_ball_temporal_selection(ball, candidates=None, fps=5.0,
                                        frame_width=640)


def test_tracker_records_candidates_for_the_selector():
    """The rescue pool must reach the selector, and must not be reported as a
    ball by the tracker itself."""
    class Stub:
        def detect(self, frame, idx=None):
            return [{'class': 'ball', 'bbox': [10, 10, 16, 16],
                     'confidence': 0.15, 'state': 'candidate_low_conf'}]

    t = FootballTracker(detector=Stub(), persist_cache=False)
    tracks = t.get_object_tracks([np.zeros((40, 40, 3), dtype=np.uint8)],
                                 read_from_cache=False, cache_path=None)
    assert tracks['ball'][0] == {}, 'tracker must not accept a rescue candidate'
    assert t.ball_candidates[0][0]['confidence'] == pytest.approx(0.15), (
        'but the candidate must still be available to the selector')


# ------------------------------------------------ tracker backend default


def test_cbiou_is_the_default_everywhere():
    import full_pipeline
    assert inspect.signature(
        FootballTracker.__init__).parameters['tracker_backend'].default == 'cbiou'
    assert inspect.signature(
        full_pipeline.FootballAnalysisPipeline.__init__
    ).parameters['tracker_backend'].default == 'cbiou'


def test_cache_key_reports_the_backend_actually_used():
    """It was hard-coded to 'bytetrack' while the default was CBIoU, so a
    cache built by one backend could be reused by the other."""
    src = inspect.getsource(__import__('full_pipeline'))
    assert "'tracker': self.adv_tracker.tracker_backend" in src
    assert "'tracker': 'bytetrack'" not in src
