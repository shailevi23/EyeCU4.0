"""
The ball must not enter human association, and must be written exactly once.

Before this patch the ball was fed to ByteTrack alongside players, received a
tracker id, was written under that id, and was then written *again* under the
literal key 1 -- two entries for one object. A measured 12-second sequence gave
the ball 9 tracker ids of its own while it competed for IoU matches against
people moving an order of magnitude slower.
"""

import numpy as np
import pytest

from trackers.detector import CLASSES, CLASS_IDS, HUMAN_CLASSES
from trackers.football_tracker import (HUMAN_TRACK_KEY, TRACK_KEY, TRACK_KEYS,
                                       FootballTracker)


def det(cls, x1, y1, x2, y2, conf=0.9, **extra):
    return {'bbox': [x1, y1, x2, y2], 'class': cls, 'confidence': conf, **extra}


class Stub:
    """Detector stub: returns a fixed script, one entry per frame."""

    def __init__(self, script):
        self.script = script
        self.seen = []

    def detect(self, frame, idx=None):
        self.seen.append(idx)
        return self.script[idx if idx is not None else 0]


def run(script, **kw):
    t = FootballTracker(detector=Stub(script), persist_cache=False, **kw)
    frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in script]
    return t, t.get_object_tracks(frames, read_from_cache=False, cache_path=None)


class TestBallIsolation:
    def test_ball_never_reaches_bytetrack(self, monkeypatch):
        seen = []

        script = [[det('player', 10, 10, 40, 90), det('ball', 300, 200, 306, 206)]] * 3
        t = FootballTracker(detector=Stub(script), persist_cache=False)
        real = t.tracker.update_with_detections

        def spy(detections):
            seen.append(list(detections.class_id))
            return real(detections)

        monkeypatch.setattr(t.tracker, 'update_with_detections', spy)
        frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in script]
        t.get_object_tracks(frames, read_from_cache=False, cache_path=None)

        assert seen, 'tracker was never called'
        ball_id = CLASS_IDS['ball']
        for batch in seen:
            assert ball_id not in batch, f'ball class id reached ByteTrack: {batch}'

    def test_ball_never_receives_a_human_tracker_id(self):
        """The ball's only key is the canonical 1, never a ByteTrack id."""
        script = [[det('player', 10, 10, 40, 90),
                   det('ball', 300, 200, 306, 206)] for _ in range(5)]
        _, tracks = run(script)
        for frame in tracks['ball']:
            assert set(frame) <= {1}, f'ball carries a tracker id: {set(frame)}'

    def test_ball_written_exactly_once_per_frame(self):
        script = [[det('ball', 300, 200, 306, 206)] for _ in range(4)]
        _, tracks = run(script)
        for frame in tracks['ball']:
            assert len(frame) == 1

    def test_two_ball_detections_still_yield_one_entry(self):
        script = [[det('ball', 300, 200, 306, 206, conf=0.9),
                   det('ball', 500, 300, 506, 306, conf=0.4)]]
        _, tracks = run(script)
        assert len(tracks['ball'][0]) == 1

    def test_ball_absent_from_every_human_bucket(self):
        script = [[det('player', 10, 10, 40, 90),
                   det('ball', 300, 200, 306, 206)] for _ in range(4)]
        _, tracks = run(script)
        for key in ('players', 'goalkeepers', 'referees'):
            for frame in tracks[key]:
                for v in frame.values():
                    assert v['bbox'] != [300, 200, 306, 206]

    def test_ball_only_frames_produce_no_human_tracks(self):
        script = [[det('ball', 300, 200, 306, 206)] for _ in range(3)]
        _, tracks = run(script)
        assert all(not f for f in tracks['players'])
        assert all(not f for f in tracks['goalkeepers'])
        assert all(not f for f in tracks['referees'])


class TestHumansStillTracked:
    @pytest.mark.parametrize('cls', HUMAN_CLASSES)
    def test_each_human_role_receives_a_tracker_id(self, cls):
        script = [[det(cls, 100, 100, 140, 200)] for _ in range(4)]
        _, tracks = run(script)
        key = TRACK_KEY[cls]
        assert any(tracks[key][i] for i in range(4)), f'{cls} never tracked'
        for frame in tracks[key]:
            for tid in frame:
                assert isinstance(tid, int) and tid >= 1

    def test_roles_stay_in_their_own_buckets(self):
        script = [[det('player', 10, 10, 50, 110),
                   det('goalkeeper', 200, 10, 240, 110),
                   det('referee', 400, 10, 440, 110)] for _ in range(4)]
        _, tracks = run(script)
        assert any(f for f in tracks['players'])
        assert any(f for f in tracks['goalkeepers'])
        assert any(f for f in tracks['referees'])

    def test_goalkeeper_is_never_folded_into_player(self):
        """The reference implementation rewrites GK to player before tracking.
        EyeCU must not: a keeper's kit deliberately differs from their team's,
        which is exactly what breaks jersey-colour team assignment."""
        script = [[det('goalkeeper', 200, 10, 240, 110)] for _ in range(4)]
        _, tracks = run(script)
        assert any(f for f in tracks['goalkeepers'])
        assert all(not f for f in tracks['players'])

    def test_humans_and_ball_coexist(self):
        script = [[det('player', 10, 10, 50, 110),
                   det('ball', 300, 200, 306, 206)] for _ in range(4)]
        _, tracks = run(script)
        assert any(f for f in tracks['players'])
        assert all(len(f) == 1 for f in tracks['ball'])


class TestContract:
    def test_four_detector_classes_preserved(self):
        assert CLASSES == ['player', 'goalkeeper', 'referee', 'ball']
        assert set(TRACK_KEY) == set(CLASSES)
        assert TRACK_KEYS == ['players', 'goalkeepers', 'referees', 'ball']

    def test_human_track_map_excludes_ball_by_construction(self):
        assert CLASS_IDS['ball'] not in HUMAN_TRACK_KEY
        assert set(HUMAN_TRACK_KEY) == {CLASS_IDS[c] for c in HUMAN_CLASSES}

    def test_rescue_candidates_still_excluded(self):
        script = [[det('ball', 300, 200, 306, 206, state='candidate_low_conf')]]
        _, tracks = run(script)
        assert tracks['ball'][0] == {}

    def test_deterministic_across_runs(self):
        script = [[det('player', 10 + i, 10, 50 + i, 110),
                   det('referee', 400, 10, 440, 110),
                   det('ball', 300 + i, 200, 306 + i, 206)] for i in range(6)]
        a = run([list(f) for f in script])[1]
        b = run([list(f) for f in script])[1]
        for key in TRACK_KEYS:
            assert [sorted(f) for f in a[key]] == [sorted(f) for f in b[key]]

    def test_bounded_ball_hold_still_applies(self):
        """The gap logic is untouched by this patch."""
        script = [[det('ball', 300, 200, 306, 206)]] + [[] for _ in range(3)]
        _, tracks = run(script, max_ball_gap=2)
        assert len(tracks['ball'][0]) == 1
        assert tracks['ball'][1].get(1, {}).get('held_for') == 1
        assert tracks['ball'][2].get(1, {}).get('held_for') == 2
        assert tracks['ball'][3] == {}, 'ball held beyond max_ball_gap'
