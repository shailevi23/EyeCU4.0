"""
Human association candidate pool: does ByteTrack's second stage actually run?

supervision splits detections at `track_activation_threshold` and associates the
low half in a second pass:

    inds_low  = scores > 0.1
    inds_high = scores < self.track_activation_threshold   # < 0.25
    inds_second = logical_and(inds_low, inds_high)

Filtering humans at 0.25 before the tracker made that pool permanently empty.
These tests exercise the mechanism itself -- an identity surviving a dip below
the accepted threshold -- rather than just asserting a flag is plumbed through.
"""

# LEGACY-SPECIFIC: this module tests ByteTrack's own second association
# stage and its hardcoded 0.1 low-score floor. Those are sv.ByteTrack
# mechanics, not EyeCU product invariants, so the backend is pinned rather
# than parametrised. The backend-invariant versions of ball isolation,
# goalkeeper semantics and identity attachment live in
# tests/test_cbiou_integration.py, parametrised over both backends.

import numpy as np
import pytest

from trackers.detector import (CLASS_IDS, CLASSES, HUMAN_ACCEPT_CONF,
                               HUMAN_CANDIDATE_CONF, HUMAN_CLASSES)
from trackers.football_tracker import FootballTracker


def det(cls, x1, y1, x2, y2, conf, **extra):
    """A detection as the pooled detector emits it, state tag included."""
    d = {'bbox': [x1, y1, x2, y2], 'class': cls, 'confidence': conf, **extra}
    if cls in HUMAN_CLASSES and 'state' not in d:
        d['state'] = 'observed' if conf >= HUMAN_ACCEPT_CONF else 'candidate_low_conf'
    return d


class Stub:
    def __init__(self, script):
        self.script = script

    def detect(self, frame, idx=None):
        return self.script[idx if idx is not None else 0]


def run(script, pool):
    t = FootballTracker(tracker_backend='legacy', detector=Stub(script), persist_cache=False,
                        human_candidate_pool=pool)
    frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in script]
    return t.get_object_tracks(frames, read_from_cache=False, cache_path=None)


def ids_of(tracks, key='players'):
    return [sorted(f) for f in tracks[key]]


# The scenario from the brief: a confident human, a dip below the accepted
# threshold, then confident again, moving plausibly throughout.
DIP = [
    [det('player', 100, 100, 140, 200, 0.90)],
    [det('player', 106, 100, 146, 200, 0.15)],
    [det('player', 112, 100, 152, 200, 0.90)],
]


class TestMechanism:
    def test_identity_survives_the_low_confidence_frame(self):
        tracks = run(DIP, pool=True)
        first = ids_of(tracks)[0]
        third = ids_of(tracks)[2]
        assert first, 'no track on frame 1'
        assert third, 'identity not recovered on frame 3'
        assert first == third, f'tracker id changed across the dip: {first} -> {third}'

    def test_low_confidence_observation_is_withheld_from_output(self):
        tracks = run(DIP, pool=True)
        assert tracks['players'][1] == {}, 'the 0.15 box leaked into accepted output'

    def test_accepted_output_only_ever_carries_confident_boxes(self):
        tracks = run(DIP, pool=True)
        for frame in tracks['players']:
            for v in frame.values():
                assert v['confidence'] >= HUMAN_ACCEPT_CONF

    def test_flag_off_reproduces_baseline(self):
        """With the pool off the detector never emits sub-threshold humans, so
        the tracker sees a one-frame gap exactly as before."""
        baseline = [[d for d in f if d['confidence'] >= HUMAN_ACCEPT_CONF] for f in DIP]
        off = run(baseline, pool=False)
        assert off['players'][1] == {}
        assert [len(f) for f in off['players']] == [1, 0, 1]

    def test_pool_changes_association_not_output_volume(self):
        """Same accepted boxes either way -- the difference is identity."""
        baseline = [[d for d in f if d['confidence'] >= HUMAN_ACCEPT_CONF] for f in DIP]
        off, on = run(baseline, pool=False), run(DIP, pool=True)
        assert [len(f) for f in off['players']] == [len(f) for f in on['players']]


class TestPoolBoundaries:
    def test_detection_below_the_floor_does_not_enter_association(self):
        """0.09 is under ByteTrack's own hardcoded 0.1 low-score floor."""
        script = [
            [det('player', 100, 100, 140, 200, 0.90)],
            [det('player', 106, 100, 146, 200, 0.09, state='candidate_low_conf')],
            [det('player', 112, 100, 152, 200, 0.90)],
        ]
        tracks = run(script, pool=True)
        assert tracks['players'][1] == {}
        first, third = ids_of(tracks)[0], ids_of(tracks)[2]
        assert first and third

    @pytest.mark.parametrize('cls', HUMAN_CLASSES)
    def test_all_three_human_classes_may_enter_the_pool(self, cls):
        script = [
            [det(cls, 100, 100, 140, 200, 0.90)],
            [det(cls, 106, 100, 146, 200, 0.15)],
            [det(cls, 112, 100, 152, 200, 0.90)],
        ]
        from trackers.football_tracker import TRACK_KEY
        tracks = run(script, pool=True)
        key = TRACK_KEY[cls]
        assert ids_of(tracks, key)[0] == ids_of(tracks, key)[2] != []

    def test_ball_never_enters_this_pool(self, monkeypatch):
        seen = []
        script = [[det('player', 10, 10, 50, 110, 0.9),
                   {'bbox': [300, 200, 306, 206], 'class': 'ball', 'confidence': 0.15}]] * 3
        t = FootballTracker(tracker_backend='legacy', detector=Stub(script), persist_cache=False,
                            human_candidate_pool=True)
        real = t.tracker.update_with_detections
        monkeypatch.setattr(t.tracker, 'update_with_detections',
                            lambda d: (seen.append(list(d.class_id)), real(d))[1])
        frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in script]
        t.get_object_tracks(frames, read_from_cache=False, cache_path=None)
        for batch in seen:
            assert CLASS_IDS['ball'] not in batch

    def test_ball_handling_unchanged_by_the_flag(self):
        script = [[{'bbox': [300, 200, 306, 206], 'class': 'ball', 'confidence': 0.9}]] * 3
        a = run(script, pool=False)['ball']
        b = run(script, pool=True)['ball']
        assert [sorted(f) for f in a] == [sorted(f) for f in b] == [[1], [1], [1]]


class TestSemanticsPreserved:
    def test_raw_class_and_confidence_survive(self):
        tracks = run(DIP, pool=True)
        vals = [v for f in tracks['players'] for v in f.values()]
        assert vals and all(v['confidence'] == pytest.approx(0.90) for v in vals)

    def test_goalkeeper_is_never_normalised_to_player(self):
        script = [
            [det('goalkeeper', 100, 100, 140, 200, 0.90)],
            [det('goalkeeper', 106, 100, 146, 200, 0.15)],
            [det('goalkeeper', 112, 100, 152, 200, 0.90)],
        ]
        tracks = run(script, pool=True)
        assert all(not f for f in tracks['players'])
        assert ids_of(tracks, 'goalkeepers')[0] == ids_of(tracks, 'goalkeepers')[2] != []

    def test_four_classes_and_thresholds_intact(self):
        assert CLASSES == ['player', 'goalkeeper', 'referee', 'ball']
        assert HUMAN_CANDIDATE_CONF == 0.10
        assert HUMAN_ACCEPT_CONF == 0.25
        assert HUMAN_CANDIDATE_CONF < HUMAN_ACCEPT_CONF

    def test_defaults_are_off(self):
        import inspect
        from trackers.detector import LocalDetector, create_detector
        for fn in (LocalDetector.__init__, create_detector, FootballTracker.__init__):
            assert inspect.signature(fn).parameters['human_candidate_pool'].default is False
