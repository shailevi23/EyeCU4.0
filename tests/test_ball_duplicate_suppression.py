"""
Patch 0b contract: ball-only duplicate suppression.

The dangerous failure here is not a missed duplicate -- it is suppressing a
real object. So most of these tests assert what suppression must NOT touch.
"""

import pytest

from trackers.detector import (BALL_ACCEPT_CONF, BALL_CANDIDATE_CONF,
                               BALL_DEDUPE_IOU, suppress_ball_duplicates)


def ball(x1, y1, x2, y2, conf, **extra):
    return dict(bbox=[x1, y1, x2, y2], **{'class': 'ball'},
                confidence=conf, **extra)


def human(cls, x1, y1, x2, y2, conf=0.9):
    return {'bbox': [x1, y1, x2, y2], 'class': cls, 'confidence': conf}


class TestBallOnly:
    def test_heavily_overlapping_humans_are_never_suppressed(self):
        """Players contesting a header overlap almost completely. YOLO26 is
        end-to-end and is meant to emit both; deleting one deletes a person."""
        dets = [human('player', 100, 100, 150, 220),
                human('player', 102, 101, 152, 221),
                human('goalkeeper', 101, 100, 151, 220),
                human('referee', 100, 102, 150, 222)]
        assert suppress_ball_duplicates(dets, BALL_DEDUPE_IOU) == dets

    def test_identical_human_boxes_survive(self):
        dets = [human('player', 10, 10, 60, 130), human('player', 10, 10, 60, 130)]
        assert len(suppress_ball_duplicates(dets, BALL_DEDUPE_IOU)) == 2

    def test_humans_untouched_while_balls_are_deduped(self):
        dets = [human('player', 0, 0, 50, 120),
                human('player', 2, 1, 52, 121),
                ball(300, 300, 306, 306, 0.8),
                ball(300, 300, 306, 306, 0.4)]
        out = suppress_ball_duplicates(dets, BALL_DEDUPE_IOU)
        assert sum(1 for d in out if d['class'] == 'player') == 2
        assert sum(1 for d in out if d['class'] == 'ball') == 1


class TestSuppression:
    def test_exact_duplicate_removed_keeping_higher_confidence(self):
        out = suppress_ball_duplicates(
            [ball(10, 10, 16, 16, 0.30), ball(10, 10, 16, 16, 0.80)],
            BALL_DEDUPE_IOU)
        assert len(out) == 1
        assert out[0]['confidence'] == 0.80

    def test_distinct_balls_both_kept(self):
        """A spare ball elsewhere on the pitch must survive."""
        out = suppress_ball_duplicates(
            [ball(10, 10, 16, 16, 0.9), ball(400, 300, 406, 306, 0.5)],
            BALL_DEDUPE_IOU)
        assert len(out) == 2

    def test_iou_just_below_threshold_is_kept(self):
        # 6x6 boxes offset by 1px: IoU = 30/42 = 0.714 -> suppressed at 0.70
        assert len(suppress_ball_duplicates(
            [ball(0, 0, 6, 6, 0.9), ball(1, 0, 7, 6, 0.5)], BALL_DEDUPE_IOU)) == 1
        # offset by 2px: IoU = 24/48 = 0.50 -> below 0.70, kept
        assert len(suppress_ball_duplicates(
            [ball(0, 0, 6, 6, 0.9), ball(2, 0, 8, 6, 0.5)], BALL_DEDUPE_IOU)) == 2

    def test_single_ball_returns_input_unchanged(self):
        dets = [ball(1, 1, 7, 7, 0.5)]
        assert suppress_ball_duplicates(dets, BALL_DEDUPE_IOU) is dets

    def test_original_ordering_preserved(self):
        dets = [human('player', 0, 0, 10, 20),
                ball(10, 10, 16, 16, 0.4),
                human('referee', 50, 0, 60, 20),
                ball(10, 10, 16, 16, 0.9)]
        out = suppress_ball_duplicates(dets, BALL_DEDUPE_IOU)
        assert [d['class'] for d in out] == ['player', 'referee', 'ball']

    def test_deterministic_on_confidence_ties(self):
        a, b = ball(0, 0, 6, 6, 0.5, tag='a'), ball(0, 0, 6, 6, 0.5, tag='b')
        for _ in range(20):
            out = suppress_ball_duplicates([a, b], BALL_DEDUPE_IOU)
            assert len(out) == 1 and out[0]['tag'] == 'a'

    def test_chain_does_not_delete_a_distinct_third_ball(self):
        """B overlaps A, C overlaps B, but C is distinct from A. Greedy
        suppression must not remove C via a chain through B."""
        out = suppress_ball_duplicates(
            [ball(0, 0, 6, 6, 0.9), ball(1, 0, 7, 6, 0.8), ball(3, 0, 9, 6, 0.7)],
            BALL_DEDUPE_IOU)
        assert len(out) == 2


class TestThresholdConstants:
    def test_measured_values(self):
        """These are measured on the frozen val split, not tuned knobs.
        Changing one invalidates the before/after in docs/results/RESULTS.md."""
        assert BALL_DEDUPE_IOU == 0.70
        assert BALL_CANDIDATE_CONF == 0.10
        assert BALL_ACCEPT_CONF == 0.25
        assert BALL_CANDIDATE_CONF < BALL_ACCEPT_CONF


class TestFlagOff:
    def test_detector_default_is_off(self):
        import inspect
        from trackers.detector import LocalDetector, create_detector
        assert inspect.signature(
            LocalDetector.__init__).parameters['ball_candidate_pool'].default is False
        assert inspect.signature(
            create_detector).parameters['ball_candidate_pool'].default is False

    def test_tracker_ignores_rescue_candidates(self, monkeypatch):
        """A low-confidence candidate must never be accepted as an observation
        while BallTemporalSelector does not exist."""
        import numpy as np
        from trackers.football_tracker import FootballTracker

        class Stub:
            def detect(self, frame, idx=None):
                return [ball(10, 10, 16, 16, 0.15, state='candidate_low_conf')]

        t = FootballTracker(detector=Stub(), persist_cache=False)
        frames = [np.zeros((40, 40, 3), dtype=np.uint8)]
        tracks = t.get_object_tracks(frames, read_from_cache=False, cache_path=None)
        assert tracks['ball'][0] == {}

    def test_tracker_accepts_stateless_detections(self):
        """Detectors without the pool emit no 'state' key; behaviour unchanged."""
        import numpy as np
        from trackers.football_tracker import FootballTracker

        class Stub:
            def detect(self, frame, idx=None):
                return [ball(10, 10, 16, 16, 0.9)]

        t = FootballTracker(detector=Stub(), persist_cache=False)
        frames = [np.zeros((40, 40, 3), dtype=np.uint8)]
        tracks = t.get_object_tracks(frames, read_from_cache=False, cache_path=None)
        assert tracks['ball'][0] != {}
