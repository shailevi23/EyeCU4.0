"""
Pin the installed supervision 0.26.1 ByteTrack semantics the wrapper audit
depends on, and the diagnostic runner's own correctness.

These are characterisation tests: they assert what the INSTALLED library does,
so that a future upgrade which changes these boundaries fails loudly instead of
silently invalidating the audit. Nothing here modifies supervision.
"""

# LEGACY-SPECIFIC: this audits the supervision wrapper itself.

import json
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from tools.audit_supervision_bytetrack_wrapper import (WRAPPER_COST_THRESH,
                                                       replay_wrapper, snapshot)

BOX = [100.0, 100.0, 140.0, 200.0]


def tensors(conf, box=None):
    b = box or BOX
    return np.array([[*b, conf]], dtype=float)


def detections(conf, box=None):
    b = box or BOX
    return sv.Detections(xyxy=np.array([b], dtype=float),
                         confidence=np.array([conf], dtype=float),
                         class_id=np.array([0], dtype=int))


class TestInstalledSemantics:
    """The constants the audit reasons about, read from the live object."""

    def test_constructor_defaults(self):
        t = sv.ByteTrack()
        assert t.track_activation_threshold == 0.25
        assert t.minimum_matching_threshold == 0.8
        assert t.minimum_consecutive_frames == 1

    def test_det_thresh_is_activation_plus_point_one(self):
        t = sv.ByteTrack()
        assert t.det_thresh == pytest.approx(0.35)
        assert sv.ByteTrack(track_activation_threshold=0.5).det_thresh == pytest.approx(0.6)

    def test_max_time_lost_scales_with_frame_rate(self):
        assert sv.ByteTrack().max_time_lost == 30
        assert sv.ByteTrack(frame_rate=25).max_time_lost == 25
        assert sv.ByteTrack(frame_rate=60).max_time_lost == 60

    def test_production_uses_bare_constructor(self):
        """
        The LEGACY backend passes no arguments, so its frame_rate is 30 even on
        25 fps video -- the historical semantics this audit documented.

        Checked on the sv.ByteTrack construction specifically rather than by
        searching the whole file: the CBIoU backend legitimately receives a
        frame_rate, and a file-wide string check would confuse the two.
        """
        src = (Path(__file__).resolve().parents[1] /
               'trackers' / 'football_tracker.py').read_text(encoding='utf-8')
        assert 'sv.ByteTrack()' in src
        constructions = [l.strip() for l in src.splitlines()
                         if 'self.tracker = sv.ByteTrack(' in l]
        assert constructions, 'legacy backend construction not found'
        for line in constructions:
            assert line == 'self.tracker = sv.ByteTrack()', line
        assert 'CBIoUTracker(frame_rate=' in src, (
            'CBIoU is given the real frame rate; only legacy is left bare')


class TestConfidenceBoundaries:
    """
    Installed masks:
        remain_inds = scores >  track_activation_threshold   # > 0.25
        inds_low    = scores >  0.1
        inds_high   = scores <  track_activation_threshold   # < 0.25
    New-track rejection is `score < det_thresh`, so exactly 0.35 initialises.
    """

    @pytest.mark.parametrize('conf,expect_track', [
        (0.0999, False),   # below the low floor
        (0.1000, False),   # excluded: condition is > 0.10, not >=
        (0.1001, False),   # in the low pool, but low pool never creates tracks
        (0.2499, False),   # low pool
        (0.2500, False),   # in NEITHER pool: high is >0.25, low is <0.25
        (0.2501, False),   # high pool, but below det_thresh -> no new track
        (0.3499, False),   # still below det_thresh
        (0.3500, True),    # rejection is `< det_thresh`, so 0.35 passes
        (0.3501, True),
    ])
    def test_new_track_creation_boundary(self, conf, expect_track):
        t = sv.ByteTrack()
        tracks = t.update_with_tensors(tensors=tensors(conf))
        assert bool(tracks) is expect_track, f'conf={conf}'

    def test_exactly_0_25_reaches_neither_pool(self):
        """A detection at exactly the activation threshold is invisible to the
        tracker: it is not > 0.25 and not < 0.25."""
        t = sv.ByteTrack()
        assert t.update_with_tensors(tensors=tensors(0.25)) == []
        t2 = sv.ByteTrack()
        assert len(t2.update_with_detections(detections(0.25))) == 0

    def test_low_pool_can_sustain_but_not_create(self):
        """0.1001 cannot start a track, but can keep an existing one alive."""
        t = sv.ByteTrack()
        assert t.update_with_tensors(tensors=tensors(0.90))          # create
        moved = [102.0, 100.0, 142.0, 200.0]
        assert t.update_with_tensors(tensors=tensors(0.15, moved))   # sustain
        t2 = sv.ByteTrack()
        assert t2.update_with_tensors(tensors=tensors(0.15)) == []   # cannot create


class TestWrapperReplication:
    def test_replay_reproduces_the_wrapper_exactly(self):
        """Two independent instances, same input; the replayed mapping must
        equal the wrapper's own output."""
        a, b = sv.ByteTrack(), sv.ByteTrack()
        rng = np.random.default_rng(0)
        boxes = np.array([[50, 50, 90, 150], [300, 80, 340, 180],
                          [500, 120, 540, 220]], dtype=float)
        for step in range(12):
            moved = boxes + np.array([[step * 2, 0, step * 2, 0]] * 3, dtype=float)
            confs = np.array([0.9, 0.8, 0.7])
            tr = a.update_with_tensors(np.hstack((moved, confs[:, None])))
            out = b.update_with_detections(sv.Detections(
                xyxy=moved.copy(), confidence=confs.copy(),
                class_id=np.zeros(3, dtype=int)))
            ids, _, _ = replay_wrapper(moved, tr)
            assert sorted(int(x) for x in ids if x != -1) == \
                   sorted(int(x) for x in out.tracker_id), f'step {step}'

    def test_wrapper_threshold_is_iou_half(self):
        """cost = 1 - IoU thresholded at 0.5 means IoU >= 0.5 matches."""
        assert WRAPPER_COST_THRESH == 0.5

    def test_replay_returns_no_ids_without_tracks(self):
        ids, ious, pairs = replay_wrapper(np.array([BOX]), [])
        assert list(ids) == [-1] and pairs == []


class TestDiagnosticRunnerSafety:
    def test_independent_instances_do_not_share_state(self):
        a, b = sv.ByteTrack(), sv.ByteTrack()
        a.update_with_tensors(tensors(0.9))
        assert len(b.tracked_tracks) == 0
        assert a.frame_id == 1 and b.frame_id == 0

    def test_calling_both_methods_on_one_instance_double_advances(self):
        """Documents exactly why the audit uses two instances."""
        one = sv.ByteTrack()
        one.update_with_tensors(tensors(0.9))
        one.update_with_detections(detections(0.9))
        assert one.frame_id == 2
        two = sv.ByteTrack()
        two.update_with_tensors(tensors(0.9))
        assert two.frame_id == 1

    def test_snapshot_is_read_only(self):
        t = sv.ByteTrack()
        t.update_with_tensors(tensors(0.9))
        before = (t.frame_id, len(t.tracked_tracks), len(t.lost_tracks))
        snapshot(t); snapshot(t)
        assert (t.frame_id, len(t.tracked_tracks), len(t.lost_tracks)) == before

    def test_diagnostic_import_does_not_touch_production_tracker(self):
        """Importing the audit tool must not alter FootballTracker behaviour."""
        import tools.audit_supervision_bytetrack_wrapper  # noqa: F401
        from trackers.football_tracker import FootballTracker

        class Stub:
            def detect(self, frame, idx=None):
                return [{'bbox': [10, 10, 50, 110], 'class': 'player',
                         'confidence': 0.9}]

        t = FootballTracker(tracker_backend='legacy', detector=Stub(), persist_cache=False)
        frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(3)]
        tracks = t.get_object_tracks(frames, read_from_cache=False, cache_path=None)
        assert any(f for f in tracks['players'])
        assert isinstance(t.tracker, sv.ByteTrack)


RESULT = Path(__file__).resolve().parents[1] / 'experiments' / 'tracking_v2' / \
    'wrapper_audit' / 'result.json'


@pytest.mark.skipif(not RESULT.exists(), reason='wrapper audit not run')
class TestAuditAccounting:
    @pytest.fixture(scope='module')
    def res(self):
        return json.loads(RESULT.read_text(encoding='utf-8'))

    def test_categories_sum_to_input(self, res):
        for s in res['sequences']:
            assert sum(s['categories'].values()) == s['input_accepted_detections'], \
                s['sequence']

    def test_preserved_equals_public_output(self, res):
        for s in res['sequences']:
            assert s['categories'].get('A wrapper-preserved', 0) == \
                s['public_returned_observations'], s['sequence']

    def test_public_never_exceeds_internal_returned(self, res):
        for s in res['sequences']:
            assert s['public_returned_observations'] <= \
                s['internal_returned_track_observations'], s['sequence']

    def test_replay_matched_the_wrapper_on_every_frame(self, res):
        assert sum(s['replay_mismatched_frames'] for s in res['sequences']) == 0

    def test_matched_ious_respect_the_threshold(self, res):
        for s in res['sequences']:
            if s['matched_iou']['min'] is not None:
                assert s['matched_iou']['min'] >= WRAPPER_COST_THRESH, s['sequence']

    def test_config_is_production_exact(self, res):
        c = res['config']
        assert c['track_activation_threshold'] == 0.25
        assert c['det_thresh'] == pytest.approx(0.35)
        assert c['frame_rate_used'] == 30
        assert c['max_time_lost'] == 30
        assert c['view'] == 'accepted'
