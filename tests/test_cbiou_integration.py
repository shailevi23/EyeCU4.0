"""
CBIoU production integration: deterministic imports, exact defaults, EyeCU
semantics, and a legacy path that still works.

The integration risk here is not that CBIoU tracks badly -- T2 measured that.
It is that the vendored copy silently becomes something other than what was
measured: a different module wins the import, a default drifts, an identity
gets attached to the wrong detection, or the ball finds its way into human
association.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / 'rf_trackers'
PROV = VENDOR / 'VENDOR_PROVENANCE.json'

pytestmark = pytest.mark.skipif(not PROV.exists(), reason='CBIoU not vendored')


@pytest.fixture(scope='module')
def provenance():
    return json.loads(PROV.read_text(encoding='utf-8'))


class TestDeterministicImports:
    def test_the_two_packages_are_distinct_and_unambiguous(self):
        import rf_trackers
        import trackers
        rf = Path(rf_trackers.__file__).resolve()
        ey = Path(trackers.__file__).resolve()
        assert rf.parent.name == 'rf_trackers'
        assert ey.parent.name == 'trackers'
        assert rf != ey
        assert REPO in rf.parents and REPO in ey.parents

    def test_cbiou_comes_from_the_vendored_package(self):
        from rf_trackers import CBIoUTracker
        assert CBIoUTracker.__module__.startswith('rf_trackers.')
        assert Path(__import__('rf_trackers.core.cbiou.tracker',
                               fromlist=['x']).__file__).is_relative_to(VENDOR)

    def test_no_module_named_plain_trackers_was_shadowed(self):
        """EyeCU's own package must still be importable and be EyeCU's."""
        from trackers.football_tracker import FootballTracker
        assert Path(FootballTracker.__module__.replace('.', '/')).name == \
            'football_tracker'

    def test_integration_uses_no_path_or_sys_modules_manipulation(self):
        src = (REPO / 'trackers' / 'football_tracker.py').read_text(encoding='utf-8')
        for bad in ('sys.path.insert', 'sys.path.append', 'del sys.modules',
                    'importlib.reload', 'os.chdir'):
            assert bad not in src, f'{bad} in the production tracker'


class TestVendorProvenance:
    def test_records_source_version_and_licence(self, provenance):
        v = provenance['vendored_from']
        assert v['package'] == 'trackers' and v['version'] == '2.6.0'
        assert v['license'] == 'Apache-2.0'
        assert (VENDOR / 'LICENSE').exists() and (VENDOR / 'NOTICE').exists()
        assert 'Apache License' in (VENDOR / 'LICENSE').read_text(encoding='utf-8')

    def test_every_vendored_file_still_hashes_as_recorded(self, provenance):
        for m in provenance['modules']:
            p = REPO / m['vendored_path']
            assert hashlib.sha256(p.read_bytes()).hexdigest() == m['vendored_sha256'], \
                m['vendored_path']

    def test_only_imports_were_rewritten(self, provenance):
        assert 'import statements' in provenance['only_change']
        assert provenance['import_lines_rewritten_total'] > 0

    def test_wheel_hash_recorded(self, provenance):
        assert provenance['vendored_from'].get('wheel_sha256')


class TestExactFrozenDefaults:
    """T2 qualified one configuration. Any drift makes the result inapplicable."""

    EXPECTED = {
        'lost_track_buffer': 30, 'track_activation_threshold': 0.7,
        'minimum_consecutive_frames': 2, 'minimum_iou_threshold_first_assoc': 0.2,
        'minimum_iou_threshold_second_assoc': 0.5,
        'minimum_iou_threshold_unconfirmed_assoc': 0.3,
        'high_conf_det_threshold': 0.6, 'instant_first_frame_activation': True,
        'buffer_ratio_first': 0.3, 'buffer_ratio_second': 0.5,
    }

    def test_constructor_defaults_match_trackers_2_6_0(self):
        import inspect
        from rf_trackers import CBIoUTracker
        sig = inspect.signature(CBIoUTracker.__init__)
        for k, v in self.EXPECTED.items():
            assert sig.parameters[k].default == v, k

    def test_production_passes_only_frame_rate(self):
        src = (REPO / 'trackers' / 'football_tracker.py').read_text(encoding='utf-8')
        assert 'CBIoUTracker(frame_rate=self.frame_rate)' in src, (
            'production must supply nothing but the frame rate')

    def test_no_tuning_knobs_were_added_to_production(self):
        import inspect
        from trackers.football_tracker import FootballTracker
        params = set(inspect.signature(FootballTracker.__init__).parameters)
        for forbidden in ('track_activation_threshold', 'high_conf_det_threshold',
                          'lost_track_buffer', 'minimum_iou_threshold',
                          'buffer_ratio_first', 'cmc_method'):
            assert forbidden not in params, forbidden


class TestBackendSelection:
    def test_both_backends_construct(self):
        from trackers.football_tracker import FootballTracker

        class _Stub:
            def detect(self, *a, **k):
                return []
        for backend, expect in (('legacy', 'ByteTrack'), ('cbiou', 'CBIoUTracker')):
            ft = FootballTracker(detector=_Stub(), persist_cache=False,
                                 tracker_backend=backend, frame_rate=25.0)
            assert type(ft.tracker).__name__ == expect
            assert ft.tracker_backend == backend

    def test_legacy_rollback_remains_available(self):
        from trackers.football_tracker import FootballTracker

        class _Stub:
            def detect(self, *a, **k):
                return []
        ft = FootballTracker(detector=_Stub(), persist_cache=False,
                             tracker_backend='legacy')
        assert type(ft.tracker).__name__ == 'ByteTrack'

    def test_unknown_backend_is_refused(self):
        from trackers.football_tracker import FootballTracker

        class _Stub:
            def detect(self, *a, **k):
                return []
        with pytest.raises(ValueError, match='unknown tracker_backend'):
            FootballTracker(detector=_Stub(), persist_cache=False,
                            tracker_backend='botsort')

    def test_frame_rate_is_carried_to_cbiou(self):
        from trackers.football_tracker import FootballTracker

        class _Stub:
            def detect(self, *a, **k):
                return []
        ft = FootballTracker(detector=_Stub(), persist_cache=False,
                             tracker_backend='cbiou', frame_rate=25.0)
        assert ft.frame_rate == 25.0
        assert ft.tracker._frame_rate == 25.0
        # the frame rate is not decorative: it scales how long a lost track is
        # kept, so a tracker told 30 fps on 25 fps video holds tracks 20% longer
        at_30 = FootballTracker(detector=_Stub(), persist_cache=False,
                                tracker_backend='cbiou', frame_rate=30.0)
        assert ft.tracker.maximum_frames_without_update !=             at_30.tracker.maximum_frames_without_update


class TestEyeCuSemantics:
    """Run both backends over synthetic frames and inspect what comes out."""

    @staticmethod
    def _tracker(backend):
        from trackers.football_tracker import FootballTracker

        class _Detector:
            """Two humans and a ball, moving slowly, on every frame."""
            def detect(self, frame, *a, **k):
                return [
                    {'class': 'player', 'bbox': [100, 100, 130, 180], 'confidence': 0.9},
                    {'class': 'goalkeeper', 'bbox': [400, 120, 428, 196], 'confidence': 0.85},
                    {'class': 'ball', 'bbox': [250, 300, 262, 312], 'confidence': 0.8},
                ]
        return FootballTracker(detector=_Detector(), persist_cache=False,
                               tracker_backend=backend, frame_rate=25.0)

    @pytest.mark.parametrize('backend', ['legacy', 'cbiou'])
    def test_ball_never_enters_human_association(self, backend):
        ft = self._tracker(backend)
        tracks = ft.get_object_tracks([np.zeros((360, 640, 3), np.uint8)] * 8,
                                      read_from_cache=False)
        for key in ('players', 'goalkeepers', 'referees'):
            for frame in tracks[key]:
                for t in frame.values():
                    assert t['bbox'] != [250, 300, 262, 312], 'ball tracked as human'
        assert any(f for f in tracks['ball']), 'ball must still be reported'

    @pytest.mark.parametrize('backend', ['legacy', 'cbiou'])
    def test_goalkeeper_is_not_normalised_into_player(self, backend):
        ft = self._tracker(backend)
        tracks = ft.get_object_tracks([np.zeros((360, 640, 3), np.uint8)] * 8,
                                      read_from_cache=False)
        assert any(f for f in tracks['goalkeepers']), 'goalkeeper class lost'
        gk_boxes = [t['bbox'] for f in tracks['goalkeepers'] for t in f.values()]
        pl_boxes = [t['bbox'] for f in tracks['players'] for t in f.values()]
        assert all(b[0] > 300 for b in gk_boxes)
        assert all(b[0] < 300 for b in pl_boxes)

    @pytest.mark.parametrize('backend', ['legacy', 'cbiou'])
    def test_identity_attaches_without_rewriting_confidence(self, backend):
        ft = self._tracker(backend)
        tracks = ft.get_object_tracks([np.zeros((360, 640, 3), np.uint8)] * 8,
                                      read_from_cache=False)
        confs = {round(t['confidence'], 4)
                 for key in ('players', 'goalkeepers')
                 for f in tracks[key] for t in f.values()}
        assert confs <= {0.9, 0.85}, confs

    @pytest.mark.parametrize('backend', ['legacy', 'cbiou'])
    def test_no_duplicate_identity_within_a_frame(self, backend):
        ft = self._tracker(backend)
        tracks = ft.get_object_tracks([np.zeros((360, 640, 3), np.uint8)] * 8,
                                      read_from_cache=False)
        for key in ('players', 'goalkeepers', 'referees'):
            for frame in tracks[key]:
                assert len(frame) == len(set(frame)), 'ids are dict keys, so a '\
                    'duplicate would silently overwrite'

    def test_cbiou_never_emits_a_non_positive_identity(self):
        """CBIoU reports -1 before minimum_consecutive_frames is met."""
        ft = self._tracker('cbiou')
        tracks = ft.get_object_tracks([np.zeros((360, 640, 3), np.uint8)] * 8,
                                      read_from_cache=False)
        for key in ('players', 'goalkeepers', 'referees'):
            for frame in tracks[key]:
                for tid in frame:
                    assert tid > 0, tid


@pytest.mark.skipif(not (REPO / 'experiments' / 'tracking_v2' / 'integration'
                         / 'equivalence.json').exists(),
                    reason='equivalence not verified')
class TestMatchesFrozenT2Reference:
    def test_output_is_exactly_the_t2_reference(self):
        eq = json.loads((REPO / 'experiments' / 'tracking_v2' / 'integration'
                         / 'equivalence.json').read_text(encoding='utf-8'))
        assert eq['exact_equivalence'] is True
        for seq, r in eq['sequences'].items():
            assert r['issues'] == [], (seq, r['issues'])
            assert r['max_box_delta'] == 0.0
            assert r['reference_rows'] == r['integrated_rows']
