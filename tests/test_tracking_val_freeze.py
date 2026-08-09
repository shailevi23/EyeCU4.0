"""
Frozen tracking-input package contract.

Every downstream tracker comparison assumes these detections are identical
across runs. If the package drifts from its manifest the comparison still
*looks* reproducible, which is the failure mode worth testing for.

The serialisation-equivalence test re-runs the real detector on a small
deterministic subset and demands an exact match against the reloaded file --
no tolerance, because JSON round-trips Python floats exactly.
"""

import json
from pathlib import Path

import pytest

from tools.validate_tracking_val import EXPECTED, N_FRAMES, validate
from trackers.detector import CLASSES, HUMAN_CLASSES

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'tracking_val_v1'
pytestmark = pytest.mark.skipif(not (ROOT / 'manifest.json').exists(),
                                reason='frozen tracking inputs not built')


@pytest.fixture(scope='module')
def manifest():
    return json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def rows(manifest):
    out = {}
    for w in manifest['windows']:
        text = (ROOT / w['detections_file']).read_text(encoding='utf-8')
        out[w['sequence']] = [json.loads(l) for l in text.splitlines() if l.strip()]
    return out


class TestPackage:
    def test_validator_passes(self):
        errors, checks = validate(ROOT)
        assert checks > 0
        assert errors == [], errors[:5]

    def test_exactly_the_four_frozen_windows(self, manifest):
        assert {w['sequence'] for w in manifest['windows']} == EXPECTED

    def test_every_window_has_exactly_300_frames(self, rows):
        for seq, r in rows.items():
            assert len(r) == N_FRAMES, seq

    def test_frame_indices_complete_and_unique(self, rows):
        for seq, r in rows.items():
            idx = [x['frame'] for x in r]
            assert idx == list(range(N_FRAMES)), seq
            assert len(set(idx)) == N_FRAMES, seq

    def test_no_ball_in_the_human_stream(self, rows):
        for seq, r in rows.items():
            for f in r:
                for d in f['detections']:
                    assert d['class'] != 'ball', seq
                    assert d['class'] in HUMAN_CLASSES, seq

    def test_no_tracker_identity_anywhere(self, rows):
        """The package must not be able to answer the question it exists to pose."""
        banned = {'tracker_id', 'id', 'track_id', 'identity'}
        for seq, r in rows.items():
            for f in r:
                for d in f['detections']:
                    assert not (banned & set(d)), f'{seq}: {set(d) & banned}'

    def test_classes_are_known(self, rows):
        for seq, r in rows.items():
            for f in r:
                for d in f['detections']:
                    assert d['class'] in CLASSES, seq

    def test_bboxes_well_formed(self, manifest, rows):
        size = {w['sequence']: (w['frame_width'], w['frame_height'])
                for w in manifest['windows']}
        for seq, r in rows.items():
            W, H = size[seq]
            for f in r:
                for d in f['detections']:
                    x1, y1, x2, y2 = d['bbox']
                    assert x2 > x1 and y2 > y1, seq
                    assert -1 <= x1 and -1 <= y1 and x2 <= W + 1 and y2 <= H + 1, seq

    def test_confidence_at_or_above_the_frozen_threshold(self, manifest, rows):
        thr = manifest['detector']['confidence']
        for seq, r in rows.items():
            for f in r:
                for d in f['detections']:
                    assert thr - 1e-9 <= d['confidence'] <= 1.0, seq

    def test_manifest_records_provenance(self, manifest):
        det = manifest['detector']
        assert det['checkpoint_sha256'] and len(det['checkpoint_sha256']) == 64
        assert det['imgsz'] == 960
        assert det['human_candidate_pool'] is False
        # v1.1 splits the single commit into the commit the detections were
        # produced at and the commit of the tool that wrote the current package.
        assert manifest.get('detector_source_commit') or manifest.get('code_commit')
        for w in manifest['windows']:
            assert len(w['detections_sha256']) == 64
            assert len(w['source_video_sha256']) == 64
            assert len(w['decoded_frames_sha256']) == 64

    def test_no_test_source(self, manifest):
        from tools.build_derived_train import TEST_MATCHES, VAL_MATCHES
        srcs = {w['match'] for w in manifest['windows']}
        assert not (srcs & TEST_MATCHES)
        assert srcs <= VAL_MATCHES

    def test_ball_settings_present_but_no_ball_data(self, manifest):
        assert 'ball_settings_for_provenance_only' in manifest
        assert manifest['detector']['ball_candidate_pool'] is False


class TestTamperDetection:
    def test_modified_file_fails_the_hash_check(self, tmp_path, manifest):
        """A silently edited detection file must not validate."""
        import shutil
        copy = tmp_path / 'pkg'
        shutil.copytree(ROOT, copy)
        target = copy / manifest['windows'][0]['detections_file']
        lines = target.read_text(encoding='utf-8').splitlines()
        obj = json.loads(lines[0])
        if obj['detections']:
            obj['detections'][0]['confidence'] = 0.999
        else:
            obj['detections'] = [{'bbox': [1.0, 1.0, 2.0, 2.0], 'class': 'player',
                                  'confidence': 0.999, 'state': None}]
        lines[0] = json.dumps(obj, sort_keys=True)
        target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        errors, _ = validate(copy)
        assert any('SHA256 mismatch' in e for e in errors), errors[:3]

    def test_dropped_frame_is_detected(self, tmp_path, manifest):
        import shutil
        copy = tmp_path / 'pkg'
        shutil.copytree(ROOT, copy)
        target = copy / manifest['windows'][0]['detections_file']
        lines = target.read_text(encoding='utf-8').splitlines()
        target.write_text('\n'.join(lines[:-1]) + '\n', encoding='utf-8')
        errors, _ = validate(copy)
        assert errors


@pytest.mark.slow
class TestSerialisationEquivalence:
    def test_reloaded_detections_match_a_fresh_detector_run_exactly(self, manifest, rows):
        """
        Proves the serialisation layer changes nothing. A deterministic subset
        of frames is re-detected and compared field by field against the
        reloaded file -- exact equality, since JSON round-trips floats.
        """
        import cv2
        import numpy as np
        from trackers.detector import LocalDetector

        w = manifest['windows'][0]
        sample = [0, 37, 150, 299]                       # fixed, not random
        cap = cv2.VideoCapture(w['source_video'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, w['start_frame'])
        frames = []
        for _ in range(N_FRAMES):
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        assert len(frames) == N_FRAMES

        det = LocalDetector(manifest['detector']['checkpoint'],
                            confidence=manifest['detector']['confidence'],
                            imgsz=manifest['detector']['imgsz'])
        frozen = rows[w['sequence']]
        for i in sample:
            live = [d for d in det.detect(frames[i], None) if d['class'] in HUMAN_CLASSES]
            saved = frozen[i]['detections']
            assert len(live) == len(saved), f'frame {i}: {len(live)} vs {len(saved)}'
            for a, b in zip(live, saved):
                assert a['class'] == b['class'], f'frame {i}'
                assert float(a['confidence']) == b['confidence'], f'frame {i}'
                assert [float(v) for v in a['bbox']] == b['bbox'], f'frame {i}'


@pytest.mark.skipif(not (ROOT / 'manifest.json').exists(), reason='package not built')
class TestCandidateView:
    """
    The candidate view is the frozen detector EVIDENCE STORE for tracker
    association. Its contract is stricter than the accepted view's in one
    respect: it must contain the accepted view exactly, or every downstream
    comparison is built on two different detector runs.
    """

    @staticmethod
    def _canon(dets):
        return sorted((d['class'], round(float(d['confidence']), 12),
                       tuple(round(float(v), 12) for v in d['bbox'])) for d in dets)

    def _rows(self, w, key):
        text = (ROOT / w[key]).read_text(encoding='utf-8')
        return [json.loads(l) for l in text.splitlines() if l.strip()]

    def test_package_declares_both_views(self, manifest):
        assert manifest['version'] == '1.1'
        assert manifest['accepted_human_threshold'] == 0.25
        assert manifest['candidate_human_floor'] == 0.01
        assert 'accepted' in manifest['views'] and 'candidate' in manifest['views']
        assert manifest['detector_source_commit'] and manifest['freeze_tool_commit']

    def test_evidence_floor_is_documented_as_a_choice(self, manifest):
        note = manifest['evidence_floor_note'].lower()
        assert 'evidence floor' in note
        assert 'not claimed to be algorithmically optimal' in note

    def test_candidate_exists_for_every_sequence(self, manifest):
        for w in manifest['windows']:
            assert (ROOT / w['candidate_file']).exists(), w['sequence']

    def test_candidate_frames_complete(self, manifest):
        for w in manifest['windows']:
            rows = self._rows(w, 'candidate_file')
            assert len(rows) == N_FRAMES
            assert [r['frame'] for r in rows] == list(range(N_FRAMES))

    def test_candidate_respects_the_declared_floor(self, manifest):
        floor = manifest['candidate_human_floor']
        for w in manifest['windows']:
            for r in self._rows(w, 'candidate_file'):
                for d in r['detections']:
                    assert d['confidence'] >= floor - 1e-9, w['sequence']

    def test_candidate_has_no_ball_and_no_identity(self, manifest):
        banned = {'tracker_id', 'id', 'track_id', 'identity'}
        for w in manifest['windows']:
            for r in self._rows(w, 'candidate_file'):
                for d in r['detections']:
                    assert d['class'] in HUMAN_CLASSES, w['sequence']
                    assert d['class'] != 'ball'
                    assert not (banned & set(d))

    def test_no_top_k_truncation(self, manifest):
        """A saturated top-k would silently cut the evidence universe."""
        for w in manifest['windows']:
            assert w['candidate_frames_at_max_det'] == 0, w['sequence']
            assert w['candidate_frames_within_90pct_max_det'] == 0, w['sequence']
            assert w['candidate_raw_boxes_max'] < 300, w['sequence']

    def test_accepted_subset_invariance_recomputed(self, manifest):
        """Recomputed here rather than trusting the manifest's own record."""
        accept = manifest['accepted_human_threshold']
        total = 0
        for w in manifest['windows']:
            acc = self._rows(w, 'detections_file')
            cand = self._rows(w, 'candidate_file')
            for a, c in zip(acc, cand):
                sub = [d for d in c['detections'] if d['confidence'] >= accept]
                assert self._canon(sub) == self._canon(a['detections']), \
                    f"{w['sequence']} frame {a['frame']}"
                total += 1
        assert total == N_FRAMES * 4

    def test_candidate_is_a_superset(self, manifest):
        for w in manifest['windows']:
            assert w['candidate_detections'] >= w['human_detections'], w['sequence']

    def test_accepted_view_unchanged_by_the_candidate_freeze(self, manifest):
        """The accepted artifacts are versioned; the candidate freeze must not
        have touched them."""
        import hashlib
        for w in manifest['windows']:
            text = (ROOT / w['detections_file']).read_text(encoding='utf-8')
            assert hashlib.sha256(text.encode()).hexdigest() == w['detections_sha256']


class TestFrozenProfileSpec:
    SPEC = Path(__file__).resolve().parents[1] / 'experiments' / 'tracking_v2' / 'adoption_criteria.json'

    @pytest.fixture(scope='class')
    def spec(self):
        return json.loads(self.SPEC.read_text(encoding='utf-8'))

    def test_spec_exists_before_results(self, spec):
        assert spec['frozen_before_any_identity_gt_or_bakeoff_result'] is True
        assert spec['adoption_criteria']['frozen_before_results'] is True

    def test_two_profiles_only(self, spec):
        assert set(spec['profiles']) == {'LIBRARY_DEFAULTS', 'EYECU_SCORE_POLICY_V1'}
        assert spec['constraints']['no_third_profile'] is True
        assert spec['constraints']['no_optuna'] is True
        assert spec['constraints']['no_threshold_sweep'] is True

    def test_score_policy_matches_the_frozen_thresholds(self, spec):
        pol = spec['profiles']['EYECU_SCORE_POLICY_V1']['confidence_overrides_only']
        for t in ('ByteTrackTracker', 'BoTSORTTracker', 'CBIoUTracker'):
            assert pol[t]['high_conf_det_threshold'] == 0.25
            assert pol[t]['track_activation_threshold'] == 0.25
        assert pol['OCSORTTracker']['high_conf_det_threshold'] == 0.25
        assert 'track_activation_threshold' not in pol['OCSORTTracker']

    def test_legacy_baseline_is_not_normalised(self, spec):
        assert 'NOT modified' in spec['legacy_baseline']['rule']

    def test_floors_agree_with_the_frozen_package(self, spec):
        assert spec['detector_evidence']['candidate_human_floor'] == 0.01
        assert spec['detector_evidence']['accepted_human_threshold'] == 0.25
