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
        assert manifest['code_commit']
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
