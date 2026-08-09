"""
Reuse of existing human detector GT inside the tracking benchmark.

The risk being tested is not that the seed is missing -- it is that the seed
quietly carries something it must not: an identity, a ball, a box from the
wrong frame, or detector output masquerading as reviewed geometry.
"""

import json
from pathlib import Path

import pytest

from tools.build_human_gt_seed import ALIGN_MARGIN, CLASSES
from tools.check_human_seed_agreement import compare
from trackers.detector import HUMAN_CLASSES

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'tracking_val_gt'
SEED = ROOT / 'human_seed'
pytestmark = pytest.mark.skipif(not SEED.exists(), reason='seed not built')


@pytest.fixture(scope='module')
def seeds():
    return [json.loads(p.read_text(encoding='utf-8')) for p in SEED.glob('*.json')]


class TestSeedContent:
    def test_seed_carries_no_identity(self, seeds):
        for s in seeds:
            assert s['contains_identity'] is False
            for fr in s['frames']:
                for b in fr['boxes']:
                    assert 'id' not in b, 'identity cannot come from detector GT'

    def test_only_human_roles_and_no_ball(self, seeds):
        for s in seeds:
            assert s['class_mapping']['dropped'] == ['ball']
            for fr in s['frames']:
                for b in fr['boxes']:
                    assert b['role'] in HUMAN_CLASSES

    def test_ball_is_a_declared_class_that_was_dropped(self):
        assert 'ball' in CLASSES and 'ball' not in HUMAN_CLASSES

    def test_frames_are_one_based_package_frames(self, seeds):
        for s in seeds:
            for fr in s['frames']:
                assert 1 <= fr['package_frame'] <= 300
                assert fr['source_frame'] >= fr['package_frame']

    def test_every_seed_frame_records_confirmed_alignment(self, seeds):
        """Filename agreement is not proof; the pixel check must be recorded."""
        for s in seeds:
            for fr in s['frames']:
                a = fr['alignment']
                assert a['nearest_neighbour_ratio'] > ALIGN_MARGIN, (
                    s['sequence'], fr['package_frame'], a)

    def test_geometry_is_absolute_pixels_inside_the_frame(self, seeds):
        for s in seeds:
            for fr in s['frames']:
                for b in fr['boxes']:
                    x1, y1, x2, y2 = b['bbox']
                    assert x2 > x1 and y2 > y1
                    assert -1 <= x1 and -1 <= y1 and x2 <= 641 and y2 <= 361
                    assert max(b['bbox']) > 1.5, 'looks normalised, not pixels'

    def test_provenance_separates_seed_from_preannotation(self, seeds):
        for s in seeds:
            assert 'preannotation' in s['provenance']['is_not']
            assert 'data/labels' in s['provenance']['source']

    def test_seed_does_not_touch_the_frozen_preannotations(self):
        man = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
        for s in man['sequences']:
            import hashlib
            det = ROOT / s['preannotation_det']
            h = hashlib.sha256(det.read_text(encoding='utf-8').encode()).hexdigest()
            assert h == s['preannotation_det_sha256'], s['sequence']


BOX = [100.0, 100.0, 140.0, 200.0]


def _seed_file(tmp_path, boxes):
    root = tmp_path / 'gt'
    (root / 'human_seed').mkdir(parents=True)
    (root / 'human_seed' / 'seq.json').write_text(json.dumps({
        'sequence': 'seq',
        'frames': [{'package_frame': 1, 'boxes': boxes}]}), encoding='utf-8')
    return root


def _ann_file(tmp_path, boxes):
    p = tmp_path / 'ann.json'
    p.write_text(json.dumps({'boxes': boxes}), encoding='utf-8')
    return p


class TestAgreementCheck:
    def test_matching_annotation_reports_no_issue(self, tmp_path):
        root = _seed_file(tmp_path, [{'bbox': BOX, 'role': 'player'}])
        ann = _ann_file(tmp_path, [{'frame': 1, 'id': 1, 'bbox': BOX,
                                    'role': 'player'}])
        issues, tally = compare(root, 'seq', ann)
        assert issues == [] and tally['matched'] == 1

    def test_role_disagreement_is_reported(self, tmp_path):
        root = _seed_file(tmp_path, [{'bbox': BOX, 'role': 'referee'}])
        ann = _ann_file(tmp_path, [{'frame': 1, 'id': 1, 'bbox': BOX,
                                    'role': 'player'}])
        issues, tally = compare(root, 'seq', ann)
        assert tally['role'] == 1
        assert any('reviewed detector GT says referee' in m for m in issues)

    def test_missing_person_is_reported(self, tmp_path):
        root = _seed_file(tmp_path, [{'bbox': BOX, 'role': 'player'}])
        ann = _ann_file(tmp_path, [])
        issues, tally = compare(root, 'seq', ann)
        assert tally['missing'] == 1 and issues

    def test_extra_person_is_counted_not_flagged(self, tmp_path):
        root = _seed_file(tmp_path, [{'bbox': BOX, 'role': 'player'}])
        ann = _ann_file(tmp_path, [
            {'frame': 1, 'id': 1, 'bbox': BOX, 'role': 'player'},
            {'frame': 1, 'id': 2, 'bbox': [300.0, 100.0, 340.0, 200.0],
             'role': 'player'}])
        issues, tally = compare(root, 'seq', ann)
        assert tally['extra'] == 1 and issues == []

    def test_agreement_check_never_writes(self, tmp_path):
        import hashlib
        root = _seed_file(tmp_path, [{'bbox': BOX, 'role': 'player'}])
        ann = _ann_file(tmp_path, [{'frame': 1, 'id': 1, 'bbox': BOX,
                                    'role': 'referee'}])
        before = hashlib.sha256(ann.read_bytes()).hexdigest()
        compare(root, 'seq', ann)
        assert hashlib.sha256(ann.read_bytes()).hexdigest() == before
