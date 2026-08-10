"""
Identity-GT tooling contract.

These test the TOOLING, never fabricated ground truth. The failure this file
exists to prevent is an unannotated or tracker-derived package being accepted
as an answer key, so most assertions are about refusal.
"""

import json
import shutil
from pathlib import Path

import pytest

from tools.validate_tracking_gt import (EXPECTED, N_FRAMES, validate_gt_content,
                                        validate_post, validate_pre,
                                        validate_verified)
from trackers.detector import HUMAN_CLASSES

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'tracking_val_gt'
pytestmark = pytest.mark.skipif(not (ROOT / 'manifest.json').exists(),
                                reason='identity GT package not built')


@pytest.fixture(scope='module')
def manifest():
    return json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))


class TestPackage:
    def test_pre_stage_validates(self):
        errors, n = validate_pre(ROOT)
        assert n > 0 and errors == [], errors[:5]

    def test_exactly_four_sequences(self, manifest):
        assert {s['sequence'] for s in manifest['sequences']} == EXPECTED

    def test_no_test_source(self, manifest):
        from tools.build_derived_train import TEST_MATCHES, VAL_MATCHES
        srcs = {s['match'] for s in manifest['sequences']}
        assert not (srcs & TEST_MATCHES)
        assert srcs <= VAL_MATCHES

    def test_exact_frame_ranges(self, manifest):
        expected = {'austin_fc_vs__club_tijuana_284': (284, 583),
                    'bayern_munich_3-1_chelsea_228': (228, 527),
                    'women_1_239': (239, 538),
                    'youth_premier_league_1133': (1133, 1432)}
        for s in manifest['sequences']:
            assert tuple(s['source_frame_range']) == expected[s['sequence']]
            assert s['frame_count'] == N_FRAMES
            assert s['package_frame_range'] == [1, N_FRAMES]

    def test_frames_on_disk_are_one_based(self, manifest):
        for s in manifest['sequences']:
            d = ROOT / 'sequences' / s['sequence'] / 'img1'
            imgs = sorted(d.glob('*.jpg'))
            assert len(imgs) == N_FRAMES
            assert imgs[0].name == '000001.jpg'
            assert imgs[-1].name == f'{N_FRAMES:06d}.jpg'

    def test_class_mapping_and_ball_exclusion(self, manifest):
        assert set(manifest['target']['classes']) == set(HUMAN_CLASSES)
        assert manifest['target']['ball_excluded'] is True
        assert 'ball' in manifest['target']['not_targets']

    def test_manifest_marks_gt_unannotated(self, manifest):
        assert manifest['identity_gt_status'] == 'UNANNOTATED'

    def test_manifest_records_manual_identity_provenance(self, manifest):
        assert 'NOT generated from tracker' in manifest['identity_provenance']


class TestPreannotationCarriesNoIdentity:
    def test_det_rows_have_no_id(self, manifest):
        for s in manifest['sequences']:
            for line in (ROOT / s['preannotation_det']).read_text(
                    encoding='utf-8').splitlines():
                if line.strip():
                    assert line.split(',')[1] == '-1', s['sequence']

    def test_cvat_xml_has_no_tracks(self, manifest):
        """A <track> element in CVAT XML *is* an identity."""
        for s in manifest['sequences']:
            t = (ROOT / s['preannotation_cvat']).read_text(encoding='utf-8')
            assert '<track' not in t, s['sequence']
            assert '<box ' in t, s['sequence']

    def test_no_ball_in_preannotation(self, manifest):
        for s in manifest['sequences']:
            t = (ROOT / s['preannotation_cvat']).read_text(encoding='utf-8')
            assert 'label="ball"' not in t, s['sequence']

    def test_preannotation_labels_are_the_three_roles(self, manifest):
        for s in manifest['sequences']:
            t = (ROOT / s['preannotation_cvat']).read_text(encoding='utf-8')
            for r in HUMAN_CLASSES:
                assert f'<name>{r}</name>' in t


class TestPostStageRefusesWithoutAnnotation:
    def test_unannotated_package_cannot_pass_as_final_gt(self):
        errors, n = validate_post(ROOT)
        assert errors, 'an unannotated benchmark passed as final GT'
        assert any('UNANNOTATED' in e for e in errors)

    def test_mot_export_refuses_without_annotation(self):
        from tools.export_tracking_gt_mot import export
        with pytest.raises(SystemExit, match='REFUSING'):
            export(ROOT, ROOT / 'mot')


def _annotated(tmp_path, boxes, roles=None, status='VERIFIED'):
    """
    Minimal synthetic annotated package built from the real structure.

    When status is VERIFIED a matching QC confirmation record is written too,
    because VERIFIED without one is precisely the state the gate must reject.
    """
    from tools.confirm_tracking_gt_qc import promote_to_verified
    dst = tmp_path / 'gt'
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns('img1', 'qc'))
    man = json.loads((dst / 'manifest.json').read_text(encoding='utf-8'))
    man['identity_gt_status'] = status
    man['sequences'] = man['sequences'][:1]
    s = man['sequences'][0]
    (dst / 'annotations').mkdir(exist_ok=True)
    (dst / s['annotation_file_expected']).write_text(
        json.dumps({'boxes': boxes}), encoding='utf-8')
    (dst / 'roles').mkdir(exist_ok=True)
    ids = {b['id'] for b in boxes}
    (dst / s['roles_expected']).write_text(json.dumps(
        {'identity_roles': roles or {str(i): 'player' for i in ids}}), encoding='utf-8')
    (dst / 'manifest.json').write_text(json.dumps(man), encoding='utf-8')
    if status == 'VERIFIED':
        promote_to_verified(dst, man, reviewer='test')
    return dst, s


BOX = [10.0, 10.0, 50.0, 110.0]


class TestPostStageRejectsBadGt:
    """Content rules, exercised without a full 1,200-frame package on disk."""

    def test_valid_synthetic_gt_passes(self, tmp_path):
        boxes = [{'frame': 1, 'id': 1, 'bbox': BOX, 'role': 'player'},
                 {'frame': 2, 'id': 1, 'bbox': BOX, 'role': 'player'}]
        dst, _ = _annotated(tmp_path, boxes)
        errors, _ = validate_gt_content(dst)
        assert errors == [], errors[:5]

    def test_duplicate_id_in_one_frame_rejected(self, tmp_path):
        boxes = [{'frame': 1, 'id': 1, 'bbox': BOX, 'role': 'player'},
                 {'frame': 1, 'id': 1, 'bbox': [60.0, 10.0, 100.0, 110.0],
                  'role': 'player'}]
        dst, _ = _annotated(tmp_path, boxes)
        errors, _ = validate_gt_content(dst)
        assert any('duplicate GT id' in e for e in errors), errors[:5]

    def test_duplicate_identical_boxes_rejected(self, tmp_path):
        boxes = [{'frame': 1, 'id': 1, 'bbox': BOX, 'role': 'player'},
                 {'frame': 1, 'id': 2, 'bbox': BOX, 'role': 'player'}]
        dst, _ = _annotated(tmp_path, boxes)
        errors, _ = validate_gt_content(dst)
        assert any('duplicate identical boxes' in e for e in errors), errors[:5]

    @pytest.mark.parametrize('bbox,why', [
        ([50.0, 10.0, 10.0, 110.0], 'bbox extent'),
        ([10.0, 10.0, 10.0, 110.0], 'bbox extent'),
        ([10.0, 10.0, 5000.0, 110.0], 'outside frame'),
    ])
    def test_invalid_bbox_rejected(self, tmp_path, bbox, why):
        dst, _ = _annotated(tmp_path, [{'frame': 1, 'id': 1, 'bbox': bbox,
                                        'role': 'player'}])
        errors, _ = validate_gt_content(dst)
        assert any(why in e for e in errors), errors[:5]

    def test_invalid_role_rejected(self, tmp_path):
        dst, _ = _annotated(tmp_path, [{'frame': 1, 'id': 1, 'bbox': BOX,
                                        'role': 'coach'}],
                            roles={'1': 'coach'})
        errors, _ = validate_gt_content(dst)
        assert any('role' in e for e in errors), errors[:5]

    def test_ball_in_gt_rejected(self, tmp_path):
        dst, _ = _annotated(tmp_path, [{'frame': 1, 'id': 1, 'bbox': BOX,
                                        'role': 'player', 'class': 'ball'}])
        errors, _ = validate_gt_content(dst)
        assert any('ball' in e for e in errors), errors[:5]

    @pytest.mark.parametrize('bad_id', [0, -3, 'x'])
    def test_invalid_identity_rejected(self, tmp_path, bad_id):
        dst, _ = _annotated(tmp_path, [{'frame': 1, 'id': bad_id, 'bbox': BOX,
                                        'role': 'player'}],
                            roles={str(bad_id): 'player'})
        errors, _ = validate_gt_content(dst)
        assert errors

    def test_role_sidecar_must_cover_every_identity(self, tmp_path):
        boxes = [{'frame': 1, 'id': 1, 'bbox': BOX, 'role': 'player'},
                 {'frame': 1, 'id': 2, 'bbox': [60.0, 10.0, 100.0, 110.0],
                  'role': 'player'}]
        dst, _ = _annotated(tmp_path, boxes, roles={'1': 'player'})
        errors, _ = validate_gt_content(dst)
        assert any('role sidecar' in e for e in errors), errors[:5]


class TestMotExport:
    def test_row_format_matches_trackeval_expectations(self, tmp_path):
        """frame,id,x,y,w,h,conf,class,visibility -- 1-based, conf!=0, class==1."""
        from tools.export_tracking_gt_mot import export
        boxes = [{'frame': 1, 'id': 7, 'bbox': [10.0, 20.0, 50.0, 120.0],
                  'role': 'player'},
                 {'frame': 2, 'id': 7, 'bbox': [12.0, 20.0, 52.0, 120.0],
                  'role': 'player'}]
        dst, s = _annotated(tmp_path, boxes)
        export(dst, tmp_path / 'mot')
        gt = (tmp_path / 'mot' / 'EyeCU-val' / s['sequence'] / 'gt' / 'gt.txt')
        rows = [r.split(',') for r in gt.read_text(encoding='utf-8').splitlines()]
        assert len(rows) == 2
        f, i, x, y, w, h, conf, cls, vis = rows[0]
        assert int(f) == 1, 'frames must be 1-based'
        assert int(i) == 7
        assert (float(x), float(y), float(w), float(h)) == (10.0, 20.0, 40.0, 100.0)
        assert int(conf) == 1, 'TrackEval drops GT rows with conf == 0'
        assert int(cls) == 1, 'TrackEval only evaluates class 1 (pedestrian)'
        assert int(vis) == 1

    def test_export_writes_seqinfo_and_seqmap(self, tmp_path):
        from tools.export_tracking_gt_mot import export
        dst, s = _annotated(tmp_path, [{'frame': 1, 'id': 1, 'bbox': BOX,
                                        'role': 'player'}])
        export(dst, tmp_path / 'mot')
        assert (tmp_path / 'mot' / 'EyeCU-val' / s['sequence'] / 'seqinfo.ini').exists()
        sm = (tmp_path / 'mot' / 'seqmaps' / 'EyeCU-val.txt').read_text(encoding='utf-8')
        assert sm.splitlines()[0] == 'name'
        assert s['sequence'] in sm


class TestShippedWomen1Occlusion:
    """The annotator's occlusion marks are present in the canonical GT."""

    ANN = ROOT / 'annotations' / 'women_1_239.json'
    pytestmark = pytest.mark.skipif(not ANN.exists(),
                                    reason='women_1 not imported')

    @pytest.fixture(scope='class')
    def boxes(self):
        if not self.ANN.exists():
            pytest.skip('women_1 not imported')
        return json.loads(self.ANN.read_text(encoding='utf-8'))['boxes']

    def test_every_box_has_a_boolean_occluded(self, boxes):
        assert boxes
        assert all(isinstance(b['occluded'], bool) for b in boxes)

    def test_occlusion_matches_the_cvat_export(self, boxes):
        import re
        xml = (ROOT / 'cvat_exports' / 'women_1_239.xml').read_text(encoding='utf-8')
        from_xml = len([m.group(0) for m in re.finditer(r'<box [^>]*>', xml)
                        if 'occluded="1"' in m.group(0)
                        and 'outside="1"' not in m.group(0)])
        assert sum(b['occluded'] for b in boxes) == from_xml

    def test_no_visibility_fraction_was_invented(self, boxes):
        """occluded is what the annotator marked; a number would be fiction."""
        for b in boxes:
            assert 'visibility' not in b
            assert b['occluded'] in (True, False)


class TestGtStateMachine:
    """
    UNANNOTATED -> ANNOTATED_PENDING_QC -> VERIFIED.

    A two-state machine lets a manifest string edit turn an unreviewed import
    into an answer key, so the tests that matter here are the refusals.
    """

    def _export_fails(self, dst, tmp_path, why):
        from tools.export_tracking_gt_mot import export
        with pytest.raises(SystemExit, match='REFUSING'):
            export(dst, tmp_path / 'mot')
        errors, _ = validate_verified(dst)
        assert any(why in e for e in errors), errors[:5]

    def test_unannotated_cannot_export(self, tmp_path):
        dst, _ = _annotated(tmp_path, [{'frame': 1, 'id': 1, 'bbox': BOX,
                                        'role': 'player'}],
                            status='UNANNOTATED')
        self._export_fails(dst, tmp_path, 'only VERIFIED')

    def test_pending_qc_cannot_export(self, tmp_path):
        dst, _ = _annotated(tmp_path, [{'frame': 1, 'id': 1, 'bbox': BOX,
                                        'role': 'player'}],
                            status='ANNOTATED_PENDING_QC')
        self._export_fails(dst, tmp_path, 'only VERIFIED')

    def test_manual_status_edit_without_qc_record_is_rejected(self, tmp_path):
        """The whole point: asserting VERIFIED is not evidence of QC."""
        dst, _ = _annotated(tmp_path, [{'frame': 1, 'id': 1, 'bbox': BOX,
                                        'role': 'player'}],
                            status='ANNOTATED_PENDING_QC')
        man = json.loads((dst / 'manifest.json').read_text(encoding='utf-8'))
        man['identity_gt_status'] = 'VERIFIED'          # hand edit, nothing else
        (dst / 'manifest.json').write_text(json.dumps(man), encoding='utf-8')
        self._export_fails(dst, tmp_path, 'no QC confirmation record')

    def test_annotation_edited_after_qc_is_rejected(self, tmp_path):
        dst, s = _annotated(tmp_path, [{'frame': 1, 'id': 1, 'bbox': BOX,
                                        'role': 'player'}])
        (dst / s['annotation_file_expected']).write_text(json.dumps(
            {'boxes': [{'frame': 1, 'id': 99, 'bbox': BOX, 'role': 'player'}]}),
            encoding='utf-8')
        self._export_fails(dst, tmp_path, 'changed since QC confirmation')

    def test_verified_with_matching_qc_record_exports(self, tmp_path):
        from tools.export_tracking_gt_mot import export
        dst, s = _annotated(tmp_path, [{'frame': 1, 'id': 1, 'bbox': BOX,
                                        'role': 'player'}])
        export(dst, tmp_path / 'mot')
        gt = tmp_path / 'mot' / 'EyeCU-val' / s['sequence'] / 'gt' / 'gt.txt'
        assert gt.read_text(encoding='utf-8').strip()


class TestQcRendererIsReadOnly:
    def test_qc_does_not_modify_annotations(self, tmp_path):
        from tools.render_tracking_gt_qc import qc
        import hashlib
        boxes = [{'frame': 1, 'id': 1, 'bbox': BOX, 'role': 'player'},
                 {'frame': 1, 'id': 1, 'bbox': [60.0, 10.0, 100.0, 110.0],
                  'role': 'player'}]
        dst, s = _annotated(tmp_path, boxes)
        ann = dst / s['annotation_file_expected']
        before = hashlib.sha256(ann.read_bytes()).hexdigest()
        issues = qc(dst, s['sequence'], tmp_path / 'qcout', stride=1, render=False)
        assert hashlib.sha256(ann.read_bytes()).hexdigest() == before
        assert any('appears 2 times' in m for m in issues)

    def test_qc_flags_non_constant_role(self, tmp_path):
        from tools.render_tracking_gt_qc import qc
        boxes = [{'frame': 1, 'id': 1, 'bbox': BOX, 'role': 'player'},
                 {'frame': 2, 'id': 1, 'bbox': BOX, 'role': 'referee'}]
        dst, s = _annotated(tmp_path, boxes, roles={'1': 'player'})
        issues = qc(dst, s['sequence'], tmp_path / 'qcout', stride=1, render=False)
        assert any('role is not constant' in m for m in issues)
