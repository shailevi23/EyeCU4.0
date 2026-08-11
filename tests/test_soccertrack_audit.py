"""
The SoccerTrack v2 audit's load-bearing claims, as assertions.

Two of these exist because the audit's own first attempts were wrong and had to
be corrected: a proportional seek into a 2.7 GB JSON silently returned 15 of 22
boxes for one frame and none for another, and the naive per-axis rescale of
bbox_image onto the downloaded video put every box in the treeline. Both errors
would have produced a confident, wrong answer about whether this data can train
a detector.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
A = REPO / 'experiments' / 'soccertrack_audit'

pytestmark = pytest.mark.skipif(not (A / 'reports' / 'AUDIT_SUMMARY.json').exists(),
                                reason='SoccerTrack audit not present')


def load(rel):
    return json.loads((A / rel).read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def summary():
    return load('reports/AUDIT_SUMMARY.json')


class TestTheTwoAbsences:
    """No ball and no referee -- the findings that decide the verdict."""

    def test_no_ball_annotations_in_any_downloaded_half(self):
        scan = load('reports/gsr_scan.json')
        assert len(scan) == 20
        for f, v in scan.items():
            assert v['annotations_per_category']['ball'] == 0, f

    def test_no_referee_annotations_in_any_downloaded_half(self):
        scan = load('reports/gsr_scan.json')
        for f, v in scan.items():
            assert v['annotations_per_category']['referee'] == 0, f

    def test_ball_finding_is_stated_explicitly(self, summary):
        s = summary['ball_localization_finding']['statement']
        assert 'NO ball bounding boxes' in s and 'NO ball image coordinates' in s

    def test_bas_events_carry_no_ball_location(self):
        bas = load('reports/components.json')['bas']
        assert bas['events_with_ball_location'] == 0
        assert bas['total_events'] > 20000
        # 'position' must not be mistaken for a place
        assert 'neither is a place' in summary_position_note()


def summary_position_note():
    return load('reports/AUDIT_SUMMARY.json')['bas_utility']['timestamp_semantics']


class TestBoxesAreNotAnnotations:
    def test_boxes_are_wider_than_tall(self):
        g = load('reports/gsr_geometry.json')['128058_1st-015.json']['geometry']
        for cat in ('player', 'goalkeeper'):
            assert g[cat]['fraction_wider_than_tall'] > 0.99, cat
            assert g[cat]['aspect_h_over_w']['median'] < 1.0, cat

    def test_height_is_a_function_of_position_not_of_the_person(self, summary):
        e = summary['human_detector_utility']['box_tightness_evidence']
        assert e['height_distinct_values'] == 13
        assert e['corr_height_vs_image_row'] > 0.95
        assert e['height_std_overall_px'] < 2.0

    def test_no_usable_boxes_are_claimed(self, summary):
        u = summary['human_detector_utility']
        assert u['usable_player_boxes'] == 0
        assert u['usable_goalkeeper_boxes'] == 0
        assert u['usable_referee_boxes'] == 0
        assert u['verdict'] == 'NOT USABLE AS DETECTOR GROUND TRUTH'


class TestAlignmentWasSearchedNotAssumed:
    def test_bbox_image_alignment_is_not_established(self, summary):
        v = summary['video_alignment']
        assert v['verdict'].startswith('NOT ESTABLISHED for bbox_image')
        assert 'treeline' in ' '.join(v['evidence'])

    def test_the_offset_search_actually_ran_and_was_flat(self):
        al = load('reports/alignment_128058.json')
        assert al['video_size'] == [4096, 1080]
        assert al['gsr_declared_image_size'] == [3840, 1504]
        probes = [p for p in al['probes'] if p.get('spread_bbox_image') is not None]
        assert probes, 'no probe recorded a bbox_image offset sweep'
        # a real alignment would show a peak; every probe was flat
        assert all(p['spread_bbox_image'] < 0.05 for p in probes)

    def test_every_probe_recovered_the_full_22_annotations(self):
        """The seek-based reader lost boxes; the sequential one must not."""
        al = load('reports/alignment_128058.json')
        for p in al['probes']:
            if p['frame'] in (1200, 2400):
                assert p['annotations'] == 22, p['frame']


class TestMotIsUnansweredNotAnswered:
    def test_mot_archive_is_empty(self, summary):
        assert summary['inventory']['MOT']['bytes'] == 158
        assert 'EMPTY' in summary['inventory']['MOT']['status']

    def test_mot_findings_are_declared_unanswerable(self, summary):
        assert summary['mot_findings'].startswith('UNANSWERABLE')

    def test_no_mot_to_gsr_linkage_is_invented(self, summary):
        link = summary['mot_to_gsr_linkage']
        assert 'CANNOT be produced' in link['status']
        assert 'No inference is offered' in link['status']


class TestCalibrationIsVerified:
    def test_two_independent_routes_agree(self):
        c = load('reports/calibration_128058.json')
        assert c['keypoints'] == 65
        assert c['homography_route_error_px']['median'] < 15
        assert c['camera_model_route_error_px']['median'] < 15
        assert c['routes_agree_px']['median'] < 20

    def test_calibration_is_the_only_high_rating(self, summary):
        sc = summary['scorecard']
        high = [k for k, v in sc.items() if v['rating'] == 'HIGH']
        assert high == ['G_pitch_calibration_homography_value']

    def test_the_bad_calibration_is_flagged_not_averaged_away(self, summary):
        rms = summary['raw_calibration_finding']['per_match_calibration_quality_rms']
        assert rms['132831'] > 1000
        assert any('132831' in c for c in summary['raw_calibration_finding']['caveats'])


class TestScopeWasRespected:
    def test_no_leakage_into_frozen_splits(self, summary):
        lk = summary['eyecu_leakage_check']
        assert lk['EXTERNAL_vs_TRAIN'] == 0
        assert lk['EXTERNAL_vs_VAL'] == 0
        assert lk['EXTERNAL_vs_TEST'] == 0
        assert 'no TEST label' in lk['test_handling'].lower() or \
            'No TEST label' in lk['test_handling']

    def test_no_license_was_invented(self, summary):
        lic = summary['license_metadata']
        assert lic['shipped_with_the_download'] == 'NONE'
        assert 'not corroborated' in lic['note'].lower() or \
            'NOT corroborated' in lic['note']

    def test_integrity_flags(self, summary):
        i = summary['audit_integrity']
        assert i == {'downloaded_assets_modified': False,
                     'eyecu_train_val_test_modified': False,
                     'detector_or_tracker_changed': False,
                     'training_performed': False,
                     'test_evaluation_performed': False,
                     'further_download_initiated': False}
        assert summary['test_accessed_for_performance'] == 'NO'

    def test_recommendation_does_not_request_more_video(self, summary):
        d = summary['should_we_download_more']
        assert d['answer'] == 'CURRENT DOWNLOAD IS ENOUGH'
        assert d['single_exception']['not_a_match_download'] is True

    def test_archive_hashes_recorded(self):
        h = load('reports/archive_hashes.json')
        assert len(h) == 6
        # the duplicated raw archive under gsr/ must be recorded as identical
        raw1 = h['raw/raw-20260811T095104Z-1-001.zip']['sha256']
        assert h['gsr/raw-20260811T095104Z-1-001.zip']['sha256'] == raw1
