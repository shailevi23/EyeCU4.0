"""
The external-data audit's load-bearing claims, as assertions.

Three of these exist because the first pass of the audit got them wrong:
S1 hides segmentation polygons in a detection export, S6 letterboxes 16:9 into
a square canvas, and S4's 68 rotated copies are invisible to an orientation-
sensitive comparison. Each mistake inflated or deflated a number that a future
training decision would rest on, so each is pinned here.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
A = REPO / 'experiments' / 'external_data_audit'

pytestmark = pytest.mark.skipif(not (A / 'reports' / 'AUDIT_SUMMARY.json').exists(),
                                reason='external data audit not present')


def load(rel):
    return json.loads((A / rel).read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def summary():
    return load('reports/AUDIT_SUMMARY.json')


@pytest.fixture(scope='module')
def inventory():
    return load('reports/inventory.json')


class TestNoLeakageIntoFrozenSplits:
    """The one result that would invalidate every future number if wrong."""

    def test_no_external_image_touches_val_or_test(self, summary):
        assert summary['leakage']['EXTERNAL_vs_VAL']['pairs'] == 0
        assert summary['leakage']['EXTERNAL_vs_TEST']['pairs'] == 0

    def test_leakage_search_was_orientation_invariant(self):
        lk = load('reports/leakage.json')['summary']
        assert '8 dihedral orientations' in lk['method']
        # a rotation-sensitive search would have found 223, not 295
        assert lk['by_source_and_split']['S4->EYECU_TRAIN']['orientations']['rot270'] == 72

    def test_s4_is_entirely_already_in_eyecu_train(self, summary):
        v = summary['leakage']['EXTERNAL_vs_TRAIN']
        assert v['external_images'] == 272
        assert v['sources'] == ['S4']
        assert summary['verdicts']['S4']['decision'] == 'REJECT'

    def test_test_was_never_evaluated(self, summary):
        assert summary['test_accessed_for_performance'] == 'NO'
        assert 'No TEST label was opened' in summary['leakage']['test_handling']


class TestBallMeasurementIsNotOverstated:
    def test_stored_and_native_are_both_reported(self, summary):
        b = summary['ball_size']
        assert 'combined_stored' in b and 'combined_native_equivalent' in b
        assert 'cannot be measured' in b['measurement_note']

    def test_the_tiny_ball_count_collapses_under_back_projection(self, summary):
        b = summary['ball_size']
        assert b['combined_stored']['le8'] == 857
        assert b['combined_native_equivalent']['le8'] == 4

    def test_external_balls_are_not_smaller_than_eyecus_own(self, summary):
        """The whole premise of the audit, stated as a check."""
        b = summary['ball_size']
        assert b['eyecu_own_reference']['median_px'] == 6.0
        assert b['combined_native_equivalent']['median_px'] >= 18.0


class TestParsingDefectsThatChangedTheNumbers:
    def test_s1_segmentation_polygons_are_counted_not_misread(self, inventory):
        assert inventory['sources']['S1']['problems']['segmentation_polygon_rows'] == 84
        # misreading them as cx,cy,w,h produced 48 phantom out-of-bounds boxes
        assert 'out_of_bounds' not in inventory['sources']['S1']['problems']

    def test_s6_letterbox_is_measured_not_assumed(self, inventory):
        s6 = inventory['sources']['S6']
        x, y, w, h = s6['measured_content_box_xywh']
        assert (w, h) == (1280, 721)
        assert 'content height 721' in s6['ball']['projection']

    def test_numeric_classes_were_confirmed_visually(self):
        cm = load('reports/class_map.json')
        for sid in ('S4', 'S6'):
            assert cm[sid]['mapping'] == {'0': 'ball', '1': 'goalkeeper',
                                          '2': 'player', '3': 'referee'}
            for row in cm[sid]['rows']:
                assert row['confidence'] == 'HIGH'
                assert '_class' in row['evidence'] or 'jpg' in row['evidence']


class TestAnnotationCompleteness:
    def test_s5_is_ball_only(self):
        c = load('reports/annotation_completeness.json')
        assert c['S5']['verdict'] == 'BALL_ONLY'
        assert c['S5']['images_without_human_labels'] == 487

    def test_s6_partial_annotation_is_flagged(self):
        c = load('reports/annotation_completeness.json')
        assert c['S6']['verdict'] == 'PARTIAL_ANNOTATION_LIKELY'
        assert c['S6']['zero_box_images'] == 342

    def test_sources_without_a_goalkeeper_class_are_flagged(self):
        c = load('reports/annotation_completeness.json')
        assert [s for s, v in c.items() if v['goalkeeper_class_absent']] == \
            ['S1', 'S3', 'S5']


class TestCandidateIndexIsMetadataOnly:
    def test_every_image_has_exactly_one_status(self):
        idx = load('candidate_index/candidate_index.json')
        allowed = {'KEEP_CANDIDATE', 'HUMAN_REVIEW', 'EXCLUDE_EXACT_DUPLICATE',
                   'EXCLUDE_NEAR_DUPLICATE', 'EXCLUDE_VAL_TEST_LEAKAGE',
                   'EXCLUDE_POOR_LABEL', 'EXCLUDE_IRRELEVANT',
                   'EXCLUDE_PARTIAL_ANNOTATION_RISK', 'EXCLUDE_AUGMENTATION_COPY'}
        assert len(idx) == 3594
        for r in idx:
            assert r['status'] in allowed
            if r['status'] != 'KEEP_CANDIDATE':
                assert r['reasons'], r['path']

    def test_index_points_at_the_audit_workspace_only(self):
        idx = load('candidate_index/candidate_index.json')
        for r in idx:
            assert r['path'].startswith('experiments/external_data_audit/extracted/')

    def test_nothing_was_copied_into_eyecu_data(self):
        for split in ('train', 'val'):
            d = REPO / 'data' / 'dataset_baseline' / 'images' / split
            assert not any(p.name.startswith(('barca_frame', 'USA_NED', 'youtube-'))
                           for p in d.iterdir())


class TestIntegrity:
    def test_all_integrity_checks_pass(self):
        i = load('reports/integrity.json')
        assert i['all_pass'] is True
        assert [c for c in i['checks'] if not c['pass']] == []

    def test_zip_hashes_match_the_recorded_sources(self):
        srcs = load('raw/SOURCES.json')['sources']
        state = load('reports/integrity.json')['state']['zips']
        for sid, s in srcs.items():
            assert state[sid]['sha256'] == s['sha256']

    def test_no_model_or_frozen_artifact_changed(self):
        i = load('reports/integrity.json')
        names = {c['check'] for c in i['checks'] if c['pass']}
        for f in ('best_A_960.pt', 'trackers/football_tracker.py',
                  'experiments/tracking_v2/integration/TRACKER_FREEZE.json',
                  'EYECU_TEST_IMAGES', 'EYECU_VAL'):
            assert f in names
