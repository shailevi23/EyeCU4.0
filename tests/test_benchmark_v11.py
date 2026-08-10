"""
EyeCU-Tracking-Val-v1.1: three independent clean-continuity matches.

The revision removed a window after it was found to straddle a broadcast
dissolve. Two things must stay true no matter what anyone runs later: the
rejected window cannot creep back into the clean aggregate, and the criteria
frozen at the moment of the revision cannot drift once results exist. Both are
easy to violate by accident and impossible to notice by reading a number.
"""

import json
from pathlib import Path

import pytest

from tools.validate_tracking_gt import EXCLUDED, EXPECTED

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'data' / 'tracking_val_gt'
CRITERIA = REPO / 'experiments' / 'tracking_v2' / 'adoption_criteria.json'
AUSTIN = 'austin_fc_vs__club_tijuana_284'

pytestmark = pytest.mark.skipif(not (ROOT / 'manifest.json').exists(),
                                reason='identity GT package not built')


@pytest.fixture(scope='module')
def manifest():
    return json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def criteria():
    return json.loads(CRITERIA.read_text(encoding='utf-8'))


class TestBenchmarkDefinition:
    def test_three_independent_matches_900_frames(self, manifest):
        d = manifest['benchmark_definition']
        assert manifest['benchmark'] == 'EyeCU-Tracking-Val-v1.1'
        assert d['independent_matches'] == 3
        assert d['total_frames'] == 900
        assert len(manifest['sequences']) == 3
        assert sum(s['frame_count'] for s in manifest['sequences']) == 900

    def test_the_three_are_the_clean_ones(self, manifest):
        assert {s['sequence'] for s in manifest['sequences']} == EXPECTED
        assert EXPECTED == {'bayern_munich_3-1_chelsea_228', 'women_1_239',
                            'youth_premier_league_1133'}

    def test_every_sequence_is_a_distinct_match(self, manifest):
        """No correlated second window: sequence count must equal match count."""
        matches = [s['match'] for s in manifest['sequences']]
        assert len(matches) == len(set(matches))

    def test_no_train_or_test_footage(self, manifest):
        from tools.build_derived_train import TEST_MATCHES, VAL_MATCHES
        srcs = {s['match'] for s in manifest['sequences']}
        assert srcs <= VAL_MATCHES
        assert not (srcs & TEST_MATCHES)


class TestAustinCannotComeBack:
    def test_austin_is_not_a_sequence(self, manifest):
        assert AUSTIN in EXCLUDED
        assert AUSTIN not in {s['sequence'] for s in manifest['sequences']}

    def test_austin_is_recorded_as_a_stress_case_not_deleted(self, manifest):
        ex = manifest['excluded_sequences']
        assert len(ex) == 1
        a = ex[0]
        assert a['original_entry']['sequence'] == AUSTIN
        assert a['status'] == 'REJECTED_FOR_CONTINUITY_BENCHMARK'
        assert a['classification'] == 'TRANSITION_STRESS_CASE'
        assert 'discontinuity' in a['reason']

    def test_austin_never_enters_the_clean_aggregate(self, manifest):
        never = manifest['excluded_sequences'][0]['never_enters']
        for metric in ('HOTA', 'AssA', 'IDF1'):
            assert any(metric in n for n in never), metric

    def test_partial_annotation_is_marked_invalid_as_continuity_gt(self, manifest):
        a = manifest['excluded_sequences'][0]
        v = a['partial_annotation_validity']
        assert 'NOT valid continuity GT' in v
        assert 'INDEPENDENT identity spaces' in a['if_reused']

    def test_the_record_does_not_credit_the_annotator_with_the_residue(self, manifest):
        """
        The post-transition boxes are a CVAT hold, not a human claim.

        Reading a tool artifact as an assertion of physical identity would
        misrepresent the annotator and, worse, make the residue look like GT
        someone stood behind.
        """
        a = manifest['excluded_sequences'][0]
        assert a['human_annotated_span'] == {'package_frames': [1, 101],
                                             'boxes': 1616}
        assert a['residual_span']['boxes'] == 3184
        assert 'not annotation' in a['residual_span']['what_it_is']
        v = a['partial_annotation_validity']
        assert 'did NOT' in v and 'assert identity continuity' in v

    def test_rejection_record_measures_where_work_stopped(self):
        rec = json.loads((ROOT / 'rejected' / AUSTIN / 'REJECTION_RECORD.json'
                          ).read_text(encoding='utf-8'))
        w = rec['annotation_work_preserved']['where_human_work_stops']
        m = w['measured_in_the_export']
        assert m['last_manual_keyframe'] == {'cvat_frame': 100,
                                             'package_frame': 101}
        assert m['identical_for_all_16_tracks'] is True
        assert m['manual_keyframes_after_that_point'] == 0
        assert rec['annotation_work_preserved']['modified_since_export'] is False

    def test_the_earlier_mischaracterisation_is_recorded_not_erased(self):
        rec = json.loads((ROOT / 'rejected' / AUSTIN / 'REJECTION_RECORD.json'
                          ).read_text(encoding='utf-8'))
        c = rec['annotation_work_preserved']['correction_of_an_earlier_statement']
        assert 'run unbroken' in c['what_was_recorded_before']
        assert c['why_that_was_wrong']

    def test_the_rejection_record_and_annotation_are_preserved(self):
        d = ROOT / 'rejected' / AUSTIN
        assert (d / 'REJECTION_RECORD.json').exists()
        assert (d / 'austin_partial_cvat_export.xml').exists()
        rec = json.loads((d / 'REJECTION_RECORD.json').read_text(encoding='utf-8'))
        assert rec['annotation_work_preserved']['deleted'] is False
        assert rec['annotation_work_preserved']['counted_in_clean_aggregate'] is False
        assert rec['tracker_output_used'] is False

    def test_the_annotation_vehicle_is_out_of_the_way(self):
        """An MP4 in cvat_video named after a rejected window invites work."""
        assert not (ROOT / 'cvat_video' / f'{AUSTIN}.mp4').exists()
        assert (ROOT / 'rejected' / AUSTIN / f'{AUSTIN}.mp4').exists()

    def test_history_is_not_rewritten(self, manifest):
        hist = {h['version']: h for h in manifest['revision_history']}
        assert set(hist) == {'EyeCU-Tracking-Val-v1', 'EyeCU-Tracking-Val-v1.1'}
        assert hist['EyeCU-Tracking-Val-v1']['status'] == 'SUPERSEDED'
        assert AUSTIN in hist['EyeCU-Tracking-Val-v1']['sequences']
        assert hist['EyeCU-Tracking-Val-v1']['independent_matches'] == 4
        assert hist['EyeCU-Tracking-Val-v1.1']['status'] == 'CURRENT'


class TestFrozenAdoptionCriteria:
    def test_spec_matches_the_revised_benchmark(self, criteria):
        assert criteria['spec_version'] == '1.1'
        assert criteria['benchmark'] == 'EyeCU-Tracking-Val-v1.1'
        assert criteria['independent_matches'] == 3
        assert criteria['total_frames'] == 900
        assert set(criteria['sequences']) == EXPECTED

    @pytest.mark.parametrize('needle', [
        'HOTA improvement >= +2.0',
        'AssA improvement >= +3.0',
        'IDF1 regression <= 0.5',
        '>= 2 of the 3 independent matches',
        'not regress by > 2.0 HOTA',
        'catastrophic identity failure',
        'invariants preserved',
        'runtime regression <= 10%',
    ])
    def test_each_frozen_criterion_is_present(self, criteria, needle):
        joined = ' | '.join(criteria['adoption_criteria']['primary'])
        assert needle in joined, joined

    def test_criteria_are_frozen_before_results(self, criteria):
        a = criteria['adoption_criteria']
        assert a['frozen_before_results'] is True
        assert 'may not be altered after any bake-off result' in a['amendment_rule']

    def test_library_defaults_cannot_select_the_winner(self, criteria):
        assert 'DIAGNOSTIC ONLY' in criteria['adoption_criteria']['profile_authority']

    def test_the_old_four_sequence_criteria_are_kept_as_superseded(self, criteria):
        old = criteria['superseded_adoption_criteria'][0]
        assert old['spec_version'] == '1.0'
        assert old['status'] == 'INVALID'
        assert AUSTIN in old['sequences']
        assert any('3 of 4' in c for c in old['criteria']['primary'])

    def test_the_revision_reason_disclaims_tracker_influence(self, criteria):
        r = criteria['revision_reason']
        assert 'no tracker output' in r['why_this_is_not_post_hoc']
        assert 'before any tracker was run' in r['when']
        for forbidden in ('TRAIN', 'TEST', 'correlated'):
            assert any(forbidden in n for n in r['not_done']), forbidden


class TestReportingViews:
    def test_three_views_are_declared_with_precedence(self, manifest):
        v = manifest['reporting_views']
        assert set(v['A_per_match']) == EXPECTED
        assert 'COMBINED_SEQ' in v['B_official']
        assert 'macro' in v['C_macro_mean'] or 'mean' in v['C_macro_mean']
        assert 'never replaces' in v['precedence']

    def test_criteria_declare_the_same_views(self, criteria):
        v = criteria['reporting_views']
        assert set(v['A_per_match']) == EXPECTED
        assert v['B_official_combined'] == 'TrackEval COMBINED_SEQ'
