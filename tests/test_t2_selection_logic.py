"""
The T2 selection branch, and why the tie-break must not fire.

I originally recorded CBIoU as winning a <0.5 HOTA tie-break against BoTSORT.
That branch presupposes both candidates passed all nine mandatory criteria.
BoTSORT's criterion 8 was NOT ESTABLISHED -- no valid paired end-to-end
measurement existed -- and an unestablished mandatory criterion is not a pass.
The answer was right; the reasoning was not, and a tie-break able to run on an
unqualified candidate would admit a future unmeasured tracker into a comparison
it is not eligible for.
"""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
E = REPO / 'experiments' / 'tracking_v2'
V = E / 't2_corrected' / 'T2_CORRECTED_verdict.json'
F = E / 'integration' / 'TRACKER_FREEZE.json'

pytestmark = pytest.mark.skipif(not V.exists(), reason='corrected T2 absent')


@pytest.fixture(scope='module')
def verdict():
    return json.loads(V.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def freeze():
    return json.loads(F.read_text(encoding='utf-8'))


def _passed_all(c):
    return all(v[0] is True for v in c['criteria'].values())


class TestSelectionBranch:
    def test_not_established_is_not_a_pass(self, verdict):
        c8 = verdict['candidates']['BoTSORTTracker']['criteria']['8_runtime<=+10%']
        assert c8[0] is not True
        assert 'NOT ESTABLISHED' in c8[1]

    def test_botsort_did_not_pass_all_criteria(self, verdict):
        assert not _passed_all(verdict['candidates']['BoTSORTTracker'])
        assert verdict['candidates']['BoTSORTTracker']['overall_pass'] is False

    def test_cbiou_passed_all_criteria(self, verdict):
        assert _passed_all(verdict['candidates']['CBIoUTracker'])
        assert verdict['candidates']['CBIoUTracker']['overall_pass'] is True

    def test_tie_break_did_not_fire(self, verdict):
        assert 'NOT INVOKED' in verdict['tie_break']

    def test_tie_break_requires_two_fully_passing_candidates(self, verdict):
        """The guard stated as an invariant rather than as prose."""
        passing = [t for t, c in verdict['candidates'].items() if _passed_all(c)]
        if len(passing) < 2:
            assert 'NOT INVOKED' in verdict['tie_break'], (
                'a tie-break ran with fewer than two qualified candidates')

    def test_selection_is_the_exactly_one_branch(self, verdict):
        passing = [t for t, c in verdict['candidates'].items() if _passed_all(c)]
        assert passing == ['CBIoUTracker']
        assert verdict['selection'] == 'CBIoUTracker'
        assert 'exactly one' in verdict['selection_logic_correction']['correct_branch']

    def test_qualification_labels(self, verdict):
        q = verdict['qualification']
        assert q['CBIoUTracker'] == 'QUALIFIED / SELECTED'
        assert 'NOT QUALIFIED' in q['BoTSORTTracker']

    def test_correction_is_recorded_not_silently_applied(self, verdict):
        c = verdict['selection_logic_correction']
        assert 'tie-break' in c['error']
        assert 'not a pass' in c['error']


class TestFreezeReferencesCorrectedResults:
    PATHS = {
        'corrected_bakeoff_primary':
            E / 'bakeoff_corrected/trackeval_raw/EYECU_SCORE_POLICY_V1.json',
        'corrected_bakeoff_diagnostic':
            E / 'bakeoff_corrected/trackeval_raw/LIBRARY_DEFAULTS.json',
        'corrected_t2_metrics': E / 't2_corrected/trackeval_raw/T2.json',
        'corrected_t2_verdict': V,
        'corrected_t2_run_report': E / 't2_corrected/run_report.json',
        'equivalence_vs_corrected_reference': E / 'integration/equivalence.json',
        'original_paired_runtime': E / 'integration/paired_runtime.json',
        'post_id_normalization_runtime_sanity':
            E / 'integration/POST_ID_NORMALIZATION_RUNTIME_SANITY.json',
    }

    def test_authoritative_hashes_still_match(self, freeze):
        for k, p in self.PATHS.items():
            assert hashlib.sha256(p.read_bytes()).hexdigest() == \
                freeze['authoritative_corrected_hashes'][k], k

    def test_frozen_contract_fields(self, freeze):
        assert freeze['production_tracker'] == 'CBIoUTracker'
        assert freeze['tracker_status'] == 'FROZEN'
        assert freeze['public_identity_contract'] == 'positive integer IDs'
        assert freeze['modern_raw_id_mapping'] == 'public_id = raw_id + 1'
        assert 'not emitted' in freeze['raw_id_negative']
        assert 'legacy' in freeze['legacy_rollback']
        assert freeze['tie_break_invoked'] is False

    def test_prefix_history_still_marked_invalid(self):
        for f in ('bakeoff/EXPERIMENT_RECORD.json', 't2/T2_EXPERIMENT_RECORD.json'):
            d = json.loads((E / f).read_text(encoding='utf-8'))
            assert d['adapter_bug_invalidation']['status'] == \
                'INVALIDATED_FOR_TRACKER_SELECTION'

    def test_runtime_sanity_is_separate_valid_and_far_from_the_boundary(self, freeze):
        s = json.loads(self.PATHS['post_id_normalization_runtime_sanity']
                       .read_text(encoding='utf-8'))
        assert s['validity']['valid'] is True
        assert s['paired_pct_vs_legacy'] < 10.0
        orig = json.loads(self.PATHS['original_paired_runtime']
                          .read_text(encoding='utf-8'))
        assert orig['paired_pct_vs_legacy'] == -0.115, (
            'the original valid paired artifact must not be overwritten')
