"""
T2 ADOPTION GATE v1.0 is frozen before execution and cannot move afterwards.

The failure this guards against is specific and easy: T2 runs, a candidate
misses one criterion by a little, and the criterion quietly becomes what the
candidate achieved. A gate that can be edited after results exist is not a gate.

The spec hash is pinned below. Editing the spec fails these tests, which is the
intended friction -- the gate may only change while nothing has been executed,
and changing it then still means consciously updating this constant and saying
so.
"""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / 'experiments' / 'tracking_v2'
SPEC = EXP / 'T2_modern_default_policy_spec.json'

# sha256 of T2_modern_default_policy_spec.json at the moment the gate was frozen
SPEC_SHA256 = '5f8d2a752c57aef1fd4fd82719d2b6bfd9bc8c7cb9e28b9433b114761a756f60'

# results T2 would produce; their presence means execution has begun
T2_RESULT_PATHS = [EXP / 't2', EXP / 'bakeoff' / 't2']

pytestmark = pytest.mark.skipif(not SPEC.exists(), reason='T2 not specified')


@pytest.fixture(scope='module')
def spec():
    return json.loads(SPEC.read_text(encoding='utf-8'))


def t2_has_been_executed():
    return any(p.exists() for p in T2_RESULT_PATHS)


class TestSpecIsFrozen:
    def test_spec_hash_is_the_frozen_one(self):
        got = hashlib.sha256(SPEC.read_bytes()).hexdigest()
        assert got == SPEC_SHA256, (
            'The T2 specification changed. If T2 has not been executed this is '
            'a deliberate re-freeze and this constant must be updated in the '
            'same commit, with the reason recorded. If T2 HAS been executed, '
            'the gate must not change at all.')

    def test_gate_cannot_change_after_execution_begins(self, spec):
        if t2_has_been_executed():
            assert hashlib.sha256(SPEC.read_bytes()).hexdigest() == SPEC_SHA256
            assert spec['adoption_gate']['frozen_before_execution'] is True

    def test_execution_state_matches_reality(self, spec):
        if spec['execution_state'] == 'NOT_STARTED':
            assert not t2_has_been_executed(), (
                'results exist but the spec still claims NOT_STARTED')

    def test_amendment_rule_is_recorded(self, spec):
        r = spec['adoption_gate']['amendment_rule']
        assert 'may be altered' in r or 'No criterion may be altered' in r
        assert 'tie-breaker' in r


class TestEvidenceClassification:
    def test_post_hoc_development(self, spec):
        assert spec['evidence_classification'] == 'POST-HOC / DEVELOPMENT'
        assert spec['evidence_role'] == 'DEVELOPMENT QUALIFICATION / REPRODUCIBILITY'
        assert spec['status'] == 'SPECIFICATION FROZEN -- NOT EXECUTED'

    def test_t2_is_not_independent_confirmation(self, spec):
        c = spec['methodological_correction']
        assert 'must NOT be described as a new independent confirmation' in c['correction']
        assert 'does NOT make its accuracy result independent confirmatory' in c['explicit']

    def test_what_t2_actually_adds_is_recorded(self, spec):
        adds = ' | '.join(spec['methodological_correction']['what_t2_actually_adds'])
        for k in ('reproducibility', 'per-match', 'END-TO-END', 'invariant'):
            assert k in adds, k

    def test_confirmatory_replication_is_forbidden_language(self, spec):
        assert 'describe T2 as confirmatory replication' in spec['not_to_be_done_in_T2']


class TestGateCriteria:
    def test_all_nine_criteria_present(self, spec):
        c = spec['adoption_gate']['criteria']
        assert list(c) == ['1_reproducibility', '2_combined_HOTA',
                           '3_combined_AssA', '4_combined_IDF1',
                           '5_per_match_HOTA_robustness',
                           '6_per_match_identity_robustness',
                           '7_eyecu_invariants',
                           '8_controlled_end_to_end_runtime', '9_no_tuning']

    def test_all_criteria_must_pass(self, spec):
        assert 'must satisfy ALL criteria' in spec['adoption_gate']['rule']

    def test_reproducibility_tolerances(self, spec):
        c = spec['adoption_gate']['criteria']['1_reproducibility']
        assert c['metric_tolerance_vs_closed_bakeoff'] == {
            'HOTA': 0.10, 'AssA': 0.10, 'IDF1': 0.10}
        assert 'STOP qualification' in c['on_drift']
        assert 'averaging away' in c['forbidden']

    @pytest.mark.parametrize('key,needle', [
        ('2_combined_HOTA', '>= +2.0'),
        ('3_combined_AssA', '>= +3.0'),
        ('4_combined_IDF1', '>= +3.0'),
        ('5_per_match_HOTA_robustness', '>= 2 of the 3'),
        ('6_per_match_identity_robustness', '2.0 absolute IDF1'),
    ])
    def test_numeric_thresholds_are_as_frozen(self, spec, key, needle):
        assert needle in spec['adoption_gate']['criteria'][key]['requirement']

    def test_idf1_must_improve_not_merely_not_regress(self, spec):
        c = spec['adoption_gate']['criteria']['4_combined_IDF1']
        assert 'improvement >= +3.0' in c['requirement']
        assert 'not sufficient' in c['why_stricter_than_the_closed_experiment']

    def test_per_match_hota_covers_both_halves(self, spec):
        r = spec['adoption_gate']['criteria']['5_per_match_HOTA_robustness']['requirement']
        assert '>= 2 of the 3' in r and 'no individual match regresses by more than 2.0' in r

    def test_invariants_are_all_listed(self, spec):
        inv = spec['adoption_gate']['criteria']['7_eyecu_invariants']['invariants']
        assert len(inv) == 9
        joined = ' | '.join(inv)
        for k in ('ball completely isolated', 'goalkeeper never normalised',
                  'duplicate tracker identity', 'no hidden detector rerun'):
            assert k in joined, k

    def test_runtime_must_be_end_to_end(self, spec):
        c = spec['adoption_gate']['criteria']['8_controlled_end_to_end_runtime']
        assert '<= +10%' in c['requirement']
        assert c['must_measure'] == 'the REAL EyeCU pipeline'
        assert any('tracker-side runtime alone' in f for f in c['forbidden'])
        assert any('contaminated' in f for f in c['forbidden'])
        for k in ('detector ms/frame', 'tracker ms/frame', 'total ms/frame',
                  'effective FPS'):
            assert k in c['report_separately']
        assert c['controls'].count('declared warmup') == 1
        assert '>= 3 measured repeats' in c['controls']
        assert 'rotated candidate order' in c['controls']

    def test_no_tuning_list_is_complete(self, spec):
        f = spec['adoption_gate']['criteria']['9_no_tuning']['forbidden']
        for k in ('confidence threshold changes', 'activation threshold changes',
                  'buffer changes', 'association threshold changes',
                  'CMC changes or ablation', 'per-match configuration',
                  'intermediate score-policy profile', 'Optuna', 'grid search'):
            assert k in f, k


class TestSelectionRule:
    def test_all_four_branches(self, spec):
        r = spec['adoption_gate']['selection_rule']
        assert r['neither_passes'] == 'KEEP LEGACY_SUPERVISION_BYTETRACK'
        assert 'higher COMBINED_SEQ HOTA' in r['both_pass']
        assert 'lower controlled END-TO-END ms/frame' in r['both_pass_and_HOTA_within_0.5']
        assert 'select that candidate' in r['one_candidate_passes_all']

    def test_rule_is_labelled_engineering_not_science(self, spec):
        r = spec['adoption_gate']['selection_rule']
        assert r['status'].startswith('ENGINEERING DEVELOPMENT RULE')
        assert 'NOT be represented as independent scientific confirmation' in r['explicit']


class TestCandidatesAndTestSeal:
    def test_candidates_unchanged(self, spec):
        assert set(spec['candidates']) == {'LEGACY_SUPERVISION_BYTETRACK',
                                           'CBIoUTracker', 'BoTSORTTracker'}
        assert set(spec['excluded_candidates']) == {'ByteTrackTracker',
                                                    'OCSORTTracker'}

    def test_cmc_ablation_forbidden(self, spec):
        assert 'Do NOT ablate CMC in T2' in \
            spec['predeclared_questions']['F_cmc_contribution']
        assert 'ablate BoTSORT CMC' in spec['not_to_be_done_in_T2']

    def test_test_split_sealed(self, spec):
        t = spec['test_split']
        assert t['status'] == 'SEALED' and t['accessed'] is False
        assert 'CBIoU' in t['may_not']
        assert 'once' in t['may_only_be_evaluated']

    def test_val_status_warning_present(self, spec):
        assert 'development / model-selection benchmark' in \
            spec['validation_set_status_warning']
