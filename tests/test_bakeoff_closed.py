"""
The closed bake-off stays closed, and production stays put.

A completed experiment is only worth anything if its result cannot drift
afterwards. These assert the three ways that drift would actually happen here:
a result file being edited, the diagnostic profile being promoted to the answer
after the fact, and a modern tracker quietly reaching production.
"""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / 'experiments' / 'tracking_v2'
BAKE = EXP / 'bakeoff'
RECORD = BAKE / 'EXPERIMENT_RECORD.json'

pytestmark = pytest.mark.skipif(not RECORD.exists(), reason='bake-off not closed')


@pytest.fixture(scope='module')
def record():
    return json.loads(RECORD.read_text(encoding='utf-8'))


class TestExperimentIsClosed:
    def test_status_and_decision(self, record):
        assert record['status'] == 'COMPLETE'
        assert record['production_decision'] == 'KEEP_LEGACY_SUPERVISION_BYTETRACK'
        assert record['immutable'] is True

    def test_no_candidate_passed(self, record):
        assert record['outcome']['candidates_passing_adoption_criteria'] == []

    def test_every_preserved_result_still_hashes_the_same(self, record):
        """The whole point of recording hashes is that someone checks them."""
        for rel, want in record['preserved_artifacts']['result_files_sha256'].items():
            got = hashlib.sha256((BAKE / rel).read_bytes()).hexdigest()
            assert got == want, f'{rel} changed after the experiment closed'

    def test_frozen_inputs_and_contracts_still_hash_the_same(self, record):
        pa = record['preserved_artifacts']
        for rel, want in pa['frozen_input_detections_sha256'].items():
            p = REPO / 'data' / 'tracking_val_v1' / rel
            assert hashlib.sha256(p.read_bytes()).hexdigest() == want, rel
        for name, want in pa['contract_files_sha256'].items():
            assert hashlib.sha256((EXP / name).read_bytes()).hexdigest() == want, name

    def test_gt_still_matches_what_was_evaluated(self, record):
        gt = REPO / 'data' / 'tracking_val_gt'
        for rel, want in record['preserved_artifacts']['gt_sha256'].items():
            assert hashlib.sha256((gt / rel).read_bytes()).hexdigest() == want, rel


class TestLibraryDefaultsCannotWinRetroactively:
    def test_recorded_as_diagnostic_only(self, record):
        s = record['library_defaults_status']
        assert s['role_in_this_experiment'] == 'DIAGNOSTIC ONLY'
        assert 'cannot retroactively select' in s['explicit']

    def test_its_numbers_are_preserved_not_hidden(self, record):
        """Suppressing the inconvenient profile would be its own dishonesty."""
        hota = record['library_defaults_status']['combined_HOTA']
        assert hota['CBIoUTracker'] > hota['LEGACY_SUPERVISION_BYTETRACK']
        assert hota['BoTSORTTracker'] > hota['LEGACY_SUPERVISION_BYTETRACK']

    def test_the_decision_is_still_keep_legacy(self, record):
        assert record['outcome']['decision'] == 'KEEP_LEGACY_SUPERVISION_BYTETRACK'


class TestProductionUnchanged:
    def test_no_modern_tracker_in_production_code(self):
        prod = [REPO / 'trackers' / 'football_tracker.py',
                REPO / 'trackers' / 'detector.py',
                REPO / 'full_pipeline.py', REPO / 'run_pipeline.py']
        for p in prod:
            if not p.exists():
                continue
            src = p.read_text(encoding='utf-8')
            for name in ('BoTSORT', 'CBIoU', 'OCSORT', 'ByteTrackTracker',
                         'bakeoff'):
                assert name not in src, f'{name} reached {p.name}'

    def test_production_still_uses_supervision_bytetrack(self):
        src = (REPO / 'trackers' / 'football_tracker.py').read_text(encoding='utf-8')
        assert 'sv.ByteTrack()' in src
        assert 'update_with_detections' in src

    def test_external_trackers_is_not_a_production_dependency(self):
        req = (REPO / 'requirements.txt').read_text(encoding='utf-8')
        for line in req.splitlines():
            pkg = line.split('#')[0].strip().lower()
            assert not pkg.startswith('trackers'), line

    def test_local_trackers_package_is_the_one_that_resolves(self):
        import importlib.util
        origin = Path(importlib.util.find_spec('trackers').origin).resolve()
        assert REPO in origin.parents, 'production must import EyeCU trackers/'


class TestPostHocFindingAndT2Spec:
    @pytest.fixture(scope='class')
    def obs(self):
        return json.loads((EXP / 'post_hoc_observation.json'
                           ).read_text(encoding='utf-8'))

    @pytest.fixture(scope='class')
    def t2(self):
        return json.loads((EXP / 'T2_modern_default_policy_spec.json'
                           ).read_text(encoding='utf-8'))

    def test_observation_is_labelled_exploratory(self, obs):
        assert obs['evidence_class'] == 'EXPLORATORY'
        assert obs['hypothesis']['status'].startswith('HYPOTHESIS ONLY')

    def test_t2_is_specification_only_and_post_hoc(self, t2):
        assert t2['status'] == 'SPECIFICATION ONLY -- NOT EXECUTED'
        assert t2['evidence_class'] == 'POST-HOC / DEVELOPMENT'
        assert 'never be described as a confirmatory replication' in \
            t2['honesty_statement']

    def test_t2_candidates_are_the_two_plus_baseline(self, t2):
        assert set(t2['candidates']) == {'LEGACY_SUPERVISION_BYTETRACK',
                                         'CBIoUTracker', 'BoTSORTTracker'}
        assert set(t2['excluded_candidates']) == {'ByteTrackTracker',
                                                  'OCSORTTracker'}

    def test_t2_forbids_tuning_and_cmc_ablation(self, t2):
        assert 'changes no parameter' in t2['no_tuning_statement']
        assert 'Do NOT ablate CMC' in t2['predeclared_questions']['F_cmc_contribution']
        assert 'ablate BoTSORT CMC' in t2['not_to_be_done_in_T2']
        assert 'access TEST' in t2['not_to_be_done_in_T2']

    def test_t2_requires_end_to_end_runtime(self, t2):
        r = t2['runtime_requirements']
        assert 'END-TO-END' in r['scope']
        assert 'substituting a tracker-side figure' in r['forbidden']

    def test_t2_does_not_inherit_spec_1_1_silently(self, t2):
        assert 'TO BE PREDECLARED AND FROZEN BEFORE T2 RUNS' in t2['adoption_criteria']


class TestValidationSetStatus:
    @pytest.fixture(scope='class')
    def man(self):
        return json.loads((REPO / 'data' / 'tracking_val_gt' / 'manifest.json'
                           ).read_text(encoding='utf-8'))

    def test_val_is_now_a_development_benchmark(self, man):
        s = man['benchmark_usage_status']
        assert s['status'] == 'DEVELOPMENT / MODEL-SELECTION BENCHMARK'
        assert 'must NOT later be represented as an untouched final' in s['therefore']

    def test_test_split_untouched_and_gated(self, man):
        t = man['benchmark_usage_status']['test_split']
        assert t['status'] == 'UNTOUCHED' and t['accessed'] is False
        assert 'CBIoU' in t['forbidden'], 'TEST must not choose between candidates'
        assert len(t['preconditions']) >= 6
