"""
Bake-off integrity: the run happened under the frozen rules, and the verdict
follows from the numbers rather than from anyone's preference.

These do not re-run trackers. They assert the properties that make the recorded
result trustworthy -- isolation, frozen inputs, valid outputs, and a verdict
that is arithmetic on the criteria as written.
"""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / 'experiments' / 'tracking_v2'
BAKE = EXP / 'bakeoff'
LEGACY = 'LEGACY_SUPERVISION_BYTETRACK'
PRIMARY = 'EYECU_SCORE_POLICY_V1'
MODERN = ['ByteTrackTracker', 'BoTSORTTracker', 'CBIoUTracker', 'OCSORTTracker']
SEQUENCES = {'women_1_239', 'youth_premier_league_1133',
             'bayern_munich_3-1_chelsea_228'}

pytestmark = pytest.mark.skipif(not (BAKE / 'adoption_verdict.json').exists(),
                                reason='bake-off not run')


@pytest.fixture(scope='module')
def env():
    return json.loads((BAKE / 'env_manifest.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def verdict():
    return json.loads((BAKE / 'adoption_verdict.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def primary():
    return json.loads((BAKE / 'trackeval_raw' / f'{PRIMARY}.json'
                       ).read_text(encoding='utf-8'))


class TestIsolation:
    def test_external_trackers_came_from_outside_the_repo(self, env):
        """The whole point: EyeCU's local trackers/ must not have been imported."""
        origin = Path(env['isolation']['external_trackers_import_path']).resolve()
        assert REPO not in origin.parents
        assert env['isolation']['inside_repo'] is False
        assert (REPO / 'trackers' / '__init__.py').exists(), 'the collision is real'

    def test_pinned_external_version(self, env):
        assert 'trackers==2.6.0' in env['isolated_pip_freeze']
        assert env['isolation']['trackers_wheel_sha256']

    def test_active_environment_untouched(self, env):
        assert env['isolation']['active_env_modified'] is False
        assert env['active_env']['supervision'] == '0.26.1', (
            'the legacy baseline must run on production supervision')

    def test_isolated_env_has_its_own_supervision(self, env):
        """A different supervision in the isolated env is fine and expected."""
        iso = [l for l in env['isolated_pip_freeze']
               if l.lower().startswith('supervision==')]
        assert iso and iso[0] != 'supervision==0.26.1'


class TestFrozenInputs:
    def test_detector_was_not_rerun(self, env):
        assert env['detector_rerun'] is False
        assert env['gt_used_to_filter_detections'] is False
        assert env['test_split_accessed'] is False

    def test_detection_hashes_still_match(self, env):
        for rel, want in env['frozen_input_hashes'].items():
            p = REPO / 'data' / 'tracking_val_v1' / rel
            assert hashlib.sha256(p.read_bytes()).hexdigest() == want, rel

    def test_no_austin_input_was_used(self, env):
        assert not any('austin' in k for k in env['frozen_input_hashes'])

    def test_contract_hashes_still_match(self, env):
        for name, want in env['frozen_contract_hashes'].items():
            p = (EXP / name if (EXP / name).exists()
                 else REPO / 'data' / 'tracking_val_gt' / 'qc' / name)
            assert hashlib.sha256(p.read_bytes()).hexdigest() == want, name


class TestOutputContract:
    @pytest.mark.parametrize('tracker', MODERN + [LEGACY])
    def test_every_output_row_is_well_formed(self, tracker):
        for seq in sorted(SEQUENCES):
            p = BAKE / 'outputs' / PRIMARY / 'EyeCU-val' / tracker / 'data' / f'{seq}.txt'
            assert p.exists(), p
            seen = set()
            for line in p.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                c = line.split(',')
                assert len(c) == 9
                f, i = int(c[0]), int(c[1])
                assert 1 <= f <= 300, 'frames are 1-based and in range'
                assert i > 0
                assert float(c[4]) > 0 and float(c[5]) > 0, 'positive box'
                assert c[7] == '1', 'one target-human class, not the role label'
                assert c[8] == '1'
                assert (f, i) not in seen, 'duplicate tracker id in a frame'
                seen.add((f, i))

    def test_no_tracker_output_was_silently_repaired(self):
        runs = json.loads((BAKE / 'run_summary.json').read_text(encoding='utf-8'))
        for key, v in runs['validation'].items():
            assert v['valid'], (key, v['issues'][:2])


class TestVerdictFollowsTheCriteria:
    def test_criteria_hash_is_the_frozen_one(self, verdict):
        got = hashlib.sha256((EXP / 'adoption_criteria.json').read_bytes()).hexdigest()
        assert verdict['criteria_sha256'] == got

    def test_every_modern_tracker_was_judged(self, verdict):
        assert set(verdict['trackers']) == set(MODERN)

    def test_criterion_1_is_arithmetic_on_combined_hota(self, verdict, primary):
        base = primary[LEGACY]['COMBINED_SEQ']['HOTA']
        for t, v in verdict['trackers'].items():
            delta = primary[t]['COMBINED_SEQ']['HOTA'] - base
            assert abs(v['criteria']['1_combined_HOTA_ge_+2.0']['value'] - delta) < 0.01
            assert v['criteria']['1_combined_HOTA_ge_+2.0']['pass'] == (delta >= 2.0)

    def test_a_failing_tracker_is_reported_as_failing(self, verdict):
        for t, v in verdict['trackers'].items():
            hard = [k for k, r in v['criteria'].items()
                    if r['pass'] is False]
            if hard:
                assert v['overall_pass'] is False, t

    def test_recommendation_matches_the_passing_set(self, verdict):
        if not verdict['passing_trackers']:
            assert verdict['recommendation'] == 'KEEP LEGACY SUPERVISION BYTETRACK'

    def test_library_defaults_did_not_decide_anything(self, verdict):
        assert verdict['profile'] == PRIMARY, (
            'the verdict must be computed on the primary profile only')

    def test_end_to_end_runtime_is_not_faked_from_tracker_side(self, verdict):
        """Criterion 8 is end-to-end; a tracker-side number is not a substitute."""
        for t, v in verdict['trackers'].items():
            c = v['criteria']['8_end_to_end_runtime_regression_le_10pct']
            if c['pass'] is None:
                assert 'NOT MEASURED' in c['value']


class TestRuntimeMeasurementIsControlled:
    def test_warmup_repeats_and_rotation(self):
        rt = json.loads((BAKE / 'runtime.json').read_text(encoding='utf-8'))
        assert rt['warmup'] >= 1
        assert rt['repeats'] >= 3
        assert 'rotated' in rt['order']
        assert rt['detector_excluded'] is True
        for t, v in rt['trackers'].items():
            for seq, samples in v['all_samples'].items():
                assert len(samples) == rt['repeats'], (t, seq)
