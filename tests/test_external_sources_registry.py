"""
The consolidated external-source workspace, as assertions.

The registry is the thing future decisions will be made from, so the checks here
are mostly about it not drifting from the audits it was built out of, and about
the rules that protect EyeCU's frozen artifacts holding: no external image in
VAL or TEST, no dataset bulk in git, no credential on disk, and no source
promoted to KEEP_ACTIVE without the evidence that justifies it.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / 'EyeCU_external_data'
XS = REPO / 'experiments' / 'external_sources'
REG = EXT / 'MASTER_EXTERNAL_SOURCE_REGISTRY.json'

pytestmark = pytest.mark.skipif(not REG.exists(), reason='registry not built')


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def reg():
    return load(REG)


@pytest.fixture(scope='module')
def by_id(reg):
    return {r['SOURCE_ID']: r for r in reg['sources']}


class TestRegistryShape:
    REQUIRED = ['SOURCE_ID', 'NAME', 'ORIGIN', 'URL_OR_REPO_ID', 'SOURCE_TYPE',
                'LICENSE', 'LICENSE_VERIFIED_FROM', 'LOCAL_PATH', 'DOWNLOAD_STATUS',
                'SIZE', 'HASH_OR_REVISION', 'DOMAIN', 'CLASSES', 'BALL_BOXES',
                'GK_SEPARATE', 'REFEREE_AVAILABLE', 'PARTIAL_ANNOTATION_RISK',
                'DUPLICATE_RISK', 'TRAIN_OVERLAP', 'VAL_OVERLAP', 'TEST_OVERLAP',
                'BALL_DETECTOR_VALUE', 'HUMAN_DETECTOR_VALUE', 'TRACKING_VALUE',
                'CALIBRATION_VALUE', 'EVENT_VALUE', 'IDENTITY_JERSEY_VALUE',
                'CURRENT_DECISION', 'NEXT_ALLOWED_ACTION', 'NOTES']

    def test_every_source_has_every_field(self, reg):
        for r in reg['sources']:
            missing = [k for k in self.REQUIRED if k not in r]
            assert not missing, f'{r.get("SOURCE_ID")} missing {missing}'

    def test_decisions_are_from_the_allowed_set(self, reg):
        allowed = set(reg['decision_values'])
        for r in reg['sources']:
            assert r['CURRENT_DECISION'] in allowed, r['SOURCE_ID']

    def test_nothing_is_left_blank(self, reg):
        """Unknown must be stated as UNKNOWN / NOT_ESTABLISHED, never empty."""
        for r in reg['sources']:
            for k in self.REQUIRED:
                assert r[k] not in (None, '', []), f'{r["SOURCE_ID"]}.{k} is blank'

    def test_all_expected_sources_present(self, by_id):
        for s in ('RB_S1', 'RB_S2', 'RB_S3', 'RB_S4', 'RB_S5', 'RB_S6',
                  'HF_KEREMBERKE', 'HF_SOCCERNET_V3', 'SOCCERTRACK_V2',
                  'SOCCERTRACK_MOT', 'HF_MARTINJOLIF', 'HF_SOCCANA',
                  'RS_TEAMTRACK', 'RS_SPORTSLABKIT'):
            assert s in by_id


class TestFrozenEyeCUIsProtected:
    def test_no_source_claims_val_or_test_overlap(self, reg):
        for r in reg['sources']:
            for k in ('VAL_OVERLAP', 'TEST_OVERLAP'):
                v = r[k]
                if isinstance(v, int):
                    assert v == 0, f'{r["SOURCE_ID"]} {k}={v}'

    def test_tracker_selection_is_recorded_as_closed(self, reg):
        assert reg['frozen_context']['tracker_selection'].startswith('CLOSED')

    def test_teamtrack_may_not_reopen_tracker_selection(self, by_id):
        assert 'do not' in by_id['RS_TEAMTRACK']['NEXT_ALLOWED_ACTION'].lower()
        assert 'reopen tracker selection' in by_id['RS_TEAMTRACK']['NEXT_ALLOWED_ACTION']


class TestInstructedDecisionsAreHonoured:
    def test_martinjolif_is_skipped_and_not_downloaded(self, by_id):
        r = by_id['HF_MARTINJOLIF']
        assert r['CURRENT_DECISION'] == 'SKIP_DUPLICATE'
        assert r['LOCAL_PATH'] == 'NOT DOWNLOADED'
        assert 'SKIP_DUPLICATE_SOURCE' in r['NOTES']
        assert not (EXT / 'huggingface' / 'martinjolif').exists()

    def test_soccana_is_deferred_and_not_downloaded(self, by_id):
        r = by_id['HF_SOCCANA']
        assert r['CURRENT_DECISION'] == 'DEFER'
        assert r['LOCAL_PATH'] == 'NOT DOWNLOADED'
        assert 'DEFER_REQUIRES_GK_RELABEL_AND_DEDUP' in r['NOTES']

    def test_soccertrack_mot_acquisition_is_closed(self, by_id):
        r = by_id['SOCCERTRACK_MOT']
        assert r['DOWNLOAD_STATUS'] == 'DISTRIBUTION_UNAVAILABLE_OR_EMPTY'
        assert 'NOT_REQUIRED_FOR_CURRENT_EYECU_DETECTOR_WORK' in r['NEXT_ALLOWED_ACTION']

    def test_sportslabkit_is_reference_only_on_its_licence(self, by_id):
        r = by_id['RS_SPORTSLABKIT']
        assert r['LICENSE'] == 'GPL-3.0'
        assert 'RESEARCH_REFERENCE_ONLY' in r['NEXT_ALLOWED_ACTION']
        assert 'do NOT copy or vendor' in r['NEXT_ALLOWED_ACTION']

    def test_blocked_access_is_recorded_not_worked_around(self, by_id):
        assert 'BLOCKED_ACCESS' in by_id['RS_TEAMTRACK']['DOWNLOAD_STATUS']


class TestConsolidationWasVerified:
    def test_roboflow_zip_hashes_survived_the_move(self):
        m = load(EXT / 'roboflow_audit' / 'manifests' / 'raw_zip_hashes.json')
        assert m['all_hashes_unchanged'] is True
        assert len(m['zips']) == 6

    def test_every_move_was_hash_verified(self):
        log = load(EXT / 'consolidation_log.json')
        assert log['errors'] == []
        assert log['verified'] == len([m for m in log['moved'] if not m.get('dry_run')])
        assert log['verified'] > 300

    def test_duplicate_copies_were_removed_with_their_twin_named(self):
        log = load(EXT / 'consolidation_log.json')
        assert log['deduplicated'], 'the RAW archive shipped twice; one copy should go'
        for d in log['deduplicated']:
            assert d['identical_to'] and d['sha256']
            assert (EXT / d['identical_to']).exists(), 'removed a copy whose twin is gone'
        assert not log['remaining_duplicate_groups']

    def test_canonical_tree_exists(self):
        for d in ('roboflow_audit/raw_zips', 'roboflow_audit/manifests',
                  'huggingface/manifests', 'huggingface/download_logs',
                  'huggingface/keremberke_football_object_detection/raw',
                  'huggingface/soccernet_v3/metadata_only',
                  'soccertrack_v2/gsr', 'soccertrack_v2/bas', 'soccertrack_v2/raw',
                  'soccertrack_v2/videos', 'soccertrack_v2/public_repo',
                  'research_sources/teamtrack', 'research_sources/sportslabkit'):
            assert (EXT / d).exists(), d


class TestDownloadRecords:
    def test_public_downloads_recorded_revision_and_hashes(self):
        for f in ('keremberke__football-object-detection.json',
                  'Voxel51__SoccerNet-V3.json'):
            m = load(EXT / 'huggingface' / 'download_logs' / f)
            assert len(m['revision']) == 40
            assert m['all_sizes_match_api'] is True
            assert m['any_html_error_pages'] is False
            assert all(len(r['sha256']) == 64 for r in m['files'])

    def test_no_credential_was_stored(self):
        m = load(EXT / 'huggingface' / 'download_logs' /
                 'keremberke__football-object-detection.json')
        assert 'no token read or stored' in m['credentials_used']
        assert not (Path.home() / '.cache' / 'huggingface' / 'token').exists()

    def test_no_secret_material_in_the_workspace(self):
        bad = []
        for p in EXT.rglob('*.json'):
            if 'public_repo' in p.parts or p.stat().st_size > 40_000_000:
                continue
            t = p.read_text(encoding='utf-8', errors='replace')
            for tok in ('hf_', 'Authorization:', 'Bearer ', 'X-Amz-Signature'):
                if tok in t:
                    bad.append(f'{p.name}:{tok}')
        assert not bad, bad


class TestBulkIsNotCommittable:
    def test_gitignore_excludes_dataset_bulk(self):
        gi = (REPO / '.gitignore').read_text(encoding='utf-8')
        assert 'EyeCU_external_data/*' in gi
        assert '!EyeCU_external_data/MASTER_EXTERNAL_SOURCE_REGISTRY.json' in gi

    def test_registry_and_manifests_are_allowed_through(self):
        gi = (REPO / '.gitignore').read_text(encoding='utf-8')
        for keep in ('!EyeCU_external_data/roboflow_audit/manifests/',
                     '!EyeCU_external_data/huggingface/download_logs/',
                     '!EyeCU_external_data/soccertrack_v2/manifests/'):
            assert keep in gi


class TestPriorityMatrixAndGate:
    @pytest.fixture(scope='class')
    def pm(self):
        p = XS / 'reports' / 'priority_matrix.json'
        if not p.exists():
            pytest.skip('priority matrix not built')
        return load(p)

    def test_rankings_are_not_collapsed_into_one(self, pm):
        assert len(pm['rankings']) == 5
        for k in ('1_CURRENT_BALL_IMPROVEMENT', '2_FUTURE_CALIBRATION',
                  '3_FUTURE_EVENTS', '4_FUTURE_IDENTITY_JERSEY',
                  '5_FUTURE_TRACKING_RESEARCH'):
            assert k in pm['rankings']

    def test_every_rating_is_from_the_scale(self, pm):
        for src, row in pm['priority_matrix'].items():
            for k, v in row.items():
                assert v in pm['rating_scale'], f'{src}.{k}={v}'

    def test_experiment_d_is_gated_not_executed(self, pm):
        g = pm['experiment_d_gate']
        assert g['not_executed'] is True
        assert g['answer'] in ('STRONG YES', 'YES', 'WEAK', 'NO')
        assert g['blocking_conditions_before_D_can_be_designed']

    def test_gk_and_referee_may_not_be_collapsed(self, pm):
        never = pm['experiment_d_gate']['design_constraints_for_a_future_D']['never']
        assert any('goalkeeper or referee collapsed' in n for n in never)
        assert any('VAL or TEST' in n for n in never)
