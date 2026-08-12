"""
The consolidated external-source workspace, as assertions.

The registry is the thing future decisions will be made from, so the checks here
are mostly about it not drifting from the audits it was built out of, and about
the rules that protect EyeCU's frozen artifacts holding: no external image in
VAL or TEST, no dataset bulk in git, no credential on disk, and no source
promoted to KEEP_ACTIVE without the evidence that justifies it.
"""

import json
import shutil
import subprocess
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
    """Asserts the OUTCOME, not one particular .gitignore spelling.

    These originally pinned exact ignore rules, which broke the moment the file
    was edited while the thing that matters -- no dataset bulk in the repo -- was
    still true. What is checked now is git's actual index.
    """

    def _tracked(self):
        import subprocess
        out = subprocess.run(['git', 'ls-files'], cwd=str(REPO),
                             capture_output=True, text=True).stdout
        return [l for l in out.splitlines() if l.strip()]

    def test_no_external_dataset_bulk_is_tracked(self):
        """Scoped to the external-source workspace.

        A first version swept the whole repo and failed on artifacts that were
        committed deliberately long ago -- the hand-corrected annotation ZIPs,
        the raw CVAT exports, the evidence renders. Those are the project's
        irreplaceable human work, not bulk, and a test that condemns them is
        wrong about the repo rather than right about a regression.
        """
        bad = []
        for f in self._tracked():
            if not f.startswith(('EyeCU_external_data/',
                                 'experiments/external_sources/')):
                continue
            p = REPO / f
            if not p.exists():
                continue
            if p.suffix.lower() in ('.mp4', '.zip', '.npy', '.npz', '.pt'):
                bad.append(f)
            elif p.suffix.lower() in ('.png', '.jpg') and                     'contact_sheets' not in f and 'reference_sheets' not in f:
                bad.append(f)
        assert not bad, f'external dataset bulk is tracked: {bad[:5]}'

    def test_regenerable_review_artifacts_are_not_tracked(self):
        tracked = set(self._tracked())
        for f in ('experiments/external_sources/keremberke_review/ledger.json',
                  'experiments/external_sources/keremberke_review/review_queue.json'):
            assert f not in tracked, f'{f} regenerates and must not be committed'
        assert not any(f.startswith('experiments/external_sources/keremberke_review/'
                                    'working_copy/') for f in tracked)

    def test_the_irreplaceable_human_work_IS_tracked(self):
        assert ('experiments/external_sources/keremberke_review/decisions.json'
                in set(self._tracked()))

    def test_no_oversized_file_in_the_review_package(self):
        big = [(f, (REPO / f).stat().st_size) for f in self._tracked()
               if f.startswith('experiments/external_sources/keremberke_review/')
               and (REPO / f).exists()
               and (REPO / f).stat().st_size > 4_000_000]
        assert not big, f'oversized review-package files: {big}'


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


class TestKeremberkeClassRepair:
    """The class-repair feasibility audit, including the two claims I got wrong first."""

    @pytest.fixture(scope='class')
    def cr(self):
        p = XS / 'reports' / 'KEREMBERKE_CLASS_REPAIR.json'
        if not p.exists():
            pytest.skip('class-repair audit not present')
        return load(p)

    def test_ignore_regions_are_not_natively_supported(self, cr):
        ig = cr['ignore_region_support']
        assert ig['verdict'] == 'NOT_SUPPORTED'
        assert ig['corollary'] == 'CUSTOM_TRAINING_REQUIRED to achieve it'
        assert ig['not_implemented_in_this_task'] is True
        # every one of the four conditions must be answered, not skipped
        c = ig['against_the_four_conditions']
        assert c['prevents_unlabelled_humans_contributing_background_loss'] is False
        assert c['does_not_require_custom_loss_or_trainer'] is False
        assert c['already_compatible_with_our_pipeline'] is False

    def test_ignore_verdict_cites_real_code(self, cr):
        ev = cr['ignore_region_support']['evidence']
        files = {e['file'].split(':')[0] for e in ev}
        assert 'ultralytics/data/utils.py' in files
        assert 'ultralytics/utils/loss.py' in files
        assert any('labels require 5 columns' in e['code'] for e in ev)
        assert any('bce_loss.sum()' in e['code'] for e in ev)

    def test_the_five_column_assertion_still_exists_in_the_installed_package(self):
        """The verdict rests on this line; if ultralytics changes it, retest."""
        p = (REPO / 'eye_env' / 'Lib' / 'site-packages' / 'ultralytics'
             / 'data' / 'utils.py')
        if not p.exists():
            pytest.skip('ultralytics not installed here')
        assert 'labels require 5 columns' in p.read_text(encoding='utf-8')

    def test_temporal_propagation_was_corrected_not_quietly_dropped(self, cr):
        t = cr['temporal_grouping']
        assert t['verdict'].startswith('IDENTITY PROPAGATION IS NOT VIABLE')
        assert 'was wrong' in t['correction']
        assert '0.52' in t['measurement']
        assert t['no_persistent_ids_invented'] is True

    def test_box_counts_are_internally_consistent(self, cr):
        b = cr['box_counts']
        assert (b['LIKELY_PLAYER'] + b['AMBIGUOUS'] + b['POSSIBLE_REFEREE']
                + b['POSSIBLE_GOALKEEPER']) == b['total_human_boxes'] == 21615
        assert (b['AMBIGUOUS'] + b['POSSIBLE_REFEREE'] + b['POSSIBLE_GOALKEEPER']
                ) == b['candidates_needing_a_human_decision']
        assert b['no_box_geometry_is_redrawn'] is True

    def test_detector_is_not_annotation_authority(self, cr):
        assert cr['method']['detector_is_annotation_authority'] is False
        assert 'recall_caveat' in cr['method']
        assert cr['constraints_respected']['detector_predictions_became_gt'] is False

    def test_clean_subset_reports_its_cost_not_just_its_size(self, cr):
        s = cr['clean_subset_option']['PERMISSIVE_no_gk_or_referee_candidate']
        assert s['images'] == 267 and s['ball_instances'] == 217
        assert '17%' in s['verdict']
        assert 'NOT free of officials' in cr['clean_subset_option']['honesty_caveat']

    def test_recommendation_is_A_and_keeps_the_ball_data(self, cr):
        r = cr['recommendation']
        assert r['choice'].startswith('A')
        A = cr['option_comparison']['A_RECLASSIFY_EXISTING_GK_REF_BOXES']
        B = cr['option_comparison']['B_HUMAN_VERIFIED_CLEAN_SUBSET']
        assert A['retained_ball_instances'] == 1263
        assert A['retained_ball_instances'] > B['retained_ball_instances'] * 5
        assert A['experiment_d_stays_data_only'] is True
        assert cr['option_comparison']['C_IGNORE_REGION_TRAINING'][
            'experiment_d_stays_data_only'] is False

    def test_keremberke_is_not_yet_ready(self, cr):
        assert cr['is_keremberke_ready_for_experiment_d']['answer'] == 'NO'
        assert cr['constraints_respected']['experiment_d_started'] is False
        assert cr['constraints_respected']['keremberke_original_modified'] is False

    def test_original_labels_are_untouched(self):
        """The audit must not have rewritten a single class id."""
        import json as _j
        base = (REPO / 'EyeCU_external_data/huggingface'
                / 'keremberke_football_object_detection/extracted')
        if not base.exists():
            pytest.skip('keremberke not extracted')
        for split in ('train', 'valid', 'test'):
            aj = list((base / split).rglob('_annotations.coco.json'))
            if not aj:
                continue
            a = _j.loads(aj[0].read_text(encoding='utf-8'))
            names = {c['name'] for c in a['categories']}
            assert names <= {'player', 'football', 'football-players'}, names
            assert 'goalkeeper' not in names and 'referee' not in names, (
                'the original export must still be two-class')


class TestKeremberkeReviewPackage:
    """The review package must be ready for a human and must not pre-empt one."""

    PKG = XS / 'keremberke_review'

    @pytest.fixture(scope='class')
    def man(self):
        p = self.PKG / 'PACKAGE_MANIFEST.json'
        if not p.exists():
            pytest.skip('review package not built')
        return load(p)

    def test_no_human_decision_has_been_fabricated(self, man):
        """Originally asserted zero decisions; the human has since reviewed.

        The claim worth protecting is not "no decisions exist" -- that expired
        the moment the reviewer started -- but that every decision on record was
        made by the human, with no tool-authored rows slipped in.
        """
        assert man['no_proposal_is_ground_truth'] is True
        d = self.PKG / 'decisions.json'
        if not d.exists() or not d.read_text(encoding='utf-8').strip():
            assert man['human_decisions_recorded'] == 0
            return
        authors, modes = set(), set()
        for line in d.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                authors.add(r.get('author'))
                modes.add(r.get('mode', 'candidates'))
        assert authors == {'human reviewer'}, authors
        assert modes <= {'candidates', 'qa_player', 'qa_nocand', 'u_resolution',
                         'missed_role', 'missed_role_manual',
                         'final_target'}, modes

    def test_ledger_starts_with_every_final_class_empty(self):
        p = self.PKG / 'ledger.json'
        if not p.exists():
            pytest.skip('ledger regenerated on demand')
        led = load(p)
        assert len(led) == 22878
        assert all(r['HUMAN_FINAL_CLASS'] is None for r in led)
        # the two must be kept distinct forever
        for r in led:
            assert 'PROPOSED_CLASS' in r and 'ORIGINAL_CLASS' in r

    def test_original_export_is_immutable_and_hashed(self, man):
        assert man['original_source_immutable'] is True
        assert man['geometry_may_change'] is False
        assert man['only_class_ids_may_change'] is True
        import hashlib
        base = (REPO / 'EyeCU_external_data/huggingface'
                / 'keremberke_football_object_detection/extracted')
        if not base.exists():
            pytest.skip('keremberke not extracted')
        for split, expect in man['original_annotation_sha256'].items():
            aj = list((base / split).rglob('_annotations.coco.json'))
            h = hashlib.sha256(aj[0].read_bytes()).hexdigest()
            assert h == expect, f'{split} original annotation changed'

    def test_queue_is_ordered_by_run_then_priority(self):
        p = self.PKG / 'review_queue.json'
        if not p.exists():
            pytest.skip('queue regenerated on demand')
        q = load(p)
        assert len(q) == 1170
        assert sum(len(x['candidate_box_ids']) for x in q) == 4153
        runs = [x['run'] for x in q]
        # each run appears as one contiguous block
        seen, blocks = set(), 0
        prev = None
        for r in runs:
            if r != prev:
                blocks += 1
                assert r not in seen, f'run {r} is not contiguous in the queue'
                seen.add(r)
                prev = r
        assert blocks == len(seen)

    def test_two_independent_qa_samples_exist(self):
        qp = load(self.PKG / 'qa_likely_player.json')
        qn = load(self.PKG / 'qa_no_candidate_images.json')
        assert qp['sample_size'] == 250 and qp['answers_recorded'] == 0
        # the stratum tuple is size x confidence-band x region x depth, so 3*3*3*2
        # = 54 is the CEILING, not a shortfall. Run is tracked separately.
        assert qp['strata_covered'] == 54, 'stratification collapsed'
        assert {r['run'] for r in qp['rows']} == {'plain_A', 'plain_B', 'pp_A', 'pp_B'}
        for f in ('size', 'detector_conf_band', 'region', 'depth'):
            vals = {r['stratum'][f] for r in qp['rows']}
            assert len(vals) >= 2, f'{f} did not vary in the sample'
        assert qn['answers_recorded'] == 0
        assert qn['kept_separate_from_candidate_precision'] is True
        assert set(qp['rows'][0]['allowed']) == {
            'TRUE_PLAYER', 'MISSED_GOALKEEPER', 'MISSED_REFEREE', 'UNCERTAIN'}

    def test_no_candidate_qa_is_a_census(self):
        qn = load(self.PKG / 'qa_no_candidate_images.json')
        assert qn['sample_size'] == qn['population'], (
            'only 57 images have no candidate, so all of them are reviewed')

    def test_reference_sheet_per_run(self):
        d = self.PKG / 'reference_sheets'
        assert {p.stem for p in d.glob('*_kits.jpg')} == {
            'plain_A_kits', 'plain_B_kits', 'pp_A_kits', 'pp_B_kits'}

    def test_gate_blocks_while_review_is_incomplete(self):
        """The first pass is now done, so this asserts the gate still BLOCKS.

        It originally asserted candidates_reviewed == 0, which expired the moment
        the human started. What must stay true is that the gate refuses while the
        second pass is outstanding.
        """
        p = self.PKG / 'REVIEW_STATUS.json'
        if not p.exists():
            pytest.skip('gate not yet run')
        s = load(p)
        failing = [g for g in s['gate'] if g['result'] == 'FAIL']
        assert failing, 'the gate must not pass while the second pass is outstanding'
        g2 = load(self.PKG / 'SECOND_PASS_GATE.json')
        assert g2['passed'] is False and g2['apply_permitted'] is False

    def test_ball_counts_are_reported_on_the_frozen_convention(self):
        p = self.PKG / 'REVIEW_STATUS.json'
        if not p.exists():
            pytest.skip('gate not yet run')
        b = load(p)['ball_counts_preserved']
        assert b['instances'] == 1263
        assert (b['le5'], b['le8'], b['le12']) == (90, 474, 969)
        assert 'stored pixels' in b['convention']

    def test_decisions_file_is_tracked_but_bulk_is_not(self):
        """Checked against git's index rather than a .gitignore spelling."""
        import subprocess
        tracked = set(subprocess.run(['git', 'ls-files'], cwd=str(REPO),
                                     capture_output=True, text=True).stdout.split())
        assert ('experiments/external_sources/keremberke_review/decisions.json'
                in tracked)
        assert ('experiments/external_sources/keremberke_review/ledger.json'
                not in tracked)


class TestKeremberkeSecondPass:
    """The coverage failure the first gate would have missed."""

    PKG = XS / 'keremberke_review'

    @pytest.fixture(scope='class')
    def sp(self):
        p = XS / 'reports' / 'KEREMBERKE_SECOND_PASS.json'
        if not p.exists():
            pytest.skip('second pass not present')
        return load(p)

    def test_first_pass_really_is_complete(self, sp):
        c = sp['actual_review_completion']
        assert c['candidates']['complete'] and c['qa_player']['complete']
        assert c['qa_nocand']['complete']
        assert c['verified_from'].startswith('decisions.json')
        # log lines must not be reported as decisions
        assert c['log_lines'] > c['unique_decisions']

    def test_completeness_is_not_treated_as_sufficiency(self, sp):
        assert sp['verdicts']['MISSED_ROLE_COVERAGE'] == 'FAIL'
        assert sp['second_pass_gate']['result'] == 'FAIL'
        assert sp['second_pass_gate']['apply_permitted'] is False

    def test_recall_failure_is_quantified_with_its_uncertainty(self, sp):
        q = sp['qa_player_results']
        assert q['missed_role_rate'] == 0.064
        assert q['MISSED_GOALKEEPER'] == 8 and q['MISSED_REFEREE'] == 8
        assert q['reported_separately_not_combined'] is True
        e = q['extrapolation']
        assert e['ci95'][0] < e['point_estimate'] < e['ci95'][1]
        assert 'small basis' in e['caveat']

    def test_nocand_limitation_is_stated(self, sp):
        assert 'cannot close the coverage gap' in \
            sp['qa_nocand_results']['limitation_confirmed']

    def test_retro_queue_does_not_overclaim(self, sp):
        r = sp['retrospective_missed_role_queue']
        assert r['boxes'] == 6984
        assert r['no_role_assigned_automatically'] is True
        assert r['no_identity_propagated'] is True
        assert '16/16' in r['validation']['honest_caveat']
        assert 'NOT a small bounded queue' in r['honest_assessment']

    def test_detector_contribution_is_stated_as_nil(self, sp):
        r = sp['retrospective_missed_role_queue']
        assert 'constant across every candidate' in \
            r['the_detector_contributes_nothing_here']

    def test_unestablished_scopes_are_not_invented(self, sp):
        for k in ('bad_geometry_scope', 'non_target_human_scope'):
            assert sp[k]['status'] == 'NOT_ESTABLISHED'
            assert sp[k].get('count') is None
        assert sp['u_resolution']['counts_per_category'].startswith('NOT_ESTABLISHED')

    def test_non_target_policy_protects_real_targets(self, sp):
        pol = sp['non_target_human_scope']['policy_already_fixed']
        assert any('NEVER become player' in p for p in pol)
        assert any('NEVER be removed and left as unlabelled background' in p
                   for p in pol)

    def test_run_totals_reconcile_with_the_audit(self, sp):
        r = sp['run_level_analysis']
        runs = [r[k] for k in ('plain_A', 'plain_B', 'pp_A', 'pp_B')]
        assert sum(x['le8'] for x in runs) + 0 == 474
        assert sum(x['le5'] for x in runs) == 90
        assert r['totals_reconcile'] == {'balls': 1263, 'le5': 90, 'le8': 474,
                                         'le12': 969}

    def test_high_wide_answer_is_both_not_one_sided(self, sp):
        w = sp['rare_wide_angle']
        assert w['answer_to_the_explicit_question'].startswith('BOTH')
        assert w['share_of_le8_pct'] == 83.8
        assert w['recommended_policy'].startswith('KEEP_ALL')
        assert 'why_not_exclude' in w and 'why_not_keep_blindly' in w

    def test_ball_gt_untouched(self, sp):
        b = sp['ball_counts_preserved']
        assert b['pre_repair'] == b['current']
        assert b['current'] == {'total': 1263, 'le5': 90, 'le8': 474, 'le12': 969}

    def test_nothing_was_applied_or_guessed(self, sp):
        c = sp['constraints_respected']
        for k in ('training_performed', 'merged_into_eyecu_train',
                  'experiment_d_started', 'original_dataset_modified',
                  'apply_run', 'model_prediction_promoted_to_gt',
                  'unresolved_classes_guessed', 'annotations_silently_deleted',
                  'geometry_silently_repaired', 'test_performance_accessed'):
            assert c[k] is False, k
        assert sp['verdicts']['READY_TO_DESIGN_EXPERIMENT_D'] == 'NO'

    def test_gate_file_blocks_apply(self):
        g = load(self.PKG / 'SECOND_PASS_GATE.json')
        assert g['passed'] is False
        assert g['apply_permitted'] is False
        # D was blocking before the U pass ran; F is blocking until the
        # retrospective sweep is done. Only F is guaranteed at every stage.
        assert 'F MISSED_ROLE_REVIEW complete' in set(g['blocking'])

    def test_second_pass_queues_exist_and_are_unanswered(self):
        u = load(self.PKG / 'u_resolution_queue.json')
        m = load(self.PKG / 'missed_role_queue.json')
        assert u['count'] == 48 and u['answers_recorded'] == 0
        # 6,984 originally; 300 already-answered boxes were removed and recorded
        # in `prefilled`, so the live queue is 6,684
        assert m['queue_boxes'] == 6684 and m['answers_recorded'] == 0
        assert m['deduplication']['original_queue_boxes'] == 6984
        assert all(r['U_RESOLUTION_CATEGORY'] is None for r in u['rows'])


class TestSecondPassProgressReachesTheGate:
    """The defect that would have silently discarded the whole second pass.

    The gate read U_RESOLUTION_CATEGORY and HUMAN_ANSWER out of the queue JSON
    files; the review server only ever appends to decisions.json and never edits
    a queue. Conditions D, E and F therefore read 0/48 and 0/6,984 no matter how
    much reviewing was done. Proven by simulation before the fix, and pinned here.
    """

    PKG = XS / 'keremberke_review'

    def test_gate_reads_decisions_not_the_queue_files(self):
        src = (REPO / 'tools' / 'kb_second_pass_gate.py').read_text(encoding='utf-8')
        assert "m == 'u_resolution'" in src and "m == 'missed_role'" in src
        assert 'DEFECT FIXED HERE' in src

    def test_gate_registers_simulated_second_pass_work(self, tmp_path):
        """End to end: feed decisions the way the server writes them."""
        import shutil
        import subprocess
        import sys as _s
        if not (self.PKG / 'missed_role_queue.json').exists():
            pytest.skip('queues absent')
        bak = self.PKG.with_name('kr__pytest_bak')
        if bak.exists():
            shutil.rmtree(bak)
        shutil.copytree(self.PKG, bak)
        try:
            u = load(self.PKG / 'u_resolution_queue.json')
            m = load(self.PKG / 'missed_role_queue.json')
            with open(self.PKG / 'decisions.json', 'a', encoding='utf-8') as f:
                for r in u['rows']:
                    f.write(json.dumps({'mode': 'u_resolution', 'BOX_ID': r['BOX_ID'],
                                        'IMAGE': r['IMAGE'],
                                        'HUMAN_FINAL_CLASS': 'FALSE_POSITIVE',
                                        'author': 'human reviewer'}) + '\n')
                for r in m['rows'][:50]:
                    f.write(json.dumps({'mode': 'missed_role', 'BOX_ID': r['BOX_ID'],
                                        'IMAGE': r['IMAGE'],
                                        'HUMAN_FINAL_CLASS': 'player',
                                        'author': 'human reviewer'}) + '\n')
            out = subprocess.run([_s.executable, 'tools/kb_second_pass_gate.py'],
                                 capture_output=True, text=True, encoding='utf-8',
                                 cwd=str(REPO))
            assert 'D all original U categorized' in out.stdout
            dline = [l for l in out.stdout.splitlines()
                     if l.startswith('D all original U')][0]
            fline = [l for l in out.stdout.splitlines()
                     if l.startswith('F MISSED_ROLE')][0]
            assert '48/48' in dline, dline
            # >= because the reviewer may already have real decisions on record
            n = int(fline.split()[-1].split('/')[0])
            assert n >= 50, fline
        finally:
            shutil.rmtree(self.PKG)
            bak.rename(self.PKG)

    def test_prior_decisions_survive_that_round_trip(self):
        """Modes grow as passes complete; none may ever disappear."""
        d = self.PKG / 'decisions.json'
        modes = {json.loads(l).get('mode') for l in
                 d.read_text(encoding='utf-8').splitlines() if l.strip()}
        assert {'candidates', 'qa_player', 'qa_nocand'} <= modes, modes
        assert modes <= {'candidates', 'qa_player', 'qa_nocand', 'u_resolution',
                         'missed_role', 'missed_role_manual',
                         'final_target'}, modes

    def test_condition_G_requires_resolution_not_absence_of_history(self):
        src = (REPO / 'tools' / 'kb_second_pass_gate.py').read_text(encoding='utf-8')
        assert 'mr_unresolved' in src and 'qa_unresolved' in src
        assert 'historical fact be false' in src

    def test_applier_folds_every_role_bearing_mode(self):
        """qa_player found 16 officials; they must not be dropped at write time."""
        src = (REPO / 'tools' / 'kb_apply_review.py').read_text(encoding='utf-8')
        assert "ROLE_MODES = ('candidates', 'qa_player', 'qa_nocand', 'missed_role'," in src
        assert 'silently drop the 16 officials' in src


class TestImageCentricReviewServer:
    def test_server_only_writes_its_own_modes(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        import re as _re
        m = _re.search(r"MODES = \(([^)]*)\)", src)
        modes = {x.strip().strip("'") for x in m.group(1).split(',') if x.strip()}
        assert modes == {'missed_role', 'missed_role_manual',
                         'missing_target_box', 'missing_target_retraction'}
        # the modes that settle other passes stay out of reach of this server
        assert not modes & {'candidates', 'qa_player', 'qa_nocand',
                            'u_resolution', 'final_target',
                            'missing_target_resolution'}
        assert "d.get('mode') not in MODES" in src

    def test_it_appends_and_never_rewrites(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert "open(PKG / 'decisions.json', 'a'" in src
        assert 'f.flush()' in src, 'autosave must hit disk immediately'

    def test_bulk_actions_are_scoped_to_the_current_image(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        # both bulk paths iterate cur().candidates only
        assert 'ALL CURRENT CANDIDATES = PLAYER, this image only' in src
        assert 'accept-all after every candidate was displayed' in src
        assert "getElementById('acc').disabled=!all" in src

    def test_it_resumes_only_its_own_modes(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert "dec = {b: v for (m, b), v in per_mode.items() if m == 'missed_role'}" in src
        assert "manual = {b: v for (m, b), v in per_mode.items() "                "if m == 'missed_role_manual'}" in src

    def test_ordering_and_context_are_preserved(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert 'key=lambda k: -order[k]' in src, 'highest-score-first must survive'
        assert "'context':" in src, 'surrounding context boxes must be shown'


class TestDecisionPrecedence:
    """One documented rule, one implementation, verified through the real applier."""

    PKG = XS / 'keremberke_review'

    def test_both_consumers_use_the_shared_resolver(self):
        ap = (REPO / 'tools' / 'kb_apply_review.py').read_text(encoding='utf-8')
        assert 'kb_decisions.resolve' in ap
        assert 'kb_decisions.by_mode' in ap, 'QA completion is a per-mode question'

    def test_rule_is_time_then_line_and_mode_carries_no_rank(self):
        src = (REPO / 'tools' / 'kb_decisions.py').read_text(encoding='utf-8')
        assert 'mode is NOT a rank' in src
        assert 'later recorded_utc wins' in src

    def test_later_decision_wins_regardless_of_mode_or_file_position(self, tmp_path):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        p = tmp_path / 'd.json'
        rows = [
            # later timestamp appears FIRST in the file: the clock must still win
            {'mode': 'missed_role', 'BOX_ID': 'x:1', 'HUMAN_FINAL_CLASS': 'goalkeeper',
             'recorded_utc': '2026-08-12T20:00:00Z'},
            {'mode': 'qa_nocand', 'BOX_ID': 'x:1', 'HUMAN_FINAL_CLASS': 'player',
             'recorded_utc': '2026-08-12T09:00:00Z'},
            # identical timestamps: later line wins
            {'mode': 'qa_player', 'BOX_ID': 'x:2', 'HUMAN_FINAL_CLASS': 'player',
             'recorded_utc': '2026-08-12T12:00:00Z'},
            {'mode': 'missed_role', 'BOX_ID': 'x:2', 'HUMAN_FINAL_CLASS': 'referee',
             'recorded_utc': '2026-08-12T12:00:00Z'},
            # a later 'uncertain' clears an earlier role
            {'mode': 'candidates', 'BOX_ID': 'x:3', 'HUMAN_FINAL_CLASS': 'referee',
             'recorded_utc': '2026-08-12T10:00:00Z'},
            {'mode': 'missed_role', 'BOX_ID': 'x:3', 'HUMAN_FINAL_CLASS': 'uncertain',
             'recorded_utc': '2026-08-12T11:00:00Z'},
            # a disposition is not a class
            {'mode': 'u_resolution', 'BOX_ID': 'x:4',
             'HUMAN_FINAL_CLASS': 'FALSE_POSITIVE',
             'recorded_utc': '2026-08-12T10:00:00Z'},
        ]
        p.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
        r = kb_decisions.resolve(p)
        assert r['x:1']['final_class'] == 'goalkeeper'
        assert r['x:1']['decided_in_mode'] == 'missed_role'
        assert r['x:2']['final_class'] == 'referee'
        assert r['x:3']['final_class'] is None
        assert r['x:3']['disposition'] == 'UNRESOLVED'
        assert r['x:4']['final_class'] is None
        assert r['x:4']['disposition'] == 'FALSE_POSITIVE'
        assert r['x:4']['action'] == 'REMOVE_ANNOTATION'
        # every superseded answer is retained, never dropped
        assert r['x:1']['superseded'] and r['x:1']['decisions_recorded'] == 2

    def test_by_mode_keeps_qa_answers_visible_after_a_later_override(self, tmp_path):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        p = tmp_path / 'd.json'
        p.write_text('\n'.join(json.dumps(r) for r in [
            {'mode': 'qa_player', 'BOX_ID': 'x:9', 'HUMAN_FINAL_CLASS': 'goalkeeper',
             'recorded_utc': '2026-08-12T10:00:00Z'},
            {'mode': 'missed_role', 'BOX_ID': 'x:9', 'HUMAN_FINAL_CLASS': 'player',
             'recorded_utc': '2026-08-12T11:00:00Z'}]), encoding='utf-8')
        assert kb_decisions.resolve(p)['x:9']['final_class'] == 'player'
        assert kb_decisions.by_mode(p)[('qa_player', 'x:9')] == 'goalkeeper'


class TestMissedRoleQueueIsDeduplicated:
    PKG = XS / 'keremberke_review'

    @pytest.fixture(scope='class')
    def q(self):
        p = self.PKG / 'missed_role_queue.json'
        if not p.exists():
            pytest.skip('queue absent')
        return load(p)

    def test_no_box_settled_ELSEWHERE_is_requeued(self, q):
        """Boxes the reviewer answers WITHIN this pass are expected to be settled.

        What must not happen is a box already answered in another mode being put
        back in front of the reviewer -- that is the redundant work the dedup
        removed, and re-asking a settled question invites a contradictory answer.
        """
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        per = kb_decisions.by_mode(self.PKG / 'decisions.json')
        elsewhere = {b for (m, b), v in per.items()
                     if m not in ('missed_role', 'missed_role_manual')
                     and v in ('player', 'goalkeeper', 'referee')}
        bad = [r['BOX_ID'] for r in q['rows'] if r['BOX_ID'] in elsewhere]
        assert bad == [], f'{len(bad)} boxes answered elsewhere were re-queued'

    def test_unresolved_boxes_are_kept(self, q):
        assert q['deduplication']['kept_because_still_unresolved'] >= 1

    def test_removals_are_recorded_not_silent(self, q):
        d = q['deduplication']
        assert d['nothing_deleted_silently'] is True
        assert len(q['prefilled']) == d['removed_already_answered']
        assert (d['original_queue_boxes']
                == q['queue_boxes'] + d['removed_already_answered'])
        for p in q['prefilled']:
            assert p['already_answered'] and p['answered_in_mode']

    def test_qa_nocand_officials_are_box_level_and_modifiable(self):
        """An image-level flag would be useless to the applier."""
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        led = {r['BOX_ID']: r for r in load(self.PKG / 'ledger.json')}
        per = kb_decisions.by_mode(self.PKG / 'decisions.json')
        offs = [b for (m, b), v in per.items()
                if m == 'qa_nocand' and v in ('goalkeeper', 'referee')]
        assert len(offs) == 25
        for b in offs:
            assert b in led, f'{b} has no ledger row to modify'
            assert led[b]['bbox_xywh'] and len(led[b]['bbox_xywh']) == 4

    def test_large_responses_are_not_truncated(self):
        """A 1.8 MB state payload lost its tail over HTTP/1.0 on Windows.

        It succeeded intermittently, which is worse than failing outright: the
        browser would silently receive a short body and fail to parse it.
        """
        for f in ('kb_review_server.py', 'kb_review_server2.py'):
            src = (REPO / 'tools' / f).read_text(encoding='utf-8')
            assert "protocol_version = 'HTTP/1.1'" in src, f
            assert 'self.wfile.flush()' in src, f
            assert 'memoryview(body)' in src, f


class TestUResolutionIsNotPrefilledByTheFirstPass:
    """48/48 with zero resolutions: the first-pass U was counted as an answer.

    An original `U` is the QUESTION this pass exists to answer. Counting it as
    progress showed a finished pass and would have let the whole review be
    skipped.
    """

    PKG = XS / 'keremberke_review'

    def test_dedicated_server_counts_only_its_own_mode(self):
        src = (REPO / 'tools' / 'kb_u_resolution_server.py').read_text(encoding='utf-8')
        assert "if d.get('mode') == MODE:" in src
        assert "MODE = 'u_resolution'" in src
        assert 'deliberately ignored' in src

    def test_it_starts_at_zero_not_forty_eight(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        if not (self.PKG / 'u_resolution_queue.json').exists():
            pytest.skip('queue absent')
        import importlib
        m = importlib.import_module('kb_u_resolution_server')
        st = m.build_state()
        assert st['total_boxes'] == 48 and st['total_images'] == 42
        # The 48 first-pass 'uncertain' answers must never appear as progress.
        # Once the human has actually done the pass this reads 48; what must
        # never happen is it reading 48 with zero u_resolution rows on record.
        import kb_decisions
        real = sum(1 for (mode, _), _ in
                   kb_decisions.by_mode(self.PKG / 'decisions.json').items()
                   if mode == 'u_resolution')
        assert len(st['decisions']) == real

    def test_all_six_categories_plus_roles_are_offered(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        m = importlib.import_module('kb_u_resolution_server')
        keys = {c[0] for c in m.CHOICES}
        vals = {c[1] for c in m.CHOICES}
        assert keys == set('PGRAONBFX')
        assert vals == {'player', 'goalkeeper', 'referee', 'AMBIGUOUS_TARGET',
                        'OCCLUDED_UNCLEAR', 'NON_TARGET_HUMAN',
                        'BALL_WRONG_HUMAN_BOX', 'FALSE_POSITIVE',
                        'PARTIAL_BODY_BAD_BOX'}
        # every option must carry a human-readable meaning shown on screen
        assert all(len(c[2]) > 10 for c in m.CHOICES)

    def test_b_means_ball_not_back(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        m = importlib.import_module('kb_u_resolution_server')
        assert dict((c[0], c[1]) for c in m.CHOICES)['B'] == 'BALL_WRONG_HUMAN_BOX'
        src = (REPO / 'tools' / 'kb_u_resolution_server.py').read_text(encoding='utf-8')
        assert "e.key==='ArrowLeft'" in src, 'previous must move off the B key'

    def test_multimode_server_no_longer_serves_second_pass_modes(self):
        src = (REPO / 'tools' / 'kb_review_server.py').read_text(encoding='utf-8')
        assert "modes['u_resolution']" not in src
        assert "modes['missed_role']" not in src
        assert "if d.get('mode', 'candidates') in served:" in src

    def test_second_pass_writes_are_vocabulary_checked(self):
        src = (REPO / 'tools' / 'kb_u_resolution_server.py').read_text(encoding='utf-8')
        assert 'value not in the U-resolution vocabulary' in src
        assert 'this server only writes u_resolution' in src


class TestFinalTargetResolution:
    """48/48 U cases REVIEWED is complete work; 3 target ROLES remain unresolved.

    The two must never be conflated. The first is a finished pass; the second is
    three boxes that would otherwise stay labelled `player` -- wrong, if any of
    them is a goalkeeper or an official.
    """

    def test_exactly_three_unresolved_targets(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        m = importlib.import_module('kb_final_targets_server')
        st = m.build_state()
        assert st['u_reviewed'] == 48 and st['u_total'] == 48, 'the U pass IS complete'
        assert len(st['items']) == 3
        assert {x['BOX_ID'] for x in st['items']} == {
            'train:5089', 'train:7331', 'train:8656'}
        assert all(x['U_CATEGORY'] in ('AMBIGUOUS_TARGET', 'OCCLUDED_UNCLEAR')
                   for x in st['items'])
        assert all(x['FIRST_HUMAN_DECISION'] == 'U' for x in st['items'])

    def test_exclude_is_a_first_class_answer(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        m = importlib.import_module('kb_final_targets_server')
        assert dict((c[0], c[1]) for c in m.CHOICES) == {
            'P': 'player', 'G': 'goalkeeper', 'R': 'referee', 'E': 'EXCLUDE_IMAGE'}
        assert 'drop this IMAGE' in dict((c[1], c[2]) for c in m.CHOICES)['EXCLUDE_IMAGE']

    def test_ui_separates_reviewed_from_unresolved(self):
        src = (REPO / 'tools' / 'kb_final_targets_server.py').read_text(encoding='utf-8')
        assert '48/48 U cases REVIEWED' in src
        assert 'this pass is complete' in src
        assert 'still UNRESOLVED' in src

    def test_writes_are_mode_and_vocabulary_isolated(self):
        src = (REPO / 'tools' / 'kb_final_targets_server.py').read_text(encoding='utf-8')
        assert "MODE = 'final_target'" in src
        assert 'this server only writes final_target' in src
        assert 'value not in the final-target vocabulary' in src

    def test_gate_honours_a_role_or_an_exclusion(self):
        src = (REPO / 'tools' / 'kb_second_pass_gate.py').read_text(encoding='utf-8')
        assert "ft_dec = {b: v for (m, b), v in last.items() if m == 'final_target'}" in src
        assert "'RESOLVED_ON_THIRD_LOOK'" in src
        assert "r['FINAL_ACTION'] = 'EXCLUDE_IMAGE'" in src

    def test_exclude_image_is_a_disposition_not_a_class(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        assert 'EXCLUDE_IMAGE' in kb_decisions.U_CATEGORIES
        assert 'EXCLUDE_IMAGE' not in kb_decisions.ROLES
        assert kb_decisions.DISPOSITION_ACTION['EXCLUDE_IMAGE'] == \
            'EXCLUDE_IMAGE_FROM_CANDIDATE_SET'


class TestManualContextCorrections:
    """Black context boxes are actionable in the same missed_role pass.

    The retrospective ranking's recall was measured on 16 held-out positives, so
    a real official can sit in a box the queue never scored highly. Catching that
    during the pass is the difference between one sweep and two.
    """

    def test_context_boxes_carry_a_box_id_and_prior_state(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        m = importlib.import_module('kb_review_server2')
        st = m.build_state()
        ctx = [b for it in st['items'][:40] for b in it['context']]
        assert ctx, 'no context boxes surfaced'
        for b in ctx:
            assert b['BOX_ID'] and len(b['bbox']) == 4
            assert 'already' in b and 'already_mode' in b

    def test_manual_mode_is_separate_from_the_required_queue(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        m = importlib.import_module('kb_review_server2')
        st = m.build_state()
        cand = {c['BOX_ID'] for it in st['items'] for c in it['candidates']}
        ctx = {b['BOX_ID'] for it in st['items'] for b in it['context']}
        assert not (cand & ctx), 'a box cannot be both required and optional'
        assert st['total_boxes'] == 6684, 'manual work must not enlarge the queue'
        assert 'manual' in st

    def test_server_accepts_its_modes_and_only_role_values(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert "'missed_role_manual'" in src and "'missing_target_box'" in src
        assert "d.get('mode') not in MODES" in src
        assert 'value not in the role vocabulary' in src

    def test_bulk_actions_never_touch_context_boxes(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert 'context boxes are never bulk-assigned' in src
        assert "for(const c of it.candidates) if(!dec[c.BOX_ID])" in src

    def test_applier_includes_manual_mode(self):
        src = (REPO / 'tools' / 'kb_apply_review.py').read_text(encoding='utf-8')
        assert "'missed_role_manual'" in src
        assert "'final_target'" in src

    def test_gate_records_manual_corrections_without_requiring_them(self):
        g = load(XS / 'keremberke_review' / 'SECOND_PASS_GATE.json')
        assert 'manual_context_corrections' in g
        mc = g['manual_context_corrections']
        assert set(mc['by_class']) == {'player', 'goalkeeper', 'referee', 'uncertain'}
        assert 'never add required workload' in mc['note']

    def test_a_disposition_counts_as_settled_whatever_mode_made_it(self):
        src = (REPO / 'tools' / 'kb_apply_review.py').read_text(encoding='utf-8')
        assert 'elif cls in kb_decisions.U_CATEGORIES:' in src
        assert 'one box short of a complete review' in src

    def test_apply_obeys_both_gates(self):
        """A sandbox run wrote while the second-pass gate reported FAIL."""
        src = (REPO / 'tools' / 'kb_apply_review.py').read_text(encoding='utf-8')
        assert 'kb_second_pass_gate.py' in src
        assert 'REFUSED: the SECOND-PASS gate has not passed' in src
        assert "if not sp.get('passed'):" in src


class TestManualCorrectionClassification:
    """A click that agrees with an existing role is not a discovery.

    Counting one would inflate the single number this pass exists to produce --
    how many officials the retrospective sweep missed -- with re-confirmations of
    officials that were already found.
    """

    PKG = XS / 'keremberke_review'

    @pytest.fixture(scope='class')
    def kb(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        return kb_decisions

    def _classify(self, kb, tmp, rows):
        p = tmp / 'd.json'
        p.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
        return kb.classify_manual(p)

    def test_same_class_reconfirmation_is_a_no_op(self, kb, tmp_path):
        r = self._classify(kb, tmp_path, [
            {'mode': 'candidates', 'BOX_ID': 'x:1', 'HUMAN_FINAL_CLASS': 'referee',
             'recorded_utc': '2026-08-12T09:00:00Z'},
            {'mode': 'missed_role_manual', 'BOX_ID': 'x:1',
             'HUMAN_FINAL_CLASS': 'referee', 'recorded_utc': '2026-08-12T10:00:00Z'}])
        assert r['x:1']['kind'] == kb.NO_OP
        assert r['x:1']['prior_class'] == 'referee'
        assert r['x:1']['prior_mode'] == 'candidates'

    def test_unresolved_to_official_is_a_real_discovery(self, kb, tmp_path):
        for prior in (None, 'player'):
            rows = []
            if prior:
                rows.append({'mode': 'candidates', 'BOX_ID': 'x:2',
                             'HUMAN_FINAL_CLASS': prior,
                             'recorded_utc': '2026-08-12T09:00:00Z'})
            rows.append({'mode': 'missed_role_manual', 'BOX_ID': 'x:2',
                         'HUMAN_FINAL_CLASS': 'goalkeeper',
                         'recorded_utc': '2026-08-12T10:00:00Z'})
            r = self._classify(kb, tmp_path, rows)
            assert r['x:2']['kind'] == kb.NEW_CORRECTION, prior

    def test_changing_one_official_to_another_is_an_override(self, kb, tmp_path):
        r = self._classify(kb, tmp_path, [
            {'mode': 'candidates', 'BOX_ID': 'x:3', 'HUMAN_FINAL_CLASS': 'referee',
             'recorded_utc': '2026-08-12T09:00:00Z'},
            {'mode': 'missed_role_manual', 'BOX_ID': 'x:3',
             'HUMAN_FINAL_CLASS': 'goalkeeper',
             'recorded_utc': '2026-08-12T10:00:00Z'}])
        assert r['x:3']['kind'] == kb.OVERRIDE, 'an intentional change must be kept'

    def test_confirming_a_player_is_not_a_discovery(self, kb, tmp_path):
        r = self._classify(kb, tmp_path, [
            {'mode': 'missed_role_manual', 'BOX_ID': 'x:4',
             'HUMAN_FINAL_CLASS': 'player', 'recorded_utc': '2026-08-12T10:00:00Z'}])
        assert r['x:4']['kind'] == kb.NO_OP

    def test_uncertain_is_neither_find_nor_no_op(self, kb, tmp_path):
        r = self._classify(kb, tmp_path, [
            {'mode': 'missed_role_manual', 'BOX_ID': 'x:5',
             'HUMAN_FINAL_CLASS': 'uncertain',
             'recorded_utc': '2026-08-12T10:00:00Z'}])
        assert r['x:5']['kind'] == kb.FLAGGED

    def test_every_manual_click_is_classified_consistently(self, kb):
        """Counts move as the reviewer works; the rules must not."""
        r = kb.classify_manual(self.PKG / 'decisions.json')
        assert r, 'manual clicks exist and must all be classified'
        for box, v in r.items():
            assert v['kind'] in (kb.NO_OP, kb.NEW_CORRECTION, kb.OVERRIDE,
                                 kb.FLAGGED, kb.DISPOSITIONED), box
            if v['kind'] == kb.DISPOSITIONED:
                assert v['manual_class'] in kb.U_CATEGORIES, box
                continue
            if v['kind'] == kb.NO_OP and v['manual_class'] != 'player':
                assert v['manual_class'] == v['prior_class'], box
            if v['kind'] == kb.NEW_CORRECTION:
                assert v['prior_class'] in (None, 'player'), box
                assert v['manual_class'] in ('goalkeeper', 'referee'), box
            if v['kind'] == kb.OVERRIDE:
                assert v['prior_class'] in kb.ROLES
                assert v['manual_class'] != v['prior_class'], box

    def test_gate_reports_true_discoveries_separately(self):
        g = load(self.PKG / 'SECOND_PASS_GATE.json')
        m = g['manual_context_corrections']
        assert 'by_kind' in m and 'true_missed_role_discoveries' in m
        assert m['true_missed_role_discoveries'] == \
            m['by_kind'].get('NEW_MISSED_ROLE_CORRECTION', 0)
        assert 'must not inflate that number' in m['note']

    def test_server_stamps_the_kind_at_write_time(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert 'kb_decisions.classify_click(' in src
        assert "d['prior_class'] = pv" in src
        assert 'trusted from the page' in src

    def test_server_and_auditor_share_one_classifier(self):
        """Two copies of the rule would drift, and the drift would be silent.

        The kind stamped at write time is what the reviewer sees in the panel;
        the kind the auditor recomputes is what the gate reports. If they were
        computed by two separate blocks of code -- as they were -- a disposition
        could be stamped HUMAN_OVERRIDE live and DISPOSITION_SET in the audit.
        """
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert 'kb_decisions.prior_non_manual(' in src
        assert 'elif pv in kb_decisions.ROLES' not in src, \
            'the server must not carry its own copy of the rule'

    def test_ui_says_already_resolved(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert 'ALREADY RESOLVED' in src
        assert 'NO_OP_CONFIRMATION, not a new' in src
        assert 'HUMAN_OVERRIDE' in src


class TestMissingTargetBoxFlag:
    """Some real targets carry no annotation at all, so nothing can be clicked.

    Flagging them in-pass is what avoids a third sweep of all 1,133 images.
    """

    PKG = XS / 'keremberke_review'

    def test_flag_is_image_level_and_cannot_use_a_real_box_id(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert "MISSING_PREFIX = 'MISSING:'" in src
        assert 'a missing-target flag must use a ' in src
        assert "MISSING: keys belong to missing_target_box" in src
        assert "d['image_level'] = True" in src
        assert "d['no_box_exists'] = True" in src

    def test_no_box_is_created_or_inferred(self):
        src = (REPO / 'tools' / 'kb_missing_targets_queue.py').read_text(encoding='utf-8')
        assert 'No box is created or inferred here' in src
        q = self.PKG / 'missing_target_queue.json'
        if q.exists():
            assert load(q)['no_box_created_or_inferred'] is True

    def test_flags_do_not_touch_the_required_queue(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        m = importlib.import_module('kb_review_server2')
        st = m.build_state()
        assert st['total_boxes'] == 6684, 'flags must not enlarge the queue'
        assert 'missing' in st
        # a flag key can never be mistaken for a real annotation
        for f in st['missing']:
            assert f['key'].startswith('MISSING:')

    def test_queue_contains_only_flagged_images(self):
        q = self.PKG / 'missing_target_queue.json'
        if not q.exists():
            pytest.skip('queue not built')
        d = load(q)
        assert d['images'] <= d['flags']
        assert 'not the 1,133 reviewed' in d['not_another_full_pass']
        assert set(d['allowed_resolutions']) == {
            'boxed_player', 'boxed_goalkeeper', 'boxed_referee', 'EXCLUDE_IMAGE'}

    def test_gate_blocks_until_every_flag_is_resolved(self):
        src = (REPO / 'tools' / 'kb_second_pass_gate.py').read_text(encoding='utf-8')
        assert 'N2 every flagged MISSING_TARGET_BOX boxed or its image excluded' in src
        assert 'missing_pending' in src
        g = load(self.PKG / 'SECOND_PASS_GATE.json')
        cond = [c for c in g['gate'] if c['condition'].startswith('N2')]
        assert cond, 'the missing-target condition must be in the gate'
        mt = g['missing_target_boxes']
        # a flag leaves the pending set exactly one way: resolved, or withdrawn
        settled = mt['flagged'] - mt['pending']
        assert max(mt['resolved'], mt['retracted']) <= settled \
            <= mt['resolved'] + mt['retracted']

    def test_multiple_flags_per_image_are_possible(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        # the key carries a timestamp, so two flags on one image cannot collide
        assert "'MISSING:'+it.IMAGE+'#'+Date.now()" in src

    def test_bulk_actions_and_nav_are_disabled_while_flagging(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        for guard in ("k==='a'&&!qMode", "e.key==='Enter'&&!qMode",
                      "k==='n'&&!qMode"):
            assert guard in src, guard
        assert "if(e.key==='Escape')" in src, 'flagging must be cancellable'


class TestReviewPageActuallyParses:
    """The page is a raw Python string, so no linter ever sees it as code.

    A single raw newline inside a JS string literal made the whole script fail to
    parse. Every symptom pointed at lost work -- blank image, zeroed counters, no
    population -- while the server was in fact healthy: 200 on /, the full 2.9 MB
    on /api/state, 4,140 decisions loaded, and a clean log. Nothing in the Python
    test suite could see it, because nothing executed the page.
    """

    @pytest.fixture(scope='class')
    def srv(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        return importlib.import_module('kb_review_server2')

    def test_the_served_script_has_no_unterminated_literal(self, srv):
        assert srv.page_script_defects() == []

    def test_the_check_catches_the_defect_it_exists_for(self, srv):
        broken = srv.PAGE.replace('retracted?\\n', 'retracted?\n')
        assert broken != srv.PAGE, 'the prompt string moved; update this test'
        d = srv.page_script_defects(broken)
        assert d and 'unterminated' in d[0]

    def test_a_multiline_template_literal_is_not_a_false_positive(self, srv):
        """Backticked HTML spans lines legitimately and must stay legal."""
        assert srv.page_script_defects(
            '<script>const a=`line one\nline two ${x?`${y}`:1}`;</script>') == []

    def test_the_server_refuses_to_serve_a_broken_page(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert 'REFUSING TO SERVE' in src
        assert 'bad = page_script_defects()' in src
        assert 'No decision has been touched' in src

    def test_node_agrees_when_node_is_available(self, srv, tmp_path):
        node = shutil.which('node')
        if not node:
            pytest.skip('node not installed')
        i = srv.PAGE.find('<script')
        js = srv.PAGE[srv.PAGE.find('>', i) + 1:srv.PAGE.rfind('</script>')]
        f = tmp_path / 'page.js'
        f.write_text(js, encoding='utf-8')
        r = subprocess.run([node, '--check', str(f)], capture_output=True,
                           text=True)
        assert r.returncode == 0, r.stderr


class TestKeyboardShortcutsActuallyFire:
    """Press the keys against the real page, in a JS engine.

    M was advertised in the legend, had its counter, its colour and its whole
    server-side path, and did nothing: the keydown handler had no 'm' branch at
    all. Asserting on the page source would not have caught it -- 'M' appeared
    in the file several times. The only check that catches a dead shortcut is
    dispatching the event and watching for the POST.

    The page script is executed under a small DOM shim with fetch stubbed, so
    nothing is written and no server is needed.
    """

    HARNESS = REPO / 'tests' / 'js' / 'press_keys.js'

    @pytest.fixture(scope='class')
    def fired(self, tmp_path_factory):
        node = shutil.which('node')
        if not node:
            pytest.skip('node not installed')
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        srv = importlib.import_module('kb_review_server2')
        d = tmp_path_factory.mktemp('keys')
        (d / 'page.html').write_text(srv.PAGE, encoding='utf-8')
        (d / 'state.json').write_text(json.dumps(srv.build_state()),
                                      encoding='utf-8')
        r = subprocess.run([node, str(self.HARNESS), str(d / 'page.html'),
                            str(d / 'state.json')], capture_output=True,
                           text=True)
        assert r.returncode == 0, r.stderr[:2000]
        return json.loads(r.stdout)

    def test_m_fires_on_a_required_candidate(self, fired):
        p = fired['posts'][0]['post']
        assert p['mode'] == 'missed_role'
        assert p['HUMAN_FINAL_CLASS'] == 'NON_TARGET_HUMAN'

    def test_m_fires_on_a_clickable_context_box(self, fired):
        p = fired['posts'][1]['post']
        assert p['mode'] == 'missed_role_manual'
        assert p['HUMAN_FINAL_CLASS'] == 'NON_TARGET_HUMAN'

    def test_the_visible_button_does_the_same_thing(self, fired):
        """A dead key must never be able to block the review again."""
        assert fired['button']['wired'] and fired['button']['posted']
        assert fired['button']['post']['HUMAN_FINAL_CLASS'] == 'NON_TARGET_HUMAN'

    def test_the_live_counter_increments(self, fired):
        assert int(fired['counters']['before']) == 0
        assert int(fired['counters']['after_candidate']) == 1
        assert int(fired['counters']['after_context']) == 2

    def test_m_does_not_advance_to_the_next_image(self, fired):
        assert fired['image_index_unchanged']

    def test_m_during_a_flag_prompt_cancels_it_and_posts_nothing(self, fired):
        """Q then M must not send a value the server would reject."""
        assert fired['m_during_q']['posted_anything'] is False
        assert fired['m_during_q']['qMode_after'] is False
        assert fired['m_during_q']['banner'] == 'none'

    def test_every_other_shortcut_still_fires(self, fired):
        k = fired['other_keys']
        assert (k['p'], k['g'], k['r'], k['u']) == \
            ('player', 'goalkeeper', 'referee', 'uncertain')
        assert k['n = next image'] == 'advanced', 'N must still navigate'
        assert k['b = previous'].startswith('i=')


class TestRevisitingUncertainBoxes:
    """M was advertised while its key was dead, so U absorbed non-active humans.

    Those boxes are not wrong -- U is an honest answer -- but some of them were
    only U because the reviewer had no other key that worked. Nothing about them
    is changed automatically: they are surfaced so the reviewer can look again
    and decide, which is the only correct way to revisit a human decision.
    """

    PKG = XS / 'keremberke_review'

    @pytest.fixture(scope='class')
    def st(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        return importlib.import_module('kb_review_server2').build_state()

    def test_open_uncertain_boxes_are_exposed(self, st):
        assert 'u_open' in st
        for u in st['u_open']:
            assert u['BOX_ID'] and u['IMAGE']

    def test_only_still_unresolved_boxes_are_listed(self, st):
        """A U later superseded by a role must not be dragged back up."""
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        res = kb_decisions.resolve(self.PKG / 'decisions.json')
        for u in st['u_open']:
            assert res[u['BOX_ID']]['disposition'] == 'UNRESOLVED', u['BOX_ID']
            assert res[u['BOX_ID']]['final_class'] is None

    def test_the_list_matches_the_log(self, st):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        bm = kb_decisions.by_mode(self.PKG / 'decisions.json')
        res = kb_decisions.resolve(self.PKG / 'decisions.json')
        expect = {b for (m, b), v in bm.items()
                  if m in ('missed_role', 'missed_role_manual')
                  and v == 'uncertain'
                  and res[b]['disposition'] == 'UNRESOLVED'}
        assert {u['BOX_ID'] for u in st['u_open']} == expect

    def test_nothing_is_rewritten_to_expose_them(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert 'nothing is changed here' in src
        assert "open(PKG / 'decisions.json', 'w'" not in src


class TestNonActiveMatchHuman:
    """M -- bench, coach, ball person, medical, sideline staff.

    These are real humans and a reviewer will meet them constantly, but they are
    outside the four-class EyeCU ontology. Without a key for them the only
    available answers were a wrong role or 'uncertain': the first poisons the
    training set with a class the model must not learn, the second manufactures
    unresolved work that no later pass can ever settle, because there is nothing
    to settle -- the box is simply not an EyeCU target.
    """

    PKG = XS / 'keremberke_review'

    @pytest.fixture(scope='class')
    def kb(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        return kb_decisions

    def test_the_key_maps_to_the_existing_bucket_not_a_new_name(self, kb):
        """A second name for the same thing would split the count in two."""
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert "NON_ACTIVE = 'NON_TARGET_HUMAN'" in src, \
            'M must reuse the category the first pass already recorded'
        assert 'NON_TARGET_HUMAN' in kb.U_CATEGORIES
        assert 'NON_ACTIVE_MATCH_HUMAN' not in kb.U_CATEGORIES, \
            'the vocabulary must not grow a synonym'

    def test_it_is_accepted_for_required_and_for_context_boxes(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert 'ROLE_VALUES = ' in src
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        m = importlib.import_module('kb_review_server2')
        assert m.NON_ACTIVE in m.ROLE_VALUES
        assert 'missed_role' in m.MODES and 'missed_role_manual' in m.MODES

    def test_it_is_a_disposition_and_never_becomes_a_class(self, kb, tmp_path):
        p = tmp_path / 'd.json'
        p.write_text(json.dumps({'mode': 'missed_role', 'BOX_ID': 'x:9',
                                 'HUMAN_FINAL_CLASS': 'NON_TARGET_HUMAN',
                                 'recorded_utc': '2026-08-12T10:00:00Z'}),
                     encoding='utf-8')
        r = kb.resolve(p)['x:9']
        assert r['final_class'] is None, 'a coach must never become a player'
        assert r['disposition'] == 'NON_TARGET_HUMAN'
        assert r['action'] == 'REMOVE_ANNOTATION_KEEP_IMAGE'

    def test_a_manual_click_on_it_is_not_a_missed_role_discovery(self, kb):
        assert kb.classify_click(None, 'NON_TARGET_HUMAN') == kb.DISPOSITIONED
        assert kb.classify_click('player', 'NON_TARGET_HUMAN') == kb.DISPOSITIONED
        assert kb.classify_click(None, 'referee') == kb.NEW_CORRECTION

    def test_the_applier_settles_it_rather_than_leaving_it_pending(self):
        src = (REPO / 'tools' / 'kb_apply_review.py').read_text(encoding='utf-8')
        assert 'elif cls in kb_decisions.U_CATEGORIES' in src
        assert 'len(dispositioned)' in src, 'a disposition must count as settled'

    def test_a_non_active_human_cannot_be_flagged_as_a_missing_target(self):
        """Otherwise M would create annotation work for a box that must not exist."""
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert 'a non-active human is not a missing TARGET' in src


class TestMissingTargetRetraction:
    """A flag raised by mistake must be withdrawable, and auditably so.

    Two flags on one image are legitimate -- two officials can both be unboxed --
    so the fix cannot be de-duplication. It has to be an explicit, reasoned
    withdrawal of one named flag, leaving the other untouched.
    """

    PKG = XS / 'keremberke_review'

    def test_retraction_is_its_own_mode_and_needs_a_reason(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert "'missing_target_retraction'" in src
        assert 'a retraction needs a reason' in src
        assert 'only a MISSING: flag can be retracted' in src

    def test_only_a_flag_can_be_retracted_not_a_real_annotation(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        m = importlib.import_module('kb_review_server2')
        assert 'missing_target_retraction' in m.MODES
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        # a retraction is the one other mode that may carry a MISSING: key
        assert "d['mode'] != 'missing_target_retraction'" in src

    def test_the_log_is_never_rewritten(self):
        """History is the point: a withdrawn flag stays visible, marked."""
        src = (REPO / 'tools' / 'kb_missing_targets_queue.py').read_text(
            encoding='utf-8')
        assert 'stays in history' in src
        assert 'RETRACTED' in src
        for tool in ('kb_review_server2.py', 'kb_missing_targets_queue.py'):
            t = (REPO / 'tools' / tool).read_text(encoding='utf-8')
            assert "open(PKG / 'decisions.json', 'w'" not in t

    def test_a_retracted_flag_is_not_pending_work(self):
        q = self.PKG / 'missing_target_queue.json'
        if not q.exists():
            pytest.skip('queue not built')
        d = load(q)
        assert d['flags'] - d['pending'] == d['resolved'] + d['retracted']
        assert 'does not block the gate' in d['retraction_note']
        for r in d['rows']:
            assert r['status'] in ('PENDING', 'RESOLVED', 'RETRACTED')
            if r['status'] == 'RETRACTED':
                assert r['retraction_reason'], 'a withdrawal must say why'

    def test_a_retracted_flag_does_not_block_the_gate(self):
        src = (REPO / 'tools' / 'kb_second_pass_gate.py').read_text(encoding='utf-8')
        assert 'missing_ret' in src
        assert 'b not in missing_ret' in src, \
            'N2 must exclude retracted flags from pending'

    def test_the_reviewer_can_see_the_flags_on_the_current_image(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert 'retract' in src
        assert "'missing_target_retraction'" in src
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        m = importlib.import_module('kb_review_server2')
        st = m.build_state()
        for f in st['missing']:
            assert 'retracted' in f and 'retraction_reason' in f
        assert st['total_boxes'] == 6684, \
            'retraction must not disturb the required queue'
