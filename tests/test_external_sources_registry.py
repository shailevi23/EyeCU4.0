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
        assert modes <= {'candidates', 'qa_player', 'qa_nocand',
                         'u_resolution', 'missed_role'}, modes

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
        gi = (REPO / '.gitignore').read_text(encoding='utf-8')
        assert 'keremberke_review/ledger.json' in gi
        assert '!experiments/external_sources/keremberke_review/decisions.json' in gi


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
        assert set(g['blocking']) >= {'D all original U categorized',
                                      'F MISSED_ROLE_REVIEW complete'}

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
            assert '50/6684' in fline, fline
        finally:
            shutil.rmtree(self.PKG)
            bak.rename(self.PKG)

    def test_prior_decisions_survive_that_round_trip(self):
        d = self.PKG / 'decisions.json'
        modes = {json.loads(l).get('mode') for l in
                 d.read_text(encoding='utf-8').splitlines() if l.strip()}
        assert modes == {'candidates', 'qa_player', 'qa_nocand'}, modes

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
    def test_server_only_writes_missed_role(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert "if d.get('mode') != 'missed_role':" in src
        assert 'this server only writes missed_role' in src

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

    def test_it_resumes_only_its_own_mode(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert "if d.get('mode') == 'missed_role':" in src

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

    def test_no_already_settled_box_is_requeued(self, q):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        res = kb_decisions.resolve(self.PKG / 'decisions.json')
        settled = [r['BOX_ID'] for r in q['rows']
                   if res.get(r['BOX_ID'], {}).get('final_class')]
        assert settled == [], f'{len(settled)} already-answered boxes re-queued'

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
