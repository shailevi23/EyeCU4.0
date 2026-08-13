"""
The consolidated external-source workspace, as assertions.

The registry is the thing future decisions will be made from, so the checks here
are mostly about it not drifting from the audits it was built out of, and about
the rules that protect EyeCU's frozen artifacts holding: no external image in
VAL or TEST, no dataset bulk in git, no credential on disk, and no source
promoted to KEEP_ACTIVE without the evidence that justifies it.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Every mode any review tool is permitted to write. Passes are added as the
# review deepens, so pinning a literal set here would fail the moment a new one
# ships -- which is a snapshot, not an invariant. The invariant is that a mode in
# the log must be one some tool actually declares, so an unrecognised mode still
# fails loudly. Kept in one place because two copies drifted apart once already.
KNOWN_MODES = {'candidates', 'qa_player', 'qa_nocand', 'u_resolution',
               'final_target', 'missed_role', 'missed_role_manual',
               'missing_target_box', 'missing_target_resolution',
               'missing_target_retraction',
               # withdraws an image exclusion; never deletes it
               'missing_target_exclusion_retraction'}

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

    # decisions.json is irreplaceable human work and is tracked deliberately.
    # It grows without bound as the review proceeds, so a size cap on it would
    # mean "stop reviewing"; the cap exists to catch regenerable bulk, which
    # decisions.json is not.
    ALWAYS_TRACKED = {'experiments/external_sources/keremberke_review/'
                      'decisions.json'}

    def test_no_oversized_file_in_the_review_package(self):
        big = [(f, (REPO / f).stat().st_size) for f in self._tracked()
               if f.startswith('experiments/external_sources/keremberke_review/')
               and f not in self.ALWAYS_TRACKED
               and (REPO / f).exists()
               and (REPO / f).stat().st_size > 4_000_000]
        assert not big, f'oversized review-package files: {big}'

    def test_the_regenerable_bulk_is_still_untracked(self):
        """The cap above is not the only guard, and must not become it."""
        tracked = set(self._tracked())
        for f in ('ledger.json', 'review_queue.json',
                  'working_copy/train_annotations.coco.json'):
            p = 'experiments/external_sources/keremberke_review/' + f
            assert p not in tracked, f'{f} regenerates; it must not be tracked'


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
        assert modes <= KNOWN_MODES, modes - KNOWN_MODES

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

    def test_the_gate_and_apply_permission_agree(self):
        """Whether the gate passes changes as the review finishes; that it and
        apply_permitted always agree does not.

        This asserted `passed is False` while the second pass was outstanding.
        The second pass finished and all seventeen conditions now pass, so the
        assertion was pinning a moment. The invariant is the agreement, plus the
        separate guarantee that a written export still cannot happen while any
        per-case policy is unrecorded -- which kb_export_v2 enforces.
        """
        g2 = load(self.PKG / 'SECOND_PASS_GATE.json')
        failing = {c['condition'] for c in g2['gate'] if c['result'] == 'FAIL'}
        assert (g2['passed'] is True) == (not failing)
        assert set(g2['blocking']) == failing
        # apply is permitted only when the review AND the export policies are
        # both settled; a passing gate alone is not enough
        assert g2['apply_permitted'] == (g2['passed']
                                         and not g2['unresolved_case_policies'])

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
        # WHICH condition blocks changes as the review proceeds -- D before the U
        # pass, F until the sweep finished, N2 while targets need boxes. Naming
        # one pins a moment. The invariant is that blocking and passed agree, and
        # that a failure always blocks apply.
        assert (g['passed'] is False) == bool(g['blocking'])
        if g['blocking']:
            assert g['apply_permitted'] is False
        failed = {c['condition'] for c in g['gate'] if c['result'] == 'FAIL'}
        assert failed == set(g['blocking']), \
            'every failing condition must appear in blocking'

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
        assert modes <= KNOWN_MODES, modes - KNOWN_MODES

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
                         'missing_target_box', 'missing_target_retraction',
                         # only to drop the image of an unreadable target
                         'final_target'}
        # the modes that settle other passes stay out of reach of this server
        assert not modes & {'candidates', 'qa_player', 'qa_nocand',
                            'u_resolution', 'missing_target_resolution'}
        # and final_target here may record one thing only
        assert 'final_target here records only' in src
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
        """One per click. The starting value rises as the review proceeds, so
        only the increment is an invariant."""
        c = fired['counters']
        base = int(c['before'])
        assert int(c['after_candidate']) == base + 1
        assert int(c['after_context']) == base + 2

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


class TestOverlappingBoxHitTesting:
    """A small box inside a large one has to be reachable.

    Boxes were absolutely-positioned divs each carrying its own onclick, so the
    click went to whichever element the browser painted last -- candidates were
    appended after context boxes, and within each group the order came from the
    ledger. A small context box under a large candidate could not be selected at
    all, and the reviewer could not mark the person inside it.

    Selection is now resolved from box geometry: every box containing the point,
    smallest area first, cycling outward on repeated clicks in the same spot.
    The fixture is synthetic geometry driven through the real served script, so
    no real decision is touched.
    """

    HARNESS = REPO / 'tests' / 'js' / 'overlap_hit_test.js'

    @pytest.fixture(scope='class')
    def r(self, tmp_path_factory):
        node = shutil.which('node')
        if not node:
            pytest.skip('node not installed')
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        srv = importlib.import_module('kb_review_server2')
        d = tmp_path_factory.mktemp('overlap')
        (d / 'page.html').write_text(srv.PAGE, encoding='utf-8')
        (d / 'state.json').write_text(json.dumps(srv.build_state()),
                                      encoding='utf-8')
        out = subprocess.run([node, str(self.HARNESS), str(d / 'page.html'),
                              str(d / 'state.json')], capture_output=True,
                             text=True)
        assert out.returncode == 0, out.stderr[:2000]
        return json.loads(out.stdout)

    def test_selection_is_not_decided_by_dom_order(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert 'pointer-events:none' in src, 'boxes must not swallow the click'
        assert "getElementById('ov').onclick=hit" in src
        assert 'e.onclick=()=>{sel=b.BOX_ID' not in src, 'per-box onclick is back'
        assert 'e.onclick=()=>{sel=c.BOX_ID' not in src, 'per-box onclick is back'

    def test_every_containing_box_is_collected_smallest_first(self, r):
        assert r['group'] == ['fx:ctx_small', 'fx:ctx_done',
                              'fx:cand_big', 'fx:ctx_huge']

    def test_a_small_box_inside_a_large_one_is_selected_first(self, r):
        """The reported bug, exactly: large box wholly containing a smaller one."""
        assert r['small_inside_large']['group'] == ['fx:cand_big', 'fx:ctx_huge']
        assert r['small_inside_large']['sequence'] == \
            ['fx:cand_big', 'fx:ctx_huge', 'fx:cand_big']

    def test_repeated_clicks_cycle_and_wrap(self, r):
        assert r['cycle'] == ['fx:ctx_small', 'fx:ctx_done', 'fx:cand_big',
                              'fx:ctx_huge', 'fx:ctx_small']

    def test_a_single_box_selects_with_no_overlap_ui(self, r):
        assert r['single']['sel'] == 'fx:cand_far'
        assert r['single']['group'] == ['fx:cand_far']
        assert r['single']['panel'] == 'none'

    def test_partial_overlap_works_too(self, r):
        assert r['partial']['group'] == ['fx:cand_big', 'fx:ctx_huge']
        assert r['partial']['sel'] == 'fx:cand_big'

    def test_an_already_resolved_box_stays_reachable(self, r):
        """fx:ctx_done carries a role already; re-selecting it is deliberate."""
        assert 'fx:ctx_done' in r['group']

    def test_tab_and_shift_tab_cycle_without_the_mouse(self, r):
        assert r['tab'] == ['fx:ctx_small', 'fx:ctx_done', 'fx:cand_big',
                            'fx:ctx_done']

    def test_m_applies_to_the_selected_box_only(self, r):
        p = r['m_post']
        assert p['BOX_ID'] == 'fx:ctx_small'
        assert p['mode'] == 'missed_role_manual'
        assert p['HUMAN_FINAL_CLASS'] == 'NON_TARGET_HUMAN'
        assert r['post_boxes'] == ['fx:ctx_small'], \
            'no other box may be written by any of these clicks'

    def test_roles_behave_the_same_way(self, r):
        for v, got in r['roles'].items():
            assert got['box'] == 'fx:ctx_small' and got['value'] == v
            assert got['mode'] == 'missed_role_manual'

    def test_numeric_keys_still_select_candidates(self, r):
        assert r['numeric']['sel'] == 'fx:cand_far'
        assert r['numeric']['kind'] == 'cand'
        assert r['numeric']['group'] == [], 'picking by number ends the cycle'

    def test_flag_mode_selects_nothing(self, r):
        assert r['qmode']['selected_changed'] is False
        assert r['qmode']['qMode'] is True
        assert r['qmode']['group'] == []

    def test_the_panel_shows_the_overlap_position(self, r):
        assert 'OVERLAPPING BOXES 1/4' in r['panel']['html']
        assert r['panel']['buttons'] == 2, 'prev and next'
        assert 'fx:ctx_small' in r['selinfo']

    def test_bulk_actions_remain_candidate_only(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        block = src[src.index('async function allPlayer'):
                    src.index('function nextUnresolved')]
        assert 'it.context' not in block, \
            'A and Enter must never touch context boxes'
        assert block.count('it.candidates') == 2


class TestUncertainRevisitIsItsOwnQueue:
    """Cleaning up seven boxes must not mean re-entering a finished pass.

    REVISIT U BOXES used to jump inside the 6,684-box population: the header
    still read `image X/1133`, N still walked 1,133 images looking for work that
    no longer existed, and the reviewer had to navigate a completed queue to
    reach seven items. The queue is now its own thing -- its own header, its own
    navigation, its own idea of "remaining" -- and it is built from the effective
    state, so a box that was uncertain and has since been answered never enters.
    """

    HARNESS = REPO / 'tests' / 'js' / 'u_revisit.js'
    PKG = XS / 'keremberke_review'

    @pytest.fixture(scope='class')
    def r(self, tmp_path_factory):
        node = shutil.which('node')
        if not node:
            pytest.skip('node not installed')
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        srv = importlib.import_module('kb_review_server2')
        d = tmp_path_factory.mktemp('urev')
        (d / 'page.html').write_text(srv.PAGE, encoding='utf-8')
        (d / 'state.json').write_text(json.dumps(srv.build_state()),
                                      encoding='utf-8')
        out = subprocess.run([node, str(self.HARNESS), str(d / 'page.html'),
                              str(d / 'state.json')], capture_output=True,
                             text=True)
        assert out.returncode == 0, out.stderr[:2000]
        return json.loads(out.stdout)

    def test_the_header_stops_counting_the_finished_population(self, r):
        h = r['revisit_header']
        assert h['mode'] == 'UNCERTAIN REVISIT'
        assert h['pos'] == 'item 1/7'
        assert h['imgs'] == '7 remaining'
        assert '1133' not in json.dumps(h) and '/6684' not in json.dumps(h)
        assert '6,684/6,684 complete' in h['rem'], \
            'the finished pass is stated, not re-counted'

    def test_the_queue_is_only_the_effective_uncertains(self, r):
        assert len(r['queue']) == 7
        assert 'fx:ctx1' not in r['queue'], \
            'a box that was uncertain and is now resolved must never appear'

    def test_the_exact_box_is_auto_selected_and_highlighted(self, r):
        assert r['first']['sel'] == r['queue'][0]
        assert r['first']['at'] == 0
        assert r['highlight'] == 1, 'exactly one box carries the revisit highlight'
        assert 'UNCERTAIN REVISIT 1/7' in r['panel']

    def test_navigation_stays_inside_the_queue(self, r):
        assert r['n_stayed_in_queue']
        assert r['n_walk'][:7] == r['queue'], 'N walks the seven in order'
        assert r['n_walk'][7] == r['queue'][0], 'and wraps rather than escaping'
        assert r['after_b'] in r['queue']

    def test_answering_shrinks_the_queue_and_advances(self, r):
        assert r['after_answer']['queue'] == 6
        assert r['after_answer']['header'] == 'item 1/6'
        assert r['after_answer']['advanced_to'] != r['answered']['box']

    def test_an_answer_lands_on_that_box_only_and_in_its_own_mode(self, r):
        assert r['answered']['post']['BOX_ID'] == r['answered']['box']
        assert r['answered']['post']['HUMAN_FINAL_CLASS'] == 'goalkeeper'
        assert set(r['boxes_written']) == set(r['queue']), \
            'no box outside the queue was written'
        by_mode = {p['box']: p['mode'] for p in r['posts']}
        assert by_mode['fx:cand0'] == 'missed_role'
        assert by_mode['fx:ctx0'] == 'missed_role_manual', \
            'a context box keeps recording as a manual correction'

    def test_completion_is_announced(self, r):
        assert r['final']['queue'] == 0
        assert r['final']['banner_shown'] == 'block'
        assert 'U REVISIT COMPLETE' in r['final']['banner']

    def test_returning_to_the_full_review_restores_it(self, r):
        assert r['back_mode'] is False
        assert r['back_header']['mode'] == 'missed_role'
        assert r['back_header']['pos'].startswith('image ')

    def test_the_population_comes_from_the_resolver(self):
        """u_open is server-side, folded by resolve(), not by a queue file."""
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        block = src[src.index('u_open = ['):src.index('per, order = {}, {}')]
        assert "resolved[b]['disposition'] == 'UNRESOLVED'" in block
        assert 'kb_decisions.UNRESOLVED' in block

    def test_the_real_open_uncertains_match_the_resolver(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        import kb_decisions
        srv = importlib.import_module('kb_review_server2')
        st = srv.build_state()
        res = kb_decisions.resolve(self.PKG / 'decisions.json')
        for u in st['u_open']:
            assert res[u['BOX_ID']]['disposition'] == 'UNRESOLVED', u
        assert st['total_boxes'] == 6684, 'the cleanup queue never resizes the pass'

    def test_the_tool_can_open_straight_into_the_queue(self):
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert "'--u-revisit'" in src
        assert "location.search.indexOf('u=1')" in src


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


class TestEffectiveStateDecidesResolution:
    """G asked the wrong fold of the log.

    valid:2887 was answered 'uncertain' in qa_player and later settled as
    NON_TARGET_HUMAN in missed_role. Condition G read the stale per-mode answer
    and counted it unresolved, so it blocked the gate with nothing left to
    review -- the same shape as a condition asserting a historical fact.

    by_mode still answers completion ("was this queue worked through"), which
    genuinely is a per-mode question. It must never answer state.
    """

    PKG = XS / 'keremberke_review'

    @pytest.fixture(scope='class')
    def kb(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        return kb_decisions

    def test_a_later_cross_mode_answer_resolves_an_earlier_uncertain(self, kb,
                                                                     tmp_path):
        p = tmp_path / 'd.json'
        p.write_text('\n'.join(json.dumps(r) for r in [
            {'mode': 'qa_player', 'BOX_ID': 'x:1', 'HUMAN_FINAL_CLASS': 'uncertain',
             'recorded_utc': '2026-08-12T09:00:00Z'},
            {'mode': 'missed_role', 'BOX_ID': 'x:1',
             'HUMAN_FINAL_CLASS': 'NON_TARGET_HUMAN',
             'recorded_utc': '2026-08-12T10:00:00Z'}]), encoding='utf-8')
        assert kb.resolve(p)['x:1']['disposition'] == 'NON_TARGET_HUMAN'
        assert kb.by_mode(p)[('qa_player', 'x:1')] == 'uncertain', \
            'per-mode completion still sees the QA answer'

    def test_an_uncertain_that_was_never_revisited_stays_unresolved(self, kb,
                                                                    tmp_path):
        """G must not be weakened: a real open uncertain still blocks."""
        p = tmp_path / 'd.json'
        p.write_text(json.dumps(
            {'mode': 'missed_role', 'BOX_ID': 'x:2',
             'HUMAN_FINAL_CLASS': 'uncertain',
             'recorded_utc': '2026-08-12T10:00:00Z'}), encoding='utf-8')
        assert kb.resolve(p)['x:2']['disposition'] == 'UNRESOLVED'

    def test_the_gate_uses_the_resolver_not_the_per_mode_fold(self):
        src = (REPO / 'tools' / 'kb_second_pass_gate.py').read_text(encoding='utf-8')
        assert 'effective = kb_decisions.resolve(' in src
        assert 'last = kb_decisions.by_mode(' in src
        assert "effective.get(b, {}).get('disposition') == 'UNRESOLVED'" in src
        assert 'valid:2887' not in src, 'no box may be special-cased'

    def test_condition_G_is_not_implied_by_F(self):
        src = (REPO / 'tools' / 'kb_second_pass_gate.py').read_text(encoding='utf-8')
        g = src[src.index('G no systematic'):src.index('H non-target')]
        assert 'mr_unresolved' in g and 'qa_unresolved' in g, \
            'G must still require resolution, not just completion'

    def test_the_real_package_has_no_stale_qa_blocker(self):
        g = load(self.PKG / 'SECOND_PASS_GATE.json')
        cond = [c for c in g['gate'] if c['condition'].startswith('G')][0]
        # whatever G says, it must not be blocked by a box that is settled
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        res = kb_decisions.resolve(self.PKG / 'decisions.json')
        bm = kb_decisions.by_mode(self.PKG / 'decisions.json')
        stale = [b for (m, b), v in bm.items()
                 if m in ('qa_player', 'qa_nocand') and v == 'uncertain'
                 and res[b]['disposition'] != 'UNRESOLVED']
        assert stale, 'this test exists because such a box exists; keep it honest'
        for b in stale:
            assert b not in cond['detail'], f'{b} is settled and must not block'


class TestDerivedReportsCannotGoStaleSilently:
    """missing_target_queue.json read `0 flags` while 51 were on record.

    Nothing marked it stale, so it read as a current report showing no
    outstanding work. A gate that trusted such a file would pass N2 on a stale
    artefact rather than on the state of the review.
    """

    PKG = XS / 'keremberke_review'

    @pytest.fixture(scope='class')
    def kb(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        return kb_decisions

    def test_derived_reports_carry_the_log_fingerprint(self, kb):
        for name in ('missing_target_queue.json', 'SECOND_PASS_GATE.json'):
            d = load(self.PKG / name)
            assert 'source_log' in d, name
            assert d['source_log']['decisions_sha256']

    def test_a_report_from_another_log_is_detected(self, kb, tmp_path):
        d = tmp_path / 'd.json'
        d.write_text(json.dumps({'mode': 'candidates', 'BOX_ID': 'x:1',
                                 'HUMAN_FINAL_CLASS': 'player'}), encoding='utf-8')
        good = {'source_log': kb.log_version(d)}
        assert kb.is_stale(good, d) is False
        d.write_text(d.read_text(encoding='utf-8') + '\n' + json.dumps(
            {'mode': 'candidates', 'BOX_ID': 'x:2',
             'HUMAN_FINAL_CLASS': 'referee'}), encoding='utf-8')
        assert kb.is_stale(good, d) is True, 'an appended decision must invalidate it'

    def test_a_report_with_no_fingerprint_fails_closed(self, kb, tmp_path):
        d = tmp_path / 'd.json'
        d.write_text('', encoding='utf-8')
        assert kb.is_stale({}, d) is True
        assert kb.is_stale(None, d) is True

    def test_the_gate_folds_from_the_log_not_from_a_report(self):
        src = (REPO / 'tools' / 'kb_second_pass_gate.py').read_text(encoding='utf-8')
        assert "load(PKG / 'missing_target_queue.json'" not in src, \
            'the gate must not read the derived queue as state'

    def test_apply_refuses_a_stale_gate_report(self):
        src = (REPO / 'tools' / 'kb_apply_review.py').read_text(encoding='utf-8')
        assert 'kb_decisions.is_stale(sp' in src
        assert 'built from a different' in src

    def test_the_drawing_tool_folds_from_the_log(self):
        src = (REPO / 'tools' / 'kb_missing_target_server.py').read_text(
            encoding='utf-8')
        assert 'missing_target_queue.json' not in src.split('"""')[2], \
            'the tool must not build its queue from the derived report'


class TestImageResolutionIsDeterministic:
    """Two copies of the image path differed by one character.

    kb_review_server2 read `extracted`; kb_missing_target_server read `extract`.
    The queue, the state and the whole panel loaded correctly and every single
    image 404'd -- the worst split available, because everything says the tool
    works except the thing being reviewed. Both copies also searched by BASENAME
    with rglob, which returns whichever file the filesystem walked first if two
    ever share a name.

    IMAGE is metadata, not a search key: `<split>/<file>` resolves to exactly one
    path or to nothing.
    """

    PKG = XS / 'keremberke_review'

    @pytest.fixture(scope='class')
    def I(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_images
        return kb_images

    def test_there_is_one_resolver_and_both_servers_use_it(self):
        for tool in ('kb_review_server2.py', 'kb_missing_target_server.py'):
            src = (REPO / 'tools' / tool).read_text(encoding='utf-8')
            assert 'import kb_images' in src, tool
            assert 'IMGROOT = kb_images.IMGROOT' in src, tool
            assert 'kb_images.read(' in src, tool
            assert '.rglob(' not in src, f'{tool} still searches by basename'

    def test_the_image_root_exists_and_holds_the_three_splits(self, I):
        assert I.IMGROOT.is_dir(), I.IMGROOT
        for s in I.SPLITS:
            assert (I.IMGROOT / s).is_dir(), s

    def test_a_split_prefixed_image_maps_to_one_path(self, I):
        p = I.path_for('test/x.jpg')
        assert p == I.IMGROOT / 'test' / 'x.jpg'
        assert I.path_for('test\\x.jpg') == p, 'backslashes normalise'
        assert I.path_for('/test/x.jpg') == p, 'a leading slash is not a new root'

    def test_traversal_and_junk_are_refused(self, I):
        for bad in ('train/../../secret.jpg', '../x.jpg', 'nosuchsplit/x.jpg',
                    'x.jpg', '', None, '/'):
            with pytest.raises(I.ImageError):
                I.path_for(bad)

    def test_each_split_serves(self, I):
        led = json.loads((self.PKG / 'ledger.json').read_text(encoding='utf-8'))
        seen = {}
        for r in led:
            seen.setdefault(r['IMAGE'].split('/', 1)[0], r['IMAGE'])
        assert set(seen) == set(I.SPLITS), seen
        for split, im in seen.items():
            body, ctype = I.read(im)
            assert body[:2] == b'\xff\xd8', f'{split} is not a JPEG'
            assert ctype == 'image/jpeg'

    def test_a_non_ascii_parent_path_works(self, I, tmp_path):
        """The project lives under a Hebrew path; cv2.imread already failed on it."""
        root = tmp_path / 'שולחן העבודה' / 'תמונות'
        (root / 'train').mkdir(parents=True)
        (root / 'train' / 'זה.jpg').write_bytes(b'\xff\xd8\xff\xe0stub')
        body, ctype = I.read('train/זה.jpg', root=root)
        assert body[:2] == b'\xff\xd8' and ctype == 'image/jpeg'

    def test_a_url_encoded_path_resolves(self, I, tmp_path):
        from urllib.parse import quote, unquote
        root = tmp_path / 'r'
        (root / 'valid').mkdir(parents=True)
        name = 'a b+c%d.jpg'
        (root / 'valid' / name).write_bytes(b'\xff\xd8ok')
        # the servers unquote before handing the value over
        assert I.read(unquote(quote(f'valid/{name}')), root=root)[0][:2] == b'\xff\xd8'

    def test_a_missing_image_names_the_path_it_looked_for(self, I):
        with pytest.raises(I.ImageError) as e:
            I.resolve('train/__definitely_not_here__.jpg')
        assert 'no file at' in str(e.value)
        assert 'extracted' in str(e.value), 'the message must show the root used'

    def test_the_servers_return_an_explicit_404_and_log_it(self):
        for tool in ('kb_review_server2.py', 'kb_missing_target_server.py'):
            src = (REPO / 'tools' / tool).read_text(encoding='utf-8')
            assert "print(f'IMAGE 404" in src, tool
            assert "{'error': str(e), 'IMAGE': want}" in src, tool

    def test_the_page_shows_a_banner_not_a_broken_icon(self):
        src = (REPO / 'tools' / 'kb_missing_target_server.py').read_text(
            encoding='utf-8')
        assert 'im.onerror' in src
        assert 'IMAGE COULD NOT BE LOADED' in src
        assert 'Do not resolve this target' in src

    def test_the_tool_refuses_to_start_if_an_image_is_unresolvable(self):
        src = (REPO / 'tools' / 'kb_missing_target_server.py').read_text(
            encoding='utf-8')
        assert 'REFUSING TO START' in src
        assert 'kb_images.preflight(pending)' in src
        assert 'No decision has been touched' in src

    def test_preflight_reports_every_problem_not_just_the_first(self, I, tmp_path):
        root = tmp_path / 'r'
        (root / 'train').mkdir(parents=True)
        (root / 'train' / 'here.jpg').write_bytes(b'\xff\xd8')
        ok, probs = I.preflight(['train/here.jpg', 'train/gone.jpg',
                                 'train/also_gone.jpg'], root=root)
        assert ok is False and len(probs) == 2
        ok2, probs2 = I.preflight(['train/here.jpg'], root=root)
        assert ok2 is True and probs2 == []

    def test_every_live_missing_target_image_resolves(self, I):
        """The read-only preflight over the real queue."""
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        m = importlib.import_module('kb_missing_target_server')
        st = m.build_state()
        imgs = {t['IMAGE'] for t in st['targets']}
        assert imgs, 'there is outstanding work'
        ok, probs = I.preflight(imgs)
        assert ok, probs
        for im in imgs:
            assert I.resolve(im).stat().st_size > 0
        # one flag -> one image, and no image resolves two ways
        assert len({I.path_for(i) for i in imgs}) == len(imgs)


class TestMissingTargetDrawingTool:
    """The 48 targets that have no annotation and therefore no BOX_ID."""

    PKG = XS / 'keremberke_review'

    @pytest.fixture(scope='class')
    def m(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        return importlib.import_module('kb_missing_target_server')

    def test_geometry_is_validated_against_the_original_image(self, m):
        W, H = 1280, 720
        ok, err = m.validate_bbox([10.0, 20.0, 30.0, 40.0], W, H)
        assert err is None and ok == [10.0, 20.0, 30.0, 40.0]
        for bad, why in (([-5, 10, 20, 20], 'negative origin'),
                         ([10, -5, 20, 20], 'negative origin'),
                         ([10, 10, 0, 20], 'zero width'),
                         ([10, 10, 20, -3], 'negative height'),
                         ([W - 2, 10, 500, 20], 'off the right edge'),
                         ([10, H - 2, 20, 500], 'off the bottom'),
                         ([10, 10, 0.4, 0.4], 'sub-pixel'),
                         ('nope', 'not a list'),
                         ([1, 2, 3], 'wrong length')):
            got, e = m.validate_bbox(bad, W, H)
            assert got is None and e, why

    def test_dimensions_must_be_known(self, m):
        got, err = m.validate_bbox([1, 1, 2, 2], None, None)
        assert got is None and 'dimensions' in err

    def test_geometry_is_stored_in_image_coordinates(self, m):
        src = (REPO / 'tools' / 'kb_missing_target_server.py').read_text(
            encoding='utf-8')
        assert "'coordinate_space': 'original image pixels'" in src
        assert 'ORIGINAL IMAGE COORDINATES' in src
        # the drawing surface converts on the way in, so a resize cannot move it
        assert 'function toImage(ev)' in src
        assert 'getBoundingClientRect' in src

    def test_no_model_proposal_may_create_geometry(self, m):
        src = (REPO / 'tools' / 'kb_missing_target_server.py').read_text(
            encoding='utf-8')
        assert "'geometry_author': 'human drawn'" in src
        assert "'no_model_proposal_used': True" in src
        assert 'No model runs here' in src

    def test_a_resolution_must_name_one_live_flag(self, m):
        src = (REPO / 'tools' / 'kb_missing_target_server.py').read_text(
            encoding='utf-8')
        assert 'unknown or retracted flag' in src
        assert "'missing_target_id': key" in src

    def test_a_drawn_target_cannot_be_left_uncertain(self, m):
        src = (REPO / 'tools' / 'kb_missing_target_server.py').read_text(
            encoding='utf-8')
        assert 'not \\n' not in src
        assert b'uncertain' in src.encode() and 'must be classified' in src
        assert m.ROLES == ('player', 'goalkeeper', 'referee')

    def test_exclusion_needs_a_reason_and_is_scoped_to_one_image(self, m):
        src = (REPO / 'tools' / 'kb_missing_target_server.py').read_text(
            encoding='utf-8')
        assert 'an exclusion needs a reason' in src
        assert "v['IMAGE'] == img" in src
        assert 'resolves_flags_in_image' in src
        assert 'and nothing else' in src

    def test_the_log_is_append_only(self, m):
        src = (REPO / 'tools' / 'kb_missing_target_server.py').read_text(
            encoding='utf-8')
        assert "open(PKG / 'decisions.json', 'a'" in src
        assert "open(PKG / 'decisions.json', 'w'" not in src
        assert "'supersedes'" in src, 'a redraw records what it replaces'

    def test_the_queue_is_only_the_live_flags(self, m):
        st = m.build_state()
        assert st['targets'], 'there is outstanding work'
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        rows = kb_decisions.read_log(self.PKG / 'decisions.json')
        retr = {r['BOX_ID'] for r in rows
                if r['mode'] == 'missing_target_retraction'}
        keys = {t['key'] for t in st['targets']}
        assert keys & retr == set(), 'a retracted flag is not work'
        assert all(k.startswith('MISSING:') for k in keys)

    def test_it_carries_existing_annotations_for_context_only(self, m):
        st = m.build_state()
        t = st['targets'][0]
        assert t['existing'], 'context must be visible'
        assert 'BOX_ID' not in str(t['existing'][0]), \
            'context is drawn, not made editable here'

    def test_the_gate_accepts_only_documented_resolutions(self):
        src = (REPO / 'tools' / 'kb_second_pass_gate.py').read_text(encoding='utf-8')
        assert "kb_decisions.missing_targets(" in src
        assert "v['state'] == 'PENDING'" in src,             'an unrecognised value must leave the flag PENDING'
        g = load(self.PKG / 'SECOND_PASS_GATE.json')['missing_target_boxes']
        assert g['flagged'] - g['pending'] == \
            g['boxed'] + g['excluded'] + g['retracted']

    def test_uncertain_targets_can_be_excluded_rather_than_left_open(self):
        """A target with no readable role must have an honest way out."""
        src = (REPO / 'tools' / 'kb_review_server2.py').read_text(encoding='utf-8')
        assert "id=\"uExcl\"" in src
        assert 'final_target' in src and 'an exclusion needs a reason' in src
        assert 'EXCLUDE = ' in src


class TestContractV2:
    """The export contract the review actually needs, and its two open policies.

    v1 promised "only class ids may change" and enforced it with one assertion
    over the whole annotation list. The review then produced 46 additions, 37
    removals, 7 geometry repairs and an image exclusion -- every one of which
    that assertion aborts on. The fix is not a looser check but a narrower one
    per kind of change, each traced to the decision that authorised it.
    """

    PKG = XS / 'keremberke_review'

    @pytest.fixture(scope='class')
    def E(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        return importlib.import_module('kb_export_v2')

    @pytest.fixture(scope='class')
    def kb(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        return kb_decisions

    def test_the_exporter_decides_no_policy(self, E):
        src = (REPO / 'tools' / 'kb_export_v2.py').read_text(encoding='utf-8')
        assert 'THIS FILE DECIDES NOTHING' in src
        assert 'BALL_WRONG_HUMAN_BOX is NOT here' in src, \
            'ball cases must not default to removal'
        assert 'BALL_WRONG_HUMAN_BOX' not in E.REMOVE
        assert 'PARTIAL_BODY_BAD_BOX' not in E.REMOVE

    def test_it_refuses_while_a_case_policy_is_unrecorded(self, E):
        """Choosing a default here would BE the policy decision."""
        S = E.load_state()
        blockers = E.unresolved_policies(S)
        for b, d in blockers:
            assert d in E.NEEDS_CASE
        r = subprocess.run([sys.executable, 'tools/kb_export_v2.py', '--check'],
                           capture_output=True, text=True, cwd=str(REPO))
        if blockers:
            assert r.returncode == 1 and 'REFUSED' in r.stdout
        else:
            assert r.returncode == 0

    def test_a_repaired_box_is_found_by_its_event_not_its_disposition(self, E):
        """A geometry_repair records a role, which clears the disposition.

        Reading the effective disposition would make a repaired box look like an
        ordinary class change and silently drop its new geometry.
        """
        src = (REPO / 'tools' / 'kb_export_v2.py').read_text(encoding='utf-8')
        assert "if b in S['reps']:" in src
        assert "if b in S['balls']:" in src
        assert 'dispositioned_ever' in src

    def test_repairs_and_reclassifications_keep_their_annotation_id(self, E):
        src = (REPO / 'tools' / 'kb_export_v2.py').read_text(encoding='utf-8')
        block = src[src.index('def new_ids'):src.index('def build')]
        assert 'keep their original id' in block
        assert "P['repaired']" not in block, 'a repair must not be given a new id'

    def test_new_ids_are_deterministic_and_collision_safe(self, E):
        S = E.load_state()
        if E.unresolved_policies(S):
            pytest.skip('export blocked; ids exercised in the sandbox run')
        _, P, out, rep = E.build(None)
        wc = {s: json.loads((self.PKG / 'working_copy' /
                             f'{s}_annotations.coco.json').read_text('utf-8'))
              for s in E.SPLITS}
        src_ids = {f'{s}:{a["id"]}' for s in E.SPLITS for a in wc[s]['annotations']}
        for b, i in rep['id_map'].items():
            s = P['added'][b]['IMAGE'].split('/')[0] if b in P['added'] \
                else b.split(':')[0]
            assert f'{s}:{i}' not in src_ids
        _, _, _, rep2 = E.build(None)
        assert rep['id_map'] == rep2['id_map']

    def test_the_ball_rule_is_narrowed_not_loosened(self, E, kb):
        """Originals set-identical; only human-approved repairs may be added."""
        src = (REPO / 'tools' / 'kb_export_v2.py').read_text(encoding='utf-8')
        assert 'C6  every ORIGINAL ball annotation survives set-identical' in src
        assert 'only explicitly' in src and 'human-approved ball repairs' in src
        assert kb.BALL_ACTIONS == ('RECLASSIFY_TO_BALL', 'DRAW_BALL_BOX',
                                   'REMOVE_ONLY')

    def test_an_excluded_image_leaves_the_split_entirely(self, E):
        src = (REPO / 'tools' / 'kb_export_v2.py').read_text(encoding='utf-8')
        assert "a['images'] = [im for im in a['images'] if im['id'] not in drop_img]" in src
        assert "if b in P['excluded_ann'] or b in P['removed']" in src
        assert 'keeps its file' in src

    def test_reconciliation_is_set_equal_not_count_equal(self, E):
        src = (REPO / 'tools' / 'kb_export_v2.py').read_text(encoding='utf-8')
        assert 'out_orig == expect' in src
        assert 'SET-equal' in src

    def test_geometry_is_frozen_except_for_authorised_repairs(self, E):
        src = (REPO / 'tools' / 'kb_export_v2.py').read_text(encoding='utf-8')
        assert 'moved <= authorised' in src

    def test_the_repair_tool_never_proposes_geometry(self):
        src = (REPO / 'tools' / 'kb_geometry_repair_server.py').read_text(
            encoding='utf-8')
        assert 'No model runs' in src
        assert "'geometry_author': 'human drawn'" in src
        assert "'no_model_proposal_used': True" in src
        assert "'annotation_id_preserved': True" in src
        assert "'original_bbox_xywh': l['bbox_xywh']" in src

    def test_the_repair_tool_membership_is_historical(self):
        """Otherwise a repaired box drops out of its own queue."""
        src = (REPO / 'tools' / 'kb_geometry_repair_server.py').read_text(
            encoding='utf-8')
        assert 'Membership is historical' in src
        assert "d.get('HUMAN_FINAL_CLASS') == want" in src

    def test_the_two_queues_are_the_right_size(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import importlib
        m = importlib.import_module('kb_geometry_repair_server')
        assert len(m.queue_ids('partial')) == 7
        assert len(m.queue_ids('ball')) == 5

    def test_a_ball_case_action_must_be_one_of_the_three(self, kb):
        src = (REPO / 'tools' / 'kb_geometry_repair_server.py').read_text(
            encoding='utf-8')
        assert 'kb_decisions.BALL_ACTIONS' in src
        assert 'documented ball outcomes' in src

    def test_v1_apply_cannot_write_a_v2_export(self):
        """v1 asserts the whole (id, bbox) set is unchanged; v2 changes it."""
        src = (REPO / 'tools' / 'kb_apply_review.py').read_text(encoding='utf-8')
        assert 'assert before == after' in src, \
            'if this assertion goes, v1 must have been replaced deliberately'


class TestImageExclusionIsReversible:
    """A reviewer pressed X meaning "drop this duplicate flag".

    X excluded the whole image, and because exclusion is recorded as one
    resolution per flag it landed on top of two boxes that had already been
    drawn -- burying twelve minutes of correct work under a keypress meant for
    something else. Nothing was lost, because the log is append-only, but the
    EFFECTIVE state was wrong and there was no way back.

    The fold now ignores a withdrawn exclusion, so each flag reverts to whatever
    it held before by itself. Nothing is reconstructed and nothing is deleted.
    """

    PKG = XS / 'keremberke_review'
    IMG = 'valid/54622_jpg.rf.8626a293639ac4a1eb395d11358994ae.jpg'

    @pytest.fixture(scope='class')
    def kb(self):
        import sys as _s
        _s.path.insert(0, str(REPO / 'tools'))
        import kb_decisions
        return kb_decisions

    def _log(self, tmp, rows):
        p = tmp / 'd.json'
        p.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
        return p

    def _scenario(self, kb, tmp, withdraw=False, retract=()):
        """4 flags: two boxed, two pending, then the image is excluded."""
        img = 'valid/x.jpg'
        keys = [f'MISSING:{img}#{n}' for n in range(1, 5)]
        rows = []
        for n, k in enumerate(keys):
            rows.append({'mode': kb.FLAG_MODE, 'BOX_ID': k, 'IMAGE': img,
                         'HUMAN_FINAL_CLASS': ['player', 'goalkeeper',
                                               'goalkeeper', 'player'][n],
                         'recorded_utc': f'2026-08-12T19:0{n}:00Z'})
        rows.append({'mode': kb.RESOLVE_MODE, 'BOX_ID': keys[0], 'IMAGE': img,
                     'HUMAN_FINAL_CLASS': 'boxed_player', 'role': 'player',
                     'bbox_xywh': [861.2, 138.6, 63.0, 207.0],
                     'recorded_utc': '2026-08-13T08:14:06Z'})
        rows.append({'mode': kb.RESOLVE_MODE, 'BOX_ID': keys[1], 'IMAGE': img,
                     'HUMAN_FINAL_CLASS': 'boxed_goalkeeper', 'role': 'goalkeeper',
                     'bbox_xywh': [1082.2, 165.6, 42.0, 86.0],
                     'recorded_utc': '2026-08-13T08:14:14Z'})
        for k in keys:
            rows.append({'mode': kb.RESOLVE_MODE, 'BOX_ID': k, 'IMAGE': img,
                         'HUMAN_FINAL_CLASS': 'EXCLUDE_IMAGE',
                         'reason': 'oops', 'recorded_utc': '2026-08-13T08:14:40Z'})
        if withdraw:
            for k in keys:
                rows.append({'mode': kb.UNEXCLUDE_MODE, 'BOX_ID': k, 'IMAGE': img,
                             'HUMAN_FINAL_CLASS': None, 'reason': 'accidental',
                             'recorded_utc': '2026-08-13T09:00:00Z'})
        for k in retract:
            rows.append({'mode': kb.RETRACT_MODE, 'BOX_ID': keys[k], 'IMAGE': img,
                         'HUMAN_FINAL_CLASS': None, 'reason': 'duplicate flag',
                         'recorded_utc': '2026-08-13T09:0%d:00Z' % (k + 1)})
        return keys, kb.missing_targets(self._log(tmp, rows))

    def test_exclusion_makes_all_four_excluded(self, kb, tmp_path):
        keys, mt = self._scenario(kb, tmp_path)
        assert [mt[k]['state'] for k in keys] == ['EXCLUDED'] * 4

    def test_withdrawal_restores_boxes_and_pending_states(self, kb, tmp_path):
        keys, mt = self._scenario(kb, tmp_path, withdraw=True)
        assert [mt[k]['state'] for k in keys] == \
            ['BOXED', 'BOXED', 'PENDING', 'PENDING']

    def test_restored_geometry_is_identical(self, kb, tmp_path):
        keys, mt = self._scenario(kb, tmp_path, withdraw=True)
        assert mt[keys[0]]['role'] == 'player'
        assert mt[keys[0]]['bbox_xywh'] == [861.2, 138.6, 63.0, 207.0]
        assert mt[keys[1]]['role'] == 'goalkeeper'
        assert mt[keys[1]]['bbox_xywh'] == [1082.2, 165.6, 42.0, 86.0]

    def test_retracting_one_flag_leaves_the_others_alone(self, kb, tmp_path):
        keys, mt = self._scenario(kb, tmp_path, withdraw=True, retract=(2,))
        assert [mt[k]['state'] for k in keys] == \
            ['BOXED', 'BOXED', 'RETRACTED', 'PENDING']

    def test_the_intended_final_state(self, kb, tmp_path):
        keys, mt = self._scenario(kb, tmp_path, withdraw=True, retract=(2, 3))
        states = [mt[k]['state'] for k in keys]
        assert states == ['BOXED', 'BOXED', 'RETRACTED', 'RETRACTED']
        assert states.count('EXCLUDED') == 0 and states.count('PENDING') == 0

    def test_the_image_is_no_longer_excluded(self, kb, tmp_path):
        keys, mt = self._scenario(kb, tmp_path, withdraw=True, retract=(2, 3))
        assert not any(v['excluded'] for v in mt.values())

    def test_the_exclusion_event_is_never_deleted(self, kb, tmp_path):
        keys, mt = self._scenario(kb, tmp_path, withdraw=True, retract=(2, 3))
        for k in keys:
            vals = [h['value'] for h in mt[k]['history']]
            assert 'EXCLUDE_IMAGE' in vals, 'history must stay auditable'
            assert mt[k]['exclusion_withdrawn'] is True

    def test_the_real_image_recovered(self, kb):
        """The actual incident, as it now stands in the real log."""
        mt = kb.missing_targets(self.PKG / 'decisions.json')
        here = {k: v for k, v in mt.items() if v['IMAGE'] == self.IMG}
        assert len(here) == 4
        boxed = [v for v in here.values() if v['state'] == 'BOXED']
        retr = [v for v in here.values() if v['state'] == 'RETRACTED']
        assert len(boxed) == 2 and len(retr) == 2
        assert not any(v['excluded'] for v in here.values())
        assert {v['role'] for v in boxed} == {'player', 'goalkeeper'}
        # every one of the four still carries the accidental exclusion in history
        assert all(any(h['value'] == 'EXCLUDE_IMAGE' for h in v['history'])
                   for v in here.values())

    def test_the_gate_reads_effective_exclusion_not_history(self):
        src = (REPO / 'tools' / 'kb_second_pass_gate.py').read_text(encoding='utf-8')
        assert 'kb_decisions.missing_targets(' in src
        assert 'asserting a historical fact' in src

    def test_a_withdrawal_and_a_flag_retraction_are_different_actions(self):
        src = (REPO / 'tools' / 'kb_missing_target_server.py').read_text(
            encoding='utf-8')
        assert '/api/retract_flag' in src and '/api/unexclude' in src
        assert "'scope': 'THIS FLAG ONLY'" in src
        assert 'a retraction needs a reason' in src
        # and the UI must not let them be confused
        assert 'RETRACT THIS FLAG ONLY' in src
        assert 'EXCLUDE THE WHOLE IMAGE' in src
        assert 'It is NOT the same as retracting one flag' in src

    def test_excluding_over_a_drawn_box_needs_acknowledgement(self):
        src = (REPO / 'tools' / 'kb_missing_target_server.py').read_text(
            encoding='utf-8')
        assert 'acknowledge_overrides_boxes' in src
        assert "'boxed_flags': sorted(would_bury)" in src
        assert 'ALREADY HAVE a ' in src, 'the page must warn before the server does'
        assert "prompt('Type EXCLUDE to override" in src

    def test_D_is_bound_and_does_not_collide(self):
        src = (REPO / 'tools' / 'kb_missing_target_server.py').read_text(
            encoding='utf-8')
        assert "k==='d'" in src and 'retractFlag()' in src
        keys = re.findall(r"k===\'([a-z])\'", src)
        assert len(keys) == len(set(keys)), f'duplicate shortcut: {keys}'


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
        assert "v['state'] == 'RETRACTED'" in src
        assert "v['state'] == 'PENDING'" in src, \
            'N2 must count only PENDING flags, which excludes retracted ones'

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
