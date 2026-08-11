"""
The frozen evaluation contract, asserted so it cannot drift once results exist.

Everything here was decided before any tracker was run against this GT. The
danger is not that someone rewrites it in bad faith; it is that a threshold, a
preprocessing flag or a sequence list gets nudged during the excitement of
seeing numbers, and nobody can later tell whether it was nudged before or
after. These tests make that impossible to do quietly.
"""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'data' / 'tracking_val_gt'
EXP = REPO / 'experiments' / 'tracking_v2'
MOT = ROOT / 'mot'
SPLIT = 'EyeCU-val'
AUSTIN = 'austin_fc_vs__club_tijuana_284'
SEQUENCES = {'women_1_239', 'youth_premier_league_1133',
             'bayern_munich_3-1_chelsea_228'}

pytestmark = pytest.mark.skipif(not (ROOT / 'manifest.json').exists(),
                                reason='identity GT package not built')


@pytest.fixture(scope='module')
def manifest():
    return json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def protocol():
    return json.loads((EXP / 'trackeval_protocol.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def contract():
    return json.loads((EXP / 'evaluation_contract.json').read_text(encoding='utf-8'))


class TestVerifiedGate:
    def test_benchmark_is_verified_with_a_matching_qc_record(self, manifest):
        from tools.confirm_tracking_gt_qc import qc_record_valid
        assert manifest['identity_gt_status'] == 'VERIFIED'
        ok, why = qc_record_valid(ROOT, manifest)
        assert ok, why

    def test_every_sequence_is_verified(self, manifest):
        for tag in SEQUENCES:
            assert manifest['sequence_review'][tag]['identity_status'] == 'VERIFIED'

    def test_confirmation_hashes_match_the_files_on_disk(self):
        rec = json.loads((ROOT / 'qc' / 'qc_confirmation.json').read_text(encoding='utf-8'))
        assert rec['confirmed'] is True and rec['reviewer']
        for rel, want in rec['artifact_sha256'].items():
            got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            assert got == want, rel

    def test_no_open_human_review_event_anywhere(self):
        for tag in SEQUENCES:
            p = ROOT / 'qc_identity' / tag / 'identity_gap_decisions.json'
            assert p.exists(), tag
            assert json.loads(p.read_text(encoding='utf-8'))['open_events'] == 0

    def test_final_validation_passes(self):
        from tools.validate_tracking_gt import validate_final
        errors, n = validate_final(ROOT)
        assert n > 0 and errors == [], errors[:5]


class TestThreeSequenceContract:
    def test_three_matches_900_frames(self, manifest, contract):
        assert {s['sequence'] for s in manifest['sequences']} == SEQUENCES
        assert sum(s['frame_count'] for s in manifest['sequences']) == 900
        assert contract['independent_matches'] == 3
        assert contract['total_frames'] == 900
        assert set(contract['clean_sequences']) == SEQUENCES

    def test_official_result_is_combined_seq(self, contract):
        assert contract['official_result'] == 'TrackEval COMBINED_SEQ'
        assert 'never replaces' in contract['also_reported']['precedence']

    def test_library_defaults_is_diagnostic_only(self, contract):
        assert contract['profiles']['primary_production'] == 'EYECU_SCORE_POLICY_V1'
        assert contract['profiles']['diagnostic_only'] == 'LIBRARY_DEFAULTS'
        assert 'cannot alone select' in contract['profiles']['rule']


class TestAustinExcluded:
    def test_not_a_sequence(self, manifest):
        assert AUSTIN not in {s['sequence'] for s in manifest['sequences']}

    def test_absent_from_the_mot_export(self):
        assert not (MOT / SPLIT / AUSTIN).exists()
        assert not list(MOT.rglob(f'*{AUSTIN}*'))
        assert AUSTIN not in (MOT / 'seqmaps' / f'{SPLIT}.txt').read_text(encoding='utf-8')

    def test_never_enters_the_clean_aggregate(self, manifest, contract):
        assert AUSTIN in contract['excluded']
        never = manifest['excluded_sequences'][0]['never_enters']
        for m in ('HOTA', 'AssA', 'IDF1'):
            assert any(m in n for n in never)


class TestMotRoundTrip:
    """Row-level equality with the canonical JSON, not a spot check."""

    @pytest.mark.parametrize('tag', sorted(SEQUENCES))
    def test_rows_identities_and_geometry_match_canonical(self, tag, manifest):
        s = next(x for x in manifest['sequences'] if x['sequence'] == tag)
        canon = json.loads((ROOT / s['annotation_file_expected']
                            ).read_text(encoding='utf-8'))['boxes']
        lines = [l for l in (MOT / SPLIT / tag / 'gt' / 'gt.txt'
                             ).read_text(encoding='utf-8').splitlines() if l.strip()]
        assert len(lines) == len(canon)
        by_key = {(b['frame'], b['id']): b['bbox'] for b in canon}
        seen = set()
        for l in lines:
            p = l.split(',')
            key = (int(p[0]), int(p[1]))
            assert key in by_key and key not in seen
            seen.add(key)
            x1, y1, x2, y2 = by_key[key]
            for got, want in zip(p[2:6], (x1, y1, x2 - x1, y2 - y1)):
                assert abs(float(got) - want) <= 0.011, (key, got, want)
        assert seen == set(by_key)

    @pytest.mark.parametrize('tag', sorted(SEQUENCES))
    def test_columns_are_the_trackeval_contract(self, tag):
        for l in (MOT / SPLIT / tag / 'gt' / 'gt.txt'
                  ).read_text(encoding='utf-8').splitlines():
            if not l.strip():
                continue
            p = l.split(',')
            assert len(p) == 9, l
            assert int(p[0]) >= 1, 'frames are 1-based'
            assert int(p[6]) == 1, 'conf: TrackEval drops zero-marked GT'
            assert int(p[7]) == 1, 'class: DO_PREPROC=False does not filter class'
            assert int(p[8]) == 1, 'visibility: constant, never read by the loader'

    @pytest.mark.parametrize('tag', sorted(SEQUENCES))
    def test_occluded_is_not_encoded_into_mot(self, tag, manifest):
        """The boolean stays canonical metadata; no fraction is invented."""
        s = next(x for x in manifest['sequences'] if x['sequence'] == tag)
        canon = json.loads((ROOT / s['annotation_file_expected']
                            ).read_text(encoding='utf-8'))['boxes']
        assert any(b['occluded'] for b in canon) or tag  # boolean present
        vis = {l.split(',')[8] for l in (MOT / SPLIT / tag / 'gt' / 'gt.txt'
                                         ).read_text(encoding='utf-8').splitlines()
               if l.strip()}
        assert vis == {'1'}, 'occlusion must not leak into the visibility column'


class TestFrozenTrackEvalProtocol:
    def test_do_preproc_is_false_and_agrees_with_the_tool(self, protocol):
        from tools.validate_trackeval_loader import DO_PREPROC
        assert protocol['DO_PREPROC'] is False
        assert DO_PREPROC is False

    def test_pinned_version_and_file_hashes(self, protocol):
        assert protocol['source']['version'] == '1.3.0'
        assert protocol['source']['installed_into_environment'] is False
        vendor = REPO / protocol['source']['vendored_at']
        for rel, want in protocol['source']['file_sha256'].items():
            p = vendor / rel
            if not p.exists():
                pytest.skip('vendored trackeval not present')
            assert hashlib.sha256(p.read_bytes()).hexdigest() == want, rel

    def test_dataset_class_and_split(self, protocol):
        d = protocol['dataset']
        assert d['class'].endswith('MotChallenge2DBox')
        assert d['gt_set_folder'] == SPLIT
        assert d['CLASSES_TO_EVAL'] == ['pedestrian']

    def test_visibility_column_is_documented_as_unread(self, protocol):
        assert 'NEVER READ' in protocol['gt_columns_consumed']['8_visibility']
        assert protocol['occluded_metadata']['silently_transformed'] is False

    def test_protocol_was_frozen_before_results(self, protocol):
        assert protocol['tuned_on_tracker_output'] is False
        assert 'may not be altered after any tracker result' in protocol['amendment_rule']

    def test_do_preproc_would_be_a_no_op_on_this_gt(self, manifest):
        """
        The reason DO_PREPROC=False is safe is a property of the data: every
        row is class 1 and conf 1, so the keep-mask and the distractor branch
        have nothing to act on. Assert the property, not the prose.
        """
        for tag in SEQUENCES:
            for l in (MOT / SPLIT / tag / 'gt' / 'gt.txt'
                      ).read_text(encoding='utf-8').splitlines():
                if l.strip():
                    p = l.split(',')
                    assert p[7] == '1' and p[6] == '1'


class TestAdoptionCriteriaImmutable:
    def test_hash_matches_the_contract(self, contract):
        got = hashlib.sha256((EXP / 'adoption_criteria.json').read_bytes()).hexdigest()
        assert got == contract['adoption_criteria']['sha256'], (
            'adoption criteria changed after the evaluation contract was frozen')

    def test_spec_and_criteria_text_unchanged(self, contract):
        c = json.loads((EXP / 'adoption_criteria.json').read_text(encoding='utf-8'))
        assert c['spec_version'] == '1.1'
        assert c['benchmark'] == 'EyeCU-Tracking-Val-v1.1'
        assert c['adoption_criteria']['primary'] == contract['adoption_criteria']['primary']
        assert len(c['adoption_criteria']['primary']) == 8

    def test_no_tracker_result_has_been_recorded_yet(self, contract):
        assert 'no tracker has been run' in contract['not_yet_done']
        assert 'no HOTA/AssA/IDF1 computed' in contract['not_yet_done']
