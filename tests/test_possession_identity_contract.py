"""
The P1 identity correspondence contract, held in place by test.

These tests exist because the contract they guard was written *after* P0 was
observed. That is legitimate only if the contract cannot then drift again to
suit a later number, so the declared constants, the four required behaviours,
and the specific P0 pathology that motivated the change are all asserted here.

Nothing in this file uses possession correctness. The contract must never be
tuned on the outcome it is used to measure, so the outcome does not appear.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

from tools.possession_identity import (DOMINANCE_MARGIN, DOMINANCE_RATIO,
                                       IOU_VOTE, MIN_SUPPORT_FRAMES,
                                       MIN_SUPPORT_RATE, UNMAPPABLE,
                                       build_correspondence, iou_1_to_n,
                                       tracks_from_frames)

REPO = Path(__file__).resolve().parents[1]
P1 = REPO / 'experiments' / 'records' / 'experiment_P1'
PV = REPO / 'data' / 'possession_val_v1'

FROZEN_SHA = '9738780c1cad3311c9b137fad23ff7d1076fcf719591d342222045d6f47c5025'
CONTRACT_SHA = 'f8b003d4590a4f16e1d6021256dfb75b7cd8ae7ae439e32ba36f3ec493248ee9'
PROTOCOL_SHA = 'bf9d25e10546abf4ee09a2abe5be888ff7866b5336bcc3739dd990b20517c916'

LABELS = ('PLAYER', 'NO_CONTROL', 'NO_BALL', 'AMBIGUOUS')


def box(x, y, w=20, h=40):
    return [x, y, x + w, y + h]


def gt_frames(spec):
    """{frame: [(gt_id, bbox), ...]}"""
    return {f: [(g, b) for g, b in v] for f, v in spec.items()}


# --------------------------------------------------------------------------
# declared constants
# --------------------------------------------------------------------------

class TestDeclaredConstants:
    def test_constants_are_exactly_the_frozen_contract_values(self):
        assert IOU_VOTE == 0.30
        assert MIN_SUPPORT_FRAMES == 2
        assert MIN_SUPPORT_RATE == 0.50
        assert DOMINANCE_RATIO == 2.0
        assert DOMINANCE_MARGIN == 2

    def test_contract_documents_are_frozen_at_their_recorded_hashes(self):
        for name, want in (('IDENTITY_CORRESPONDENCE_CONTRACT', CONTRACT_SHA),
                           ('ANNOTATION_PROTOCOL_V2', PROTOCOL_SHA)):
            md = P1 / f'{name}.md'
            assert md.exists(), md
            got = hashlib.sha256(md.read_bytes()).hexdigest()
            assert got == want, f'{name} changed after freeze'
            assert (P1 / f'{name}.sha256').read_text(encoding='utf-8').strip() == want

    def test_no_control_definition_contains_no_algorithmic_term(self):
        """The whole point of protocol v2: GT must not restate the system rule."""
        text = (P1 / 'ANNOTATION_PROTOCOL_V2.md').read_text(encoding='utf-8')
        section = text[text.index('### `NO_CONTROL`'):text.index('### `NO_BALL`')]
        # The section ends with a paragraph that explicitly FORBIDS these terms,
        # so it necessarily names them. Only the definition prose above it is
        # scanned -- that is the part an annotator applies.
        body = section[:section.index('The definition MUST NOT')].lower()
        flat = re.sub(r'\s+', ' ', body)
        for banned in ('70 px', '70px', 'max_distance', 'playerballassigner',
                       'pixel', 'distance', 'detector', 'iou'):
            assert banned not in body, f'algorithmic term in definition: {banned}'
        assert 'no player has clear control of it' in flat


# --------------------------------------------------------------------------
# the four required behaviours
# --------------------------------------------------------------------------

class TestRequiredBehaviours:
    def test_stable_track_maps_to_the_correct_gt_identity(self):
        pred = {5: {f: box(100 + f, 50) for f in range(1, 11)}}
        gt = gt_frames({f: [(7, box(100 + f, 50)), (9, box(400, 50))]
                        for f in range(1, 11)})
        rec = build_correspondence(pred, gt)[5]
        assert rec['gt_id'] == 7
        assert rec['reason'] == 'mapped'
        assert rec['purity'] == 1.0

    def test_two_disjoint_fragments_of_one_player_both_map_to_that_player(self):
        """Fragment tolerance. Requirement C: many-to-one is legal."""
        pred = {1: {f: box(100 + f, 50) for f in range(1, 6)},
                2: {f: box(100 + f, 50) for f in range(20, 26)}}
        gt = gt_frames({f: [(7, box(100 + f, 50)), (9, box(400, 50))]
                        for f in list(range(1, 6)) + list(range(20, 26))})
        got = build_correspondence(pred, gt)
        assert got[1]['gt_id'] == 7 and got[1]['reason'] == 'mapped'
        assert got[2]['gt_id'] == 7 and got[2]['reason'] == 'mapped'

    def test_ambiguous_correspondence_is_unmappable_not_forced(self):
        """A track straddling two players near-evenly must refuse to decide."""
        pred = {1: {}}
        gt = {}
        for f in range(1, 11):
            # predicted box sits on player 7 on odd frames, on 9 on even ones
            on7 = (f % 2 == 1)
            pred[1][f] = box(100, 50) if on7 else box(300, 50)
            gt[f] = [(7, box(100, 50)), (9, box(300, 50))]
        rec = build_correspondence(pred, gt_frames(gt))[1]
        assert rec['gt_id'] == UNMAPPABLE
        assert rec['reason'] == 'not_dominant'
        assert rec['top_votes'] == 5 and rec['runner_votes'] == 5

    def test_simultaneous_conflicting_tracks_do_not_both_claim_one_identity(self):
        """Requirement E: co-existing claimants, no silent shared identity."""
        pred = {1: {f: box(100, 50) for f in range(1, 11)},           # 10 votes
                2: {**{f: box(108, 50) for f in range(1, 7)},          # 6 votes
                    **{f: box(600, 300) for f in range(7, 11)}}}
        gt = gt_frames({f: [(7, box(100, 50))] for f in range(1, 11)})
        got = build_correspondence(pred, gt)
        mapped = [t for t, r in got.items() if r['gt_id'] == 7]
        assert len(mapped) <= 1, 'two co-existing tracks claimed the same player'
        assert mapped == [1], 'the stronger claimant should keep the identity'
        assert got[2]['gt_id'] == UNMAPPABLE
        assert got[2]['reason'] in ('conflict_lost', 'not_dominant',
                                    'low_support_rate', 'no_support')

    def test_tied_simultaneous_claimants_both_become_unmappable(self):
        pred = {1: {f: box(100, 50) for f in range(1, 11)},
                2: {f: box(100, 50) for f in range(1, 11)}}          # identical
        gt = gt_frames({f: [(7, box(100, 50))] for f in range(1, 11)})
        got = build_correspondence(pred, gt)
        assert got[1]['gt_id'] == UNMAPPABLE and got[2]['gt_id'] == UNMAPPABLE
        assert got[1]['reason'] == 'conflict_tied'

    def test_a_demoted_track_is_not_reassigned_to_its_runner_up(self):
        """Losing a contest is not evidence for a different identity."""
        pred = {1: {f: box(100, 50) for f in range(1, 11)},
                2: {**{f: box(105, 50) for f in range(1, 9)},
                    9: box(300, 50), 10: box(300, 50)}}
        gt = gt_frames({f: [(7, box(100, 50)), (9, box(300, 50))]
                        for f in range(1, 11)})
        got = build_correspondence(pred, gt)
        assert got[2]['gt_id'] == UNMAPPABLE
        assert got[2]['gt_id'] != 9

    def test_single_frame_fragment_is_unmappable_by_construction(self):
        pred = {1: {4: box(100, 50)}}
        gt = gt_frames({4: [(7, box(100, 50))]})
        rec = build_correspondence(pred, gt)[1]
        assert rec['gt_id'] == UNMAPPABLE and rec['reason'] == 'too_short'

    def test_track_overlapping_nobody_is_unmappable(self):
        pred = {1: {f: box(600, 300) for f in range(1, 11)}}
        gt = gt_frames({f: [(7, box(100, 50))] for f in range(1, 11)})
        rec = build_correspondence(pred, gt)[1]
        assert rec['gt_id'] == UNMAPPABLE
        assert rec['reason'] in ('no_support', 'low_support_rate')

    def test_decision_is_deterministic_under_reordering(self):
        pred = {1: {f: box(100, 50) for f in range(1, 11)}}
        a = gt_frames({f: [(7, box(100, 50)), (9, box(400, 50))]
                       for f in range(1, 11)})
        b = gt_frames({f: [(9, box(400, 50)), (7, box(100, 50))]
                       for f in range(1, 11)})
        assert (build_correspondence(pred, a)[1]['gt_id']
                == build_correspondence(pred, b)[1]['gt_id'] == 7)


# --------------------------------------------------------------------------
# the P0 pathology, reproduced
# --------------------------------------------------------------------------

class TestP0FlickerPathology:
    """A stable identity that dips below single-frame IoU 0.50 on ONE frame.

    Under the P0 mapper the identity vanishes on that frame and a labelled
    PLAYER frame scores as an error. The new contract must not flicker.
    """

    @staticmethod
    def _scenario():
        gt, pred = {}, {}
        for f in range(195, 205):
            gt[f] = [(7, box(100, 50)), (18, box(420, 50))]
            # frames 199 and 200: the predicted box shifts enough that
            # single-frame IoU with GT 7 falls under 0.50 but stays over 0.30
            pred[f] = box(108, 50) if f in (199, 200) else box(100, 50)
        return pred, gt

    def test_old_single_frame_mapper_flickers(self):
        pred, gt = self._scenario()
        old = {}
        for f, b in pred.items():
            ious = iou_1_to_n(b, [x for _, x in gt[f]])
            j = int(ious.argmax())
            old[f] = gt[f][j][0] if float(ious[j]) >= 0.50 else None
        assert old[198] == 7 and old[201] == 7, 'scenario must be stable either side'
        assert old[199] is None and old[200] is None, 'scenario must reproduce the dip'

    def test_new_contract_does_not_flicker_on_one_frame(self):
        pred, gt = self._scenario()
        rec = build_correspondence({6: pred}, gt_frames(gt))[6]
        assert rec['gt_id'] == 7, 'sequence-level evidence must survive the dip'
        assert rec['reason'] == 'mapped'
        # the dipped frames still vote for 7 at IOU_VOTE, so support is complete
        assert rec['support'] == 10 and rec['top_votes'] == 10

    def test_new_contract_survives_even_if_the_dip_loses_the_vote(self):
        """Harder version: the two frames overlap nobody at all."""
        pred, gt = self._scenario()
        pred[199] = box(600, 300)
        pred[200] = box(600, 300)
        rec = build_correspondence({6: pred}, gt_frames(gt))[6]
        assert rec['gt_id'] == 7 and rec['reason'] == 'mapped'
        assert rec['top_votes'] == 8

    def test_single_frame_perturbation_cannot_move_a_clear_decision(self):
        pred, gt = self._scenario()
        base = build_correspondence({6: pred}, gt_frames(gt))[6]['gt_id']
        for f in list(pred):
            probe = dict(pred)
            probe[f] = box(600, 300)                 # kill exactly one frame
            got = build_correspondence({6: probe}, gt_frames(gt))[6]['gt_id']
            assert got == base, f'flicker reintroduced by frame {f}'


# --------------------------------------------------------------------------
# frozen population and annotation completeness
# --------------------------------------------------------------------------

class TestFrozenPopulation:
    def test_population_file_still_hashes_to_the_frozen_value(self):
        p = PV / 'POSSESSION_VAL_V1_FROZEN.json'
        assert hashlib.sha256(p.read_bytes()).hexdigest() == FROZEN_SHA
        assert (PV / 'POSSESSION_VAL_V1_FROZEN.sha256'
                ).read_text(encoding='utf-8').strip() == FROZEN_SHA

    def test_population_is_still_exactly_six_windows_of_ten_frames(self):
        fr = json.loads((PV / 'POSSESSION_VAL_V1_FROZEN.json')
                        .read_text(encoding='utf-8'))
        assert fr['n_windows'] == 6 and fr['n_frames'] == 60
        assert len(fr['windows']) == 6
        for w in fr['windows']:
            assert len(w['frames_1based']) == 10

    def test_annotations_cover_the_frozen_population_exactly(self):
        fr = json.loads((PV / 'POSSESSION_VAL_V1_FROZEN.json')
                        .read_text(encoding='utf-8'))
        ann = json.loads((PV / 'POSSESSION_VAL_V1_ANNOTATIONS.json')
                         .read_text(encoding='utf-8'))
        want = {f"{w['window_id']}:{f}" for w in fr['windows']
                for f in w['frames_1based']}
        got = {r['frame_id'] for r in ann['annotations']}
        assert got == want, 'annotation population drifted from the frozen list'
        assert ann['frozen_list_sha256'] == FROZEN_SHA

    def test_no_frame_remains_unlabelled(self):
        ann = json.loads((PV / 'POSSESSION_VAL_V1_ANNOTATIONS.json')
                         .read_text(encoding='utf-8'))
        unl = [r['frame_id'] for r in ann['annotations']
               if r['label_state'] == 'UNLABELLED']
        assert unl == [], f'{len(unl)} frames still unlabelled'

    def test_label_states_are_legal_and_player_frames_name_an_identity(self):
        ann = json.loads((PV / 'POSSESSION_VAL_V1_ANNOTATIONS.json')
                         .read_text(encoding='utf-8'))
        for r in ann['annotations']:
            assert r['label_state'] in LABELS, r
            if r['label_state'] == 'PLAYER':
                assert isinstance(r['gt_player_track_id'], int), r
            else:
                assert r['gt_player_track_id'] is None, r

    def test_player_labels_name_an_identity_that_exists_in_gt_on_that_frame(self):
        """A label pointing at a non-existent GT id is unscoreable."""
        mot = REPO / 'data' / 'tracking_val_gt' / 'mot' / 'EyeCU-val'
        if not mot.exists():
            pytest.skip('identity GT package not present')
        ann = json.loads((PV / 'POSSESSION_VAL_V1_ANNOTATIONS.json')
                         .read_text(encoding='utf-8'))
        cache = {}
        for r in ann['annotations']:
            if r['label_state'] != 'PLAYER':
                continue
            seq = r['sequence']
            if seq not in cache:
                ids = {}
                for line in (mot / seq / 'gt' / 'gt.txt').read_text(
                        encoding='utf-8').splitlines():
                    q = line.split(',')
                    if len(q) >= 6:
                        ids.setdefault(int(q[0]), set()).add(int(q[1]))
                cache[seq] = ids
            present = cache[seq].get(r['frame_1based'], set())
            assert r['gt_player_track_id'] in present, r['frame_id']

    def test_annotation_hash_matches_its_recorded_value(self):
        rec = PV / 'POSSESSION_VAL_V1_ANNOTATIONS.sha256'
        if not rec.exists():
            pytest.skip('annotation not yet frozen')
        got = hashlib.sha256((PV / 'POSSESSION_VAL_V1_ANNOTATIONS.json')
                             .read_bytes()).hexdigest()
        assert rec.read_text(encoding='utf-8').strip() == got


class TestHelpers:
    def test_tracks_from_frames_keys_by_absolute_frame_id(self):
        frames = [{1: {'bbox': box(0, 0)}}, {}, {1: {'bbox': box(5, 0)},
                                                 2: {'bbox': box(90, 0)}}]
        got = tracks_from_frames(frames, [10, 11, 12])
        assert sorted(got) == [1, 2]
        assert sorted(got[1]) == [10, 12]
        assert sorted(got[2]) == [12]
