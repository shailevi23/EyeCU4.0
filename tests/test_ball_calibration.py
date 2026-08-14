"""Calibration of the ball proposal generator against human-confirmed adds.

The finding is negative -- 23.0% recall -- and negative findings are exactly the
ones that get quietly softened later, so these tests pin the things that make it
trustworthy: the denominator is human-drawn boxes only, the matching rule is the
same one the review tooling uses, and the numbers in the report reconcile.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))

import kb_ball_calibrate as CAL                                   # noqa: E402
import kb_ball_candidates as CAND                                 # noqa: E402
import kb_ball_pp_sweep_server as PP                              # noqa: E402

pytestmark = pytest.mark.skipif(
    not CAL.REPORT.is_file(), reason='calibration not run')


@pytest.fixture(scope='module')
def rep():
    return json.loads(CAL.REPORT.read_text(encoding='utf-8'))


def test_denominator_is_human_drawn_boxes_only():
    """Existing blue GT is unverified -- one proved to be a player -- so it
    must never enter the denominator."""
    adds = CAL.confirmed_additions()
    ans = PP.answers()
    for a in adds[:50]:
        v = ans[a['IMAGE']]
        assert v['answer'] == 'MISSING_BALL'
        assert a['bbox_xywh'] in [m['bbox_xywh'] for m in v['missing']]


def test_confirmed_additions_come_from_the_effective_fold():
    """A re-answered image contributes only the boxes that currently stand."""
    adds = CAL.confirmed_additions()
    ans = PP.answers()
    total = sum(len(v['missing']) for v in ans.values()
                if v['answer'] == 'MISSING_BALL')
    queue = {r['IMAGE'] for r in
             json.loads(PP.QUEUE.read_text(encoding='utf-8'))['images']}
    in_queue = sum(len(v['missing']) for im, v in ans.items()
                   if v['answer'] == 'MISSING_BALL' and im in queue)
    assert len(adds) == in_queue <= total


def test_calibration_is_labelled_as_proposal_recall_not_detector_recall(rep):
    assert rep['calibration'] == 'PROPOSAL RECALL ON HUMAN_CONFIRMED_ADDITIONS'
    assert 'not gold' in rep['not_detector_recall']
    assert 'player' in rep['not_detector_recall']


def test_matching_rule_is_the_one_the_review_tooling_uses(rep):
    assert 'centre distance' in rep['matching_rule']
    ref = [{'BOX_ID': 'human', 'bbox_xywh': [100.0, 100.0, 6.0, 6.0]}]
    assert CAND.matches_gt([103.0, 101.0, 6.0, 6.0], ref) == 'human'
    assert CAND.matches_gt([300.0, 300.0, 6.0, 6.0], ref) is None


def test_curve_is_monotone_and_reconciles(rep):
    curve = rep['curve']
    assert [r['threshold'] for r in curve] == list(CAL.THRESHOLDS)
    n = rep['frame']['human_confirmed_additions']
    for r in curve:
        assert r['confirmed_covered'] + r['confirmed_missed'] == n
        assert r['confirmed_recall'] == pytest.approx(
            r['confirmed_covered'] / n)
        assert r['proposals_unmatched_to_existing_gt'] <= r['proposals_total']
        buckets = r['recall_by_size']
        assert sum(b['n'] for b in buckets.values()) == n
        assert sum(b['covered'] for b in buckets.values()) == \
            r['confirmed_covered']
    for a, b in zip(curve, curve[1:]):
        assert b['proposals_total'] <= a['proposals_total']
        assert b['confirmed_covered'] <= a['confirmed_covered']


def test_the_target_recall_was_not_reachable(rep):
    """The honest outcome: no threshold reaches 95%, and the report says so
    rather than reporting the best available number as if it were the target."""
    assert rep['tiers']['target_recall'] == 0.95
    assert rep['tiers']['target_reachable'] is False
    assert max(r['confirmed_recall'] for r in rep['curve']) < 0.95


def test_tiering_is_explicitly_not_recommended(rep):
    assert rep['tiers']['recommendation'] == 'DO NOT TIER -- the premise does not hold'
    head = rep['headline_finding']
    assert '23.0%' in head
    assert 'not in the tail' in head


def test_second_detector_probe_is_recorded_as_not_useful(rep):
    p = rep['second_detector_probe']
    assert p['verdict'] == 'NOT USEFUL'
    assert p['recall'] < rep['curve'][0]['confirmed_recall']
    assert 'no information' in p['why']


def test_residual_risk_is_stated(rep):
    r = rep['residual_risk']
    assert 'not negatives' in r
    assert 'residual full-frame QA' in r


def test_calibration_wrote_no_verdict_and_no_annotation():
    """It may only write its own two report files.

    Checked against the AST rather than by substring, because `append(` matches
    ordinary list building and a text search would fail on correct code -- the
    same trap the detector-import check hit.
    """
    import ast
    src = (REPO / 'tools' / 'kb_ball_calibrate.py').read_text(encoding='utf-8')
    for bad in ('DECISIONS', 'ball_candidate_review', '.train('):
        assert bad not in src, f'{bad} does not belong in a calibration tool'
    written = set()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('write_text', 'write_bytes', 'open')):
            v = node.func.value
            written.add(v.id if isinstance(v, ast.Name) else '?')
    assert written <= {'REPORT', 'TIERS'}, f'writes to {written}'
