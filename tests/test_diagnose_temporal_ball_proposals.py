"""Guards for the B2 zero-proposal diagnostic.

The diagnostic deliberately looks below the production candidate threshold. That
is only safe if the below-threshold floor cannot escape into the frozen path, so
these tests are mostly about containment rather than about the statistic itself.
"""


import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / 'tools' / 'diagnose_temporal_ball_proposals.py'
FROZEN_EVAL = REPO / 'tools' / 'eval_temporal_val.py'

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools'))


def test_frozen_eval_matches_committed_content():
    """tools/eval_temporal_val.py must be byte-identical to HEAD.

    The diagnostic exists precisely so the frozen evaluator does not have to
    grow a below-threshold mode. If this fails, that separation was broken.
    """
    out = subprocess.run(
        ['git', '-C', str(REPO), 'diff', '--quiet', 'HEAD', '--',
         'tools/eval_temporal_val.py'])
    assert out.returncode == 0, 'tools/eval_temporal_val.py differs from HEAD'


def test_production_thresholds_unchanged():
    from trackers.detector import (BALL_ACCEPT_CONF, BALL_CANDIDATE_CONF,
                                   BALL_DEDUPE_IOU)
    assert BALL_CANDIDATE_CONF == 0.10
    assert BALL_ACCEPT_CONF == 0.25
    assert BALL_DEDUPE_IOU == 0.70


def test_diagnostic_does_not_import_the_selector():
    """The diagnostic reports raw proposals; involving the selector would make
    a recovered or interpolated box look like detector evidence."""
    src = TOOL.read_text(encoding='utf-8')
    assert 'BallTemporalSelector' not in src
    assert 'ball_temporal' not in src


def test_match_iou_and_floor_defaults():
    src = TOOL.read_text(encoding='utf-8')
    assert 'MATCH_IOU = 0.5' in src
    assert "default=0.01" in src


@pytest.mark.parametrize('floor', ['0.0', '0.25', '0.5', '-0.01'])
def test_proposal_floor_must_stay_below_production(floor, tmp_path):
    """The floor is a below-production diagnostic knob, not a threshold dial.

    Anything at or above BALL_CANDIDATE_CONF would silently turn this into a
    tuning surface, so it is refused.
    """
    r = subprocess.run(
        [sys.executable, str(TOOL), '--model', 'best_A_960.pt',
         '--proposal-floor', floor, '--out', str(tmp_path / 'x.json')],
        capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode != 0
    assert 'proposal-floor' in (r.stdout + r.stderr)


def test_sealed_test_is_unreachable():
    """No split argument exists, and the only data root is the VAL_ONLY benchmark."""
    src = TOOL.read_text(encoding='utf-8')
    assert '--split' not in src
    assert "TV = Path('data/temporal_val')" in src
    assert 'dataset_baseline' not in src
    split_marker = (REPO / 'data' / 'temporal_val' / 'SPLIT').read_text(encoding='utf-8')
    assert split_marker.strip().startswith('VAL_ONLY')


def test_best_iou_matching_behaviour():
    """A GT ball counts as proposed only at IoU >= 0.5 against some raw box."""
    sys.path.insert(0, str(REPO / 'tools'))
    from diagnose_temporal_ball_proposals import best_iou

    gt = np.array([100.0, 100.0, 110.0, 110.0])          # 10x10
    assert best_iou(gt, []) == 0.0
    assert best_iou(gt, [{'bbox': [100, 100, 110, 110]}]) == pytest.approx(1.0)
    # half-overlap -> IoU 1/3, below the 0.5 rule
    assert best_iou(gt, [{'bbox': [105, 100, 115, 110]}]) == pytest.approx(1 / 3)
    # disjoint
    assert best_iou(gt, [{'bbox': [200, 200, 210, 210]}]) == 0.0
    # best of several wins
    assert best_iou(gt, [{'bbox': [200, 200, 210, 210]},
                         {'bbox': [100, 100, 110, 110]}]) == pytest.approx(1.0)


def test_gt_reader_matches_frozen_eval_convention():
    """Same single-class YOLO reader as the frozen evaluator, or the two tools
    would disagree about which ball is which."""
    sys.path.insert(0, str(REPO / 'tools'))
    from diagnose_temporal_ball_proposals import load_gt_ball

    p = Path(__file__).parent / '_tmp_gt.txt'
    p.write_text('0 0.5 0.5 0.1 0.2\n', encoding='utf-8')
    try:
        box = load_gt_ball(p, 1000, 500)
        assert box.shape == (1, 4)
        assert box[0] == pytest.approx([450.0, 200.0, 550.0, 300.0])
        assert load_gt_ball(Path('does-not-exist.txt'), 100, 100).shape == (0, 4)
    finally:
        p.unlink()
