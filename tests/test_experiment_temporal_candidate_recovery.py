"""Containment guards for T1.

T1 reaches below the production candidate threshold. That is only defensible if
the 0.01 floor cannot leak into production, the frozen selector and evaluator are
untouched, and the association gate stays the frozen number.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / 'tools' / 'experiment_temporal_candidate_recovery.py'

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools'))


@pytest.mark.parametrize('rel', [
    'trackers/ball_temporal.py',
    'tools/eval_temporal_val.py',
    'experiments/records/experiment_B2/experiment_B2_spec.md',
])
def test_frozen_files_match_head(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'diff', '--quiet', 'HEAD', '--', rel])
    assert r.returncode == 0, f'{rel} differs from HEAD'


# trackers/detector.py was dropped from the byte-identity list above when T1 was
# formally closed (experiments/records/experiment_T1/experiment_T1_record.md:
# STATUS CLOSED, VERDICT FAIL, the 0.01 floor NOT adopted). Freezing that file
# forever was never what T1 required -- it required that T1's experimental floor
# stay out of production -- and D2 subsequently made an authorised architecture
# change there (the selectable SN3D ball branch). The assertions below replace
# the byte check with the invariant it was standing in for, so an authorised
# detector change passes while the T1 floor reaching production still fails.
def test_t1_floor_is_not_a_production_constant():
    """The experimental 0.01 floor must never become a production threshold."""
    from trackers import detector

    for name in ('BALL_CANDIDATE_CONF', 'BALL_ACCEPT_CONF', 'BALL_DEDUPE_IOU',
                 'HUMAN_CANDIDATE_CONF', 'HUMAN_ACCEPT_CONF'):
        value = getattr(detector, name)
        assert value != 0.01, (
            f'{name} == 0.01: the T1 experimental candidate floor has leaked '
            'into production. T1 FAILED (hallucinated empty frames 2/27 -> 6/27) '
            'and the floor was not adopted.')


def test_production_detector_never_emits_below_the_frozen_floor():
    """
    Behavioural, not textual: run the real LocalDetector._predict over a stubbed
    model that offers balls at 0.01, 0.05 and 0.30. The 0.01 and 0.05 boxes are
    exactly what T1 wanted to rescue, and production must drop them regardless of
    how detector.py is refactored.
    """
    from trackers.detector import BALL_CANDIDATE_CONF, LocalDetector

    class _Box:
        def __init__(self, conf):
            self.cls = 0
            self.conf = conf
            self.xyxy = [type('T', (), {'tolist': lambda s: [10.0, 10.0, 16.0, 16.0]})()]

    class _Result:
        boxes = [_Box(0.01), _Box(0.05), _Box(0.30)]

    class _Model:
        names = {0: 'ball'}

        def predict(self, image, **kwargs):
            assert kwargs['conf'] >= BALL_CANDIDATE_CONF, (
                'production asked the model for boxes below the frozen candidate '
                f"floor: conf={kwargs['conf']}")
            return [_Result()]

    det = LocalDetector.__new__(LocalDetector)
    det.confidence = 0.25
    det.iou = 0.5
    det.imgsz = 960
    det.device = None
    det.ball_candidate_pool = True
    det.human_candidate_pool = False
    det.model = _Model()
    det._class_map = {0: 'ball'}

    emitted = [d['confidence'] for d in det._predict(None)]
    assert emitted == [0.30], (
        f'production emitted {emitted}; anything below {BALL_CANDIDATE_CONF} means '
        'the T1 experimental floor has reached production')


def test_detector_module_still_exposes_the_frozen_contract():
    """
    detector.py may evolve (D2 added the SN3D ball branch); the pieces T1 leaned
    on must survive that evolution.
    """
    from trackers.detector import (BALL_ACCEPT_CONF, BALL_CANDIDATE_CONF,
                                   BALL_DEDUPE_IOU, suppress_ball_duplicates)

    assert (BALL_CANDIDATE_CONF, BALL_ACCEPT_CONF, BALL_DEDUPE_IOU) == (0.10, 0.25, 0.70)
    # dedupe still behaves at the frozen IoU: a clear duplicate is merged and the
    # more confident box survives, while a pair just under 0.70 stays separate.
    a = {'bbox': [0, 0, 10, 10], 'class': 'ball', 'confidence': 0.9}
    dup = {'bbox': [0.5, 0.5, 10.5, 10.5], 'class': 'ball', 'confidence': 0.4}  # IoU 0.822
    kept = suppress_ball_duplicates([a, dup], BALL_DEDUPE_IOU)
    assert len(kept) == 1 and kept[0]['confidence'] == 0.9

    near = {'bbox': [1, 1, 11, 11], 'class': 'ball', 'confidence': 0.4}         # IoU 0.681
    assert len(suppress_ball_duplicates([a, near], BALL_DEDUPE_IOU)) == 2


def test_production_thresholds_unchanged():
    from trackers.detector import (BALL_ACCEPT_CONF, BALL_CANDIDATE_CONF,
                                   BALL_DEDUPE_IOU)
    assert BALL_CANDIDATE_CONF == 0.10
    assert BALL_ACCEPT_CONF == 0.25
    assert BALL_DEDUPE_IOU == 0.70


def test_frozen_gate_constants_unchanged():
    from trackers.ball_temporal import (GATE_BASE_PX, GATE_GROWTH_PX,
                                        MAX_INTERP_GAP_SECONDS,
                                        MAX_RESCUE_GAP_SECONDS, REFERENCE_WIDTH)
    assert GATE_BASE_PX == 60.0
    assert GATE_GROWTH_PX == 40.0
    assert REFERENCE_WIDTH == 640.0
    assert MAX_RESCUE_GAP_SECONDS == 0.6
    assert MAX_INTERP_GAP_SECONDS == 0.4


def test_t1_floor_is_confined_to_the_experiment():
    """0.01 appears only as the experiment's own constant, never written back."""
    src = TOOL.read_text(encoding='utf-8')
    assert 'T1_FLOOR = 0.01' in src
    for forbidden in ('BALL_CANDIDATE_CONF =', 'BALL_ACCEPT_CONF =', 'GATE_BASE_PX ='):
        assert forbidden not in src, f'T1 must not reassign {forbidden}'


def test_t1_reuses_rather_than_reimplements_interpolation():
    """Clause 4D was dropped: pass 2 must be the frozen method, not a copy."""
    src = TOOL.read_text(encoding='utf-8')
    assert '_interpolate(' in src
    assert 'def _interpolate' not in src, 'T1 must not define its own interpolation'


def test_sealed_test_is_unreachable():
    src = TOOL.read_text(encoding='utf-8')
    assert '--split' not in src
    assert "TV = Path('data/temporal_val')" in src
    assert 'dataset_baseline' not in src
    assert (REPO / 'data' / 'temporal_val' / 'SPLIT').read_text(
        encoding='utf-8').strip().startswith('VAL_ONLY')


def test_pass1_reproduces_the_frozen_selector_when_unmodified():
    """With the frozen floor and the t+1 anchor off, T1's pass 1 must be
    identical to BallTemporalSelector.run's. This is what makes the CONTROL and
    T1 arms comparable at all."""
    from trackers.ball_temporal import (BallTemporalSelector, FrameInput,
                                        OBSERVED, RECOVERED, UNKNOWN)
    from experiment_temporal_candidate_recovery import t1_pass1

    def c(x, y, conf):
        return {'bbox': [x, y, x + 8, y + 8], 'confidence': conf, 'class': 'ball'}

    frames = [
        FrameInput(candidates=[c(100, 100, 0.9)], timestamp=0.0, dt=0.2, cut=False),
        FrameInput(candidates=[c(110, 100, 0.9)], timestamp=0.2, dt=0.2, cut=False),
        FrameInput(candidates=[c(121, 100, 0.15)], timestamp=0.4, dt=0.2, cut=False),
        FrameInput(candidates=[], timestamp=0.6, dt=0.2, cut=False),
        FrameInput(candidates=[c(140, 100, 0.9)], timestamp=0.8, dt=0.2, cut=False),
    ]
    sel = BallTemporalSelector(frame_width=640.0)
    frozen = sel.run(frames)

    mine, _ = t1_pass1(frames, sel, use_next_anchor=False)
    sel._interpolate(frames, mine)

    assert [o.state for o in mine] == [o.state for o in frozen]
    assert [o.bbox for o in mine] == [o.bbox for o in frozen]
    # and the fixture actually exercises the paths, not just UNKNOWNs
    assert frozen[2].state == RECOVERED
    assert frozen[0].state == OBSERVED


def test_widening_the_floor_can_only_add_candidates():
    """A 0.01 pool must never lose a ball the 0.10 pool would have rescued."""
    from trackers.ball_temporal import BallTemporalSelector, FrameInput, RECOVERED
    from experiment_temporal_candidate_recovery import t1_pass1

    def c(x, conf):
        return {'bbox': [x, 100, x + 8, 108], 'confidence': conf, 'class': 'ball'}

    frames = [
        FrameInput(candidates=[c(100, 0.9)], timestamp=0.0, dt=0.2, cut=False),
        FrameInput(candidates=[c(110, 0.9)], timestamp=0.2, dt=0.2, cut=False),
        FrameInput(candidates=[c(121, 0.04)], timestamp=0.4, dt=0.2, cut=False),
    ]
    at_010, _ = t1_pass1(frames, BallTemporalSelector(frame_width=640.0,
                                                      candidate_conf=0.10),
                         use_next_anchor=False)
    at_001, _ = t1_pass1(frames, BallTemporalSelector(frame_width=640.0,
                                                      candidate_conf=0.01),
                         use_next_anchor=False)
    assert at_010[2].state != RECOVERED     # 0.04 is below the frozen floor
    assert at_001[2].state == RECOVERED     # T1 reaches it
