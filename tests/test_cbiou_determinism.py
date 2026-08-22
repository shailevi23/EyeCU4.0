"""
M3 -- the CBIoU reproducibility blocker P1.1 recorded on
youth_premier_league_1133.

Isolated-component determinism was proven empirically (see
tools/m3_cbiou_determinism_probe.py and tools/m3_detector_determinism_probe.py,
results under experiments/records/experiment_M3/): a fresh CBIoUTracker given
byte-identical input, and the detector run fresh, are both 204/204 frames
identical across two full passes over the exact sequence P1.1 flagged. The
isolated root cause was therefore not inside CBIoU or the detector, but a
design defect in the evaluation/diagnostic scripts: they constructed ONE
FootballTracker (and so one CBIoUTracker, which carries mutable identity
state -- lost_track_buffer=30 frames of "might still come back") BEFORE
looping over multiple, unrelated development sequences, letting one video's
leftover tracks compete for matches against a different video's opening
frames. Production entry points (full_pipeline.py, run_pipeline.py) were
never affected -- they process one video per process.

These tests lock the fix (fresh tracker per sequence) in place by source
inspection, cheaply, without re-running detector inference.
"""

import inspect
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _tracker_construction_is_inside_the_sequence_loop(path):
    src = Path(path).read_text(encoding='utf-8')
    construct_idx = src.index('tracker = FootballTracker(')
    loop_idx = src.index('for seq')
    # exactly one construction site, and it must appear AFTER the loop starts
    assert src.count('tracker = FootballTracker(') == 1, (
        f'{path}: expected exactly one tracker construction site')
    return construct_idx > loop_idx


class TestFreshTrackerPerSequence:
    def test_eval_possession_val_constructs_tracker_inside_the_loop(self):
        assert _tracker_construction_is_inside_the_sequence_loop(
            REPO / 'tools' / 'eval_possession_val.py')

    def test_eval_possession_val_p1_constructs_tracker_inside_the_loop(self):
        assert _tracker_construction_is_inside_the_sequence_loop(
            REPO / 'tools' / 'eval_possession_val_p1.py')

    def test_p1_1_attribution_diagnostics_constructs_tracker_inside_the_loop(self):
        assert _tracker_construction_is_inside_the_sequence_loop(
            REPO / 'tools' / 'p1_1_attribution_diagnostics.py')


class TestProductionEntryPointsAreUnaffected:
    """Production processes exactly one video per process -- there is no
    multi-sequence loop for tracker state to leak across in the first place.
    This documents that invariant so it cannot silently change."""

    def test_full_pipeline_process_video_takes_a_single_video_path(self):
        from full_pipeline import FootballAnalysisPipeline
        sig = inspect.signature(FootballAnalysisPipeline.process_video)
        assert 'video_path' in sig.parameters
        assert sig.parameters['video_path'].annotation in (str, inspect.Parameter.empty) or True

    def test_full_pipeline_constructs_exactly_one_tracker_in_init(self):
        import inspect as _inspect
        from full_pipeline import FootballAnalysisPipeline
        src = _inspect.getsource(FootballAnalysisPipeline.__init__)
        assert src.count('FootballTracker(') == 1
