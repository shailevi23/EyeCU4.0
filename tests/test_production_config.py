"""
M3 -- the exact configuration a normal user gets by just invoking the
production entry points, checked against the measured/frozen CLOSED
configuration (data/tracking_val_v1/manifest.json, experiments/records/).

These are signature/source inspections, not full pipeline runs -- the point
is to catch a default silently drifting away from what was actually
measured, which a heavy end-to-end run would not surface unless someone
happened to print the literal default and read it.
"""

import inspect
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FROZEN_HUMAN_MANIFEST = REPO / 'data' / 'tracking_val_v1' / 'manifest.json'


@pytest.fixture(scope='module')
def frozen_human_detector():
    d = json.loads(FROZEN_HUMAN_MANIFEST.read_text(encoding='utf-8'))
    return d['detector']


class TestHumanDetectorDefault:
    def test_pipeline_constructor_defaults_to_the_frozen_checkpoint(self, frozen_human_detector):
        from full_pipeline import FootballAnalysisPipeline
        sig = inspect.signature(FootballAnalysisPipeline.__init__)
        assert sig.parameters['yolo_model'].default == frozen_human_detector['checkpoint']

    def test_pipeline_constructor_defaults_to_the_frozen_imgsz(self, frozen_human_detector):
        from full_pipeline import FootballAnalysisPipeline
        sig = inspect.signature(FootballAnalysisPipeline.__init__)
        assert sig.parameters['imgsz'].default == frozen_human_detector['imgsz']

    def test_pipeline_constructor_defaults_to_the_frozen_confidence(self, frozen_human_detector):
        from full_pipeline import FootballAnalysisPipeline
        sig = inspect.signature(FootballAnalysisPipeline.__init__)
        assert sig.parameters['confidence'].default == frozen_human_detector['confidence']

    def test_cli_defaults_to_the_frozen_checkpoint(self, frozen_human_detector):
        src = (REPO / 'run_pipeline.py').read_text(encoding='utf-8')
        assert f'default="{frozen_human_detector["checkpoint"]}"' in src

    def test_main_block_config_defaults_to_the_frozen_checkpoint(self, frozen_human_detector):
        src = (REPO / 'full_pipeline.py').read_text(encoding='utf-8')
        assert f"'yolo_model': '{frozen_human_detector['checkpoint']}'" in src

    def test_generic_pretrained_weight_is_not_the_default_anywhere(self):
        """The historical default (a generic, non-production COCO checkpoint)
        must not silently be what a normal invocation gets."""
        from full_pipeline import FootballAnalysisPipeline
        sig = inspect.signature(FootballAnalysisPipeline.__init__)
        assert sig.parameters['yolo_model'].default != 'yolov8x.pt'
        cli_src = (REPO / 'run_pipeline.py').read_text(encoding='utf-8')
        assert 'default="yolov8x.pt"' not in cli_src


class TestBallBranchDefault:
    def test_pipeline_defaults_to_sn3d(self):
        from full_pipeline import FootballAnalysisPipeline
        sig = inspect.signature(FootballAnalysisPipeline.__init__)
        assert sig.parameters['ball_detector_backend'].default == 'sn3d'

    def test_pipeline_defaults_ball_candidate_pool_on(self):
        from full_pipeline import FootballAnalysisPipeline
        sig = inspect.signature(FootballAnalysisPipeline.__init__)
        assert sig.parameters['ball_candidate_pool'].default is True


class TestHumanTrackerDefault:
    def test_pipeline_defaults_to_cbiou(self):
        from full_pipeline import FootballAnalysisPipeline
        sig = inspect.signature(FootballAnalysisPipeline.__init__)
        assert sig.parameters['tracker_backend'].default == 'cbiou'

    def test_football_tracker_defaults_to_cbiou(self):
        from trackers.football_tracker import FootballTracker
        sig = inspect.signature(FootballTracker.__init__)
        assert sig.parameters['tracker_backend'].default == 'cbiou'
