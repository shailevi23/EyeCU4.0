"""Shared test fixtures.

Tests must not download or load real YOLO weights: they need to be fast and
deterministic. `FakeDetector` implements the same BaseDetector contract, so it
exercises the real caching, counting and tracking code paths.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackers.detector import BaseDetector  # noqa: E402

FRAME_W, FRAME_H = 640, 360


class FakeDetector(BaseDetector):
    """
    Returns scripted detections. `_predict` is the abstract hook, so every call
    goes through BaseDetector.detect() and is counted exactly like a real one.
    """

    def __init__(self, script, confidence=0.25):
        super().__init__(confidence)
        self._script = script
        self.predict_calls = 0
        self._frame_no = 0

    def _predict(self, image):
        self.predict_calls += 1
        detections = self._script(self._frame_no)
        self._frame_no += 1
        return detections


def box(x, y, w=20, h=40):
    return [float(x), float(y), float(x + w), float(y + h)]


def det(cls, x, y, conf=0.9, **kw):
    return {'bbox': box(x, y, **kw), 'class': cls, 'confidence': conf}


@pytest.fixture
def frames():
    """Cheap synthetic frames; FakeDetector ignores pixel content."""
    def _make(n):
        return [np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8) for _ in range(n)]
    return _make


@pytest.fixture(scope='session')
def sample_video():
    """A real short clip, skipped if the repo has no input videos."""
    for name in ('short.mp4', '08fd33_4.mp4'):
        p = REPO_ROOT / 'input-videos' / name
        if p.exists():
            return str(p)
    pytest.skip('no sample video in input-videos/')
