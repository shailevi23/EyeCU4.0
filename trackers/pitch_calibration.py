"""
M2 -- minimal, fail-closed pitch calibration.

Replaces the guessed `pixels_per_meter` scale path (trackers/speed_distance.py,
still present as an explicitly-UNCALIBRATED pixel-space diagnostic -- see its
module docstring) with real image->pitch homographies, each valid only on the
specific stable-camera segment it was fit and validated on.

Contract:
    NO VALID CALIBRATION FOR THIS (sequence, frame)  ->  NO metric result.
There is no fallback to a guessed scale anywhere in this module. Every public
function that cannot establish calibration returns None (Python's UNKNOWN)
rather than a number.

Coordinate convention: pitch coordinates in metres, in the LOCAL frame each
calibration artifact declares (`coordinate_convention` field) -- centred on
that segment's centre circle, using an orthogonal local basis fixed by that
segment's own correspondences. Segments are not assumed to share one global
pitch frame (this milestone calibrates isolated segments, not a stitched
whole-pitch model), so positions from two different segments are NOT
comparable to each other, only within one segment.

Player ground point: bottom-centre (ground contact point) of the bbox, i.e.
`((x1+x2)/2, y2)` for an xyxy box. Never the bbox centre.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def ground_point(bbox) -> Tuple[float, float]:
    """Bottom-centre (ground contact point) of an xyxy bbox. See module docstring."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


@dataclass(frozen=True)
class CalibratedSegment:
    sequence: str
    frame_start_1based: int
    frame_end_1based: int
    H_image_to_pitch: np.ndarray
    artifact_sha256: str
    source_path: str

    def covers(self, sequence: str, frame_1based: int) -> bool:
        return (sequence == self.sequence
                and self.frame_start_1based <= frame_1based <= self.frame_end_1based)

    def image_to_pitch(self, image_xy: Tuple[float, float]) -> Tuple[float, float]:
        x, y = image_xy
        p = self.H_image_to_pitch @ np.array([x, y, 1.0])
        if abs(p[2]) < 1e-12:
            raise ValueError('degenerate homography projection (w ~ 0)')
        return float(p[0] / p[2]), float(p[1] / p[2])


class CalibrationStore:
    """Loads frozen calibration artifacts and answers ONLY 'is this covered,
    and if so what is it in pitch metres' -- everything else is UNKNOWN.
    """

    def __init__(self, segments: Optional[List[CalibratedSegment]] = None):
        self._segments = list(segments) if segments else []

    @classmethod
    def load(cls, artifact_paths: List[str]) -> 'CalibrationStore':
        segments = []
        for p in artifact_paths:
            segments.append(load_segment(p))
        return cls(segments)

    def find(self, sequence: str, frame_1based: int) -> Optional[CalibratedSegment]:
        for seg in self._segments:
            if seg.covers(sequence, frame_1based):
                return seg
        return None

    def image_to_pitch(self, sequence: str, frame_1based: int,
                       image_xy: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """UNKNOWN (None) unless this exact (sequence, frame) falls inside a
        validated, still-current calibration segment. No guessed fallback."""
        seg = self.find(sequence, frame_1based)
        if seg is None:
            return None
        return seg.image_to_pitch(image_xy)


def load_segment(artifact_path: str) -> CalibratedSegment:
    raw = Path(artifact_path).read_text(encoding='utf-8')
    d = json.loads(raw)
    recomputed = hashlib.sha256(
        json.dumps({k: v for k, v in d.items() if k != 'calibration_artifact_sha256'},
                  indent=1, sort_keys=True).encode('utf-8')).hexdigest()
    if recomputed != d['calibration_artifact_sha256']:
        raise ValueError(f'calibration artifact hash mismatch for {artifact_path}: '
                         f'file may have been edited after freezing')
    start, end = d['frame_range_1based']
    return CalibratedSegment(
        sequence=d['sequence'],
        frame_start_1based=start,
        frame_end_1based=end,
        H_image_to_pitch=np.array(d['homography_image_to_pitch']),
        artifact_sha256=d['calibration_artifact_sha256'],
        source_path=str(artifact_path),
    )


def short_window_displacement_and_speed(
        store: CalibrationStore, sequence: str,
        frame_a_1based: int, image_xy_a: Tuple[float, float],
        frame_b_1based: int, image_xy_b: Tuple[float, float],
        native_fps: float) -> Optional[Dict]:
    """Metric displacement (m) and speed (m/s) between two frames of ONE track,
    using real dt from native_fps. UNKNOWN (None) unless BOTH frames fall
    inside the SAME calibrated segment -- a homography from one segment must
    never be applied to a frame outside it, and two different segments' local
    pitch frames are not comparable (see module docstring).
    """
    seg_a = store.find(sequence, frame_a_1based)
    seg_b = store.find(sequence, frame_b_1based)
    if seg_a is None or seg_b is None or seg_a.artifact_sha256 != seg_b.artifact_sha256:
        return None
    if native_fps is None or native_fps <= 0:
        return None
    pa = seg_a.image_to_pitch(image_xy_a)
    pb = seg_b.image_to_pitch(image_xy_b)
    dt = (frame_b_1based - frame_a_1based) / native_fps
    if dt <= 0:
        return None
    disp_m = float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))
    return {
        'segment_artifact_sha256': seg_a.artifact_sha256,
        'pitch_a_m': pa, 'pitch_b_m': pb,
        'dt_s': dt, 'displacement_m': disp_m,
        'speed_mps': disp_m / dt,
    }
