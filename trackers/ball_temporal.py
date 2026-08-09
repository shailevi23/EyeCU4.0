#!/usr/bin/env python
"""
Bounded, provenance-aware temporal ball recovery.

The reference implementation this project started from interpolates ball
positions with an unbounded `interpolate().bfill().ffill()`, which can invent a
ball before the first observation and hold one indefinitely after the last.
EyeCU's own version was worse: it filled gaps with [0,0,0,0] and emitted a ball
at the frame origin on every missed frame. Both fail the same way -- they make
a confident claim where there is no evidence.

This module refuses to do that. Every output frame carries one of four states,
and "I don't know" is a first-class answer:

    observed                a detection at or above the accept threshold
    recovered_low_conf      a real detection from the low-confidence candidate
                            pool, accepted only because it fell inside a
                            motion-predicted gate
    interpolated_short_gap  no detection at all; a bounded estimate between two
                            anchors that are close in time, with no camera cut
                            between them
    unknown                 insufficient evidence -- no ball is reported

Safety properties, each covered by a test:
  - no history anchor      -> no invented ball
  - gap beyond the limit   -> unknown, never a held box
  - camera cut             -> state reset, gate cleared
  - no backward fill before the first observation
  - interpolation requires anchors on BOTH sides
  - deterministic candidate choice (closest to prediction, then confidence)

=======================================================================
PREDECLARED v1 SETTINGS
These were fixed before the selector was run on the temporal benchmark even
once. They are derived from geometry, not from results:

A 640px-wide frame spans roughly 68m of pitch, so ~9.4 px/m. A hard pass at
20 m/s covers ~4m in the 0.2s between samples at 5 FPS -- about 38px. The base
gate is set to 60px to allow for that plus perspective, and grows by 40px for
each additional missed frame.

    ACCEPT_CONF                 0.25    high-confidence observation
    CANDIDATE_CONF              0.10    rescue pool floor (measured, RESULTS.md)
    MAX_RESCUE_GAP_SECONDS      0.6     3 frames at 5 FPS
    GATE_BASE_PX                60.0    at 640px width, scaled to frame width
    GATE_GROWTH_PX              40.0    per additional missed frame
    MIN_HISTORY_FOR_VELOCITY    2       one anchor implies zero velocity
    MAX_INTERP_GAP_SECONDS      0.4     fills at most 2 missing frames
    CUT_MEAN_ABSDIFF            30.0    on 64x36 grayscale, 0-255

Any change to these after seeing a result is tuning and must be reported as
such.
=======================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# --- predeclared v1 constants (see docstring) ---------------------------
ACCEPT_CONF = 0.25
CANDIDATE_CONF = 0.10
MAX_RESCUE_GAP_SECONDS = 0.6
GATE_BASE_PX = 60.0
GATE_GROWTH_PX = 40.0
MIN_HISTORY_FOR_VELOCITY = 2
MAX_INTERP_GAP_SECONDS = 0.4
CUT_MEAN_ABSDIFF = 30.0

REFERENCE_WIDTH = 640.0

OBSERVED = 'observed'
RECOVERED = 'recovered_low_conf'
INTERPOLATED = 'interpolated_short_gap'
UNKNOWN = 'unknown'


def _centre(bbox):
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


@dataclass
class FrameInput:
    """One sampled frame's ball candidates plus its timing and cut flag."""
    candidates: List[Dict] = field(default_factory=list)  # {'bbox','confidence'}
    timestamp: float = 0.0        # seconds
    dt: float = 0.2               # seconds since the previous sampled frame
    cut: bool = False             # camera cut immediately before this frame


@dataclass
class BallOutput:
    state: str
    bbox: Optional[List[float]] = None
    confidence: Optional[float] = None


def detect_cuts(gray_thumbs: Sequence, threshold: float = CUT_MEAN_ABSDIFF) -> List[bool]:
    """
    Mark a camera cut before frame i when the mean absolute difference against
    frame i-1 exceeds `threshold`. Thumbnails keep this insensitive to the
    ordinary motion of players and the camera pan.
    """
    import numpy as np
    cuts = [False] * len(gray_thumbs)
    for i in range(1, len(gray_thumbs)):
        a = np.asarray(gray_thumbs[i - 1], dtype=float)
        b = np.asarray(gray_thumbs[i], dtype=float)
        cuts[i] = float(np.abs(a - b).mean()) > threshold
    return cuts


class BallTemporalSelector:
    """Offline, two-pass. See module docstring for the predeclared settings."""

    def __init__(self,
                 accept_conf: float = ACCEPT_CONF,
                 candidate_conf: float = CANDIDATE_CONF,
                 max_rescue_gap_seconds: float = MAX_RESCUE_GAP_SECONDS,
                 gate_base_px: float = GATE_BASE_PX,
                 gate_growth_px: float = GATE_GROWTH_PX,
                 min_history_for_velocity: int = MIN_HISTORY_FOR_VELOCITY,
                 max_interp_gap_seconds: float = MAX_INTERP_GAP_SECONDS,
                 frame_width: float = REFERENCE_WIDTH):
        self.accept_conf = accept_conf
        self.candidate_conf = candidate_conf
        self.max_rescue_gap_seconds = max_rescue_gap_seconds
        self.scale = float(frame_width) / REFERENCE_WIDTH
        self.gate_base_px = gate_base_px * self.scale
        self.gate_growth_px = gate_growth_px * self.scale
        self.min_history_for_velocity = min_history_for_velocity
        self.max_interp_gap_seconds = max_interp_gap_seconds

    # -- pass 1 ---------------------------------------------------------
    def _predict(self, history, now: float, dt: float):
        """(predicted centre, gate radius) or (None, 0) when unusable."""
        if not history:
            return None, 0.0
        i_last, c_last, t_last = history[-1]
        gap = now - t_last
        if gap <= 0 or gap > self.max_rescue_gap_seconds:
            return None, 0.0

        vx = vy = 0.0
        if len(history) >= self.min_history_for_velocity:
            _, c_prev, t_prev = history[-2]
            span = t_last - t_prev
            if span > 0:
                vx = (c_last[0] - c_prev[0]) / span
                vy = (c_last[1] - c_prev[1]) / span

        pred = (c_last[0] + vx * gap, c_last[1] + vy * gap)
        missed = max(0.0, gap / dt - 1.0) if dt > 0 else 0.0
        return pred, self.gate_base_px + self.gate_growth_px * missed

    def run(self, frames: Sequence[FrameInput]) -> List[BallOutput]:
        out: List[BallOutput] = []
        history = []          # (index, centre, timestamp) of accepted anchors

        for i, fr in enumerate(frames):
            if fr.cut:
                # A cut invalidates position, velocity and gap entirely. Never
                # carry a trajectory across it.
                history = []

            observed = [c for c in fr.candidates
                        if c['confidence'] >= self.accept_conf]
            if observed:
                best = max(observed, key=lambda c: (c['confidence'], -c['bbox'][0]))
                out.append(BallOutput(OBSERVED, list(best['bbox']),
                                      float(best['confidence'])))
                history.append((i, _centre(best['bbox']), fr.timestamp))
                continue

            pool = [c for c in fr.candidates
                    if self.candidate_conf <= c['confidence'] < self.accept_conf]
            pred, radius = self._predict(history, fr.timestamp, fr.dt)
            if pred is not None and pool:
                scored = []
                for c in pool:
                    cx, cy = _centre(c['bbox'])
                    d = ((cx - pred[0]) ** 2 + (cy - pred[1]) ** 2) ** 0.5
                    if d <= radius:
                        scored.append((d, -c['confidence'], c))
                if scored:
                    scored.sort(key=lambda s: (s[0], s[1]))   # closest, then most confident
                    best = scored[0][2]
                    out.append(BallOutput(RECOVERED, list(best['bbox']),
                                          float(best['confidence'])))
                    history.append((i, _centre(best['bbox']), fr.timestamp))
                    continue

            out.append(BallOutput(UNKNOWN))

        self._interpolate(frames, out)
        return out

    # -- pass 2 ---------------------------------------------------------
    def _interpolate(self, frames, out) -> None:
        """
        Fill runs of UNKNOWN that lie between two anchors close in time, with
        no camera cut in between. Anchors on both sides are required, so
        nothing is ever fabricated before the first or after the last
        observation.
        """
        n = len(out)
        i = 0
        while i < n:
            if out[i].state != UNKNOWN:
                i += 1
                continue
            j = i
            while j < n and out[j].state == UNKNOWN:
                j += 1
            left, right = i - 1, j
            if left >= 0 and right < n and out[left].bbox and out[right].bbox:
                span = frames[right].timestamp - frames[left].timestamp
                gap = span - frames[right].dt          # duration actually filled
                cut_between = any(frames[k].cut for k in range(left + 1, right + 1))
                if not cut_between and 0 < gap <= self.max_interp_gap_seconds:
                    a, b = out[left].bbox, out[right].bbox
                    for k in range(i, j):
                        w = ((frames[k].timestamp - frames[left].timestamp) / span
                             if span > 0 else 0.5)
                        out[k] = BallOutput(
                            INTERPOLATED,
                            [a[m] + (b[m] - a[m]) * w for m in range(4)],
                            None)
            i = j
