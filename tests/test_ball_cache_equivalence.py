"""A cached run must give BallTemporalSelector the same evidence as a fresh one.

This is a regression suite for a shipped bug. The cache stored only `tracks`,
so a cache hit left `ball_candidates` unset and the selector fell back to
reconstructing candidates from `tracks['ball']` -- which contains boxes the
tracker COPIED FORWARD during a gap. Those copies came back as
state='observed' carrying the original detector's confidence, so a held
position became indistinguishable from a measurement, and cached and fresh runs
of the same video disagreed on 4 of 12 frames.

The three things kept apart throughout:

    A  detector candidate   -- raw detection, any confidence
    B  tracker observation  -- tracks['ball'], INCLUDING held copies
    C  selector output      -- the final measured/recovered/interpolated/unknown

Only A may reach the selector.
"""

import pickle

import numpy as np
import pytest

from trackers.ball_temporal import (INTERPOLATED, OBSERVED, RECOVERED, UNKNOWN)
from trackers.football_tracker import FootballTracker

W, H = 640, 360
FPS = 5.0

# One scenario exercising every path at once:
#   0-2   plain observations
#   3-5   nothing at all -> long gap, and the tracker HOLDS a box here
#   6     observation again
#   7     ONLY a low-confidence candidate -> rescue territory
#   8     nothing -> short gap between two anchors
#   9-11  observations
GAP = (3, 4, 5)
RESCUE = 7
SHORT_GAP = 8
N_FRAMES = 12


class Stub:
    """Deterministic detector. No weights, no GPU."""

    def __init__(self):
        self.i = 0

    def detect(self, frame, idx=None):
        i, self.i = self.i, self.i + 1
        x = 100 + i * 6
        if i in GAP or i == SHORT_GAP:
            return []
        if i == RESCUE:
            return [{'class': 'ball', 'bbox': [x, 100, x + 10, 110],
                     'confidence': 0.15, 'state': 'candidate_low_conf'}]
        return [{'class': 'ball', 'bbox': [x, 100, x + 10, 110],
                 'confidence': 0.9}]


def frames():
    return [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(N_FRAMES)]


def summarise(out):
    """(state, bbox, confidence) per frame -- the full downstream contract."""
    rows = []
    for f in out:
        if not f:
            rows.append((UNKNOWN, None, None))
        else:
            e = next(iter(f.values()))
            rows.append((e['state'],
                         tuple(round(v, 4) for v in e['bbox']),
                         e.get('confidence')))
    return rows


def run(tracker, cache, read):
    tracks = tracker.get_object_tracks(frames(), read_from_cache=read,
                                       cache_path=str(cache))
    return summarise(tracker.apply_ball_temporal_selection(
        tracks['ball'], candidates=tracker.ball_candidates,
        fps=FPS, frame_width=W)), tracks


def tracker(tmp_path):
    return FootballTracker(detector=Stub(), persist_cache=True,
                           cache_dir=str(tmp_path), ball_candidate_pool=True)


# ------------------------------------------------------- the invariant


@pytest.fixture
def both(tmp_path):
    cache = tmp_path / 'tracks.pkl'
    fresh, tr_fresh = run(tracker(tmp_path), cache, read=False)
    assert cache.exists(), 'the fresh run must have written a cache'
    cached, tr_cached = run(tracker(tmp_path), cache, read=True)
    return fresh, cached, tr_fresh, tr_cached, cache


def test_fresh_and_cached_final_ball_output_are_identical(both):
    fresh, cached, *_ = both
    assert fresh == cached, (
        'cached run disagreed with fresh run:\n' +
        '\n'.join(f'  frame {i}: fresh={a} cached={b}'
                  for i, (a, b) in enumerate(zip(fresh, cached)) if a != b))


def test_the_scenario_actually_exercises_every_state(both):
    """A green equivalence test over an all-UNKNOWN clip would prove nothing."""
    fresh, *_ = both
    states = {s for s, _, _ in fresh}
    assert OBSERVED in states
    assert RECOVERED in states, 'the rescue-only frame must be recovered'
    assert INTERPOLATED in states, 'the short gap must be interpolated'
    assert UNKNOWN in states, 'the long gap must stay unknown'


@pytest.mark.parametrize('field,idx', [('state', 0), ('bbox', 1),
                                       ('confidence', 2)])
def test_each_field_matches_frame_by_frame(both, field, idx):
    fresh, cached, *_ = both
    assert [r[idx] for r in fresh] == [r[idx] for r in cached]


def test_missing_frames_match(both):
    fresh, cached, *_ = both
    assert ([i for i, r in enumerate(fresh) if r[0] == UNKNOWN] ==
            [i for i, r in enumerate(cached) if r[0] == UNKNOWN])


# ------------------------------------------------- provenance invariant


def test_the_tracker_really_does_hold_boxes_in_the_gap(both):
    """Guard for the guard: if the hold ever stops happening, the test below
    would pass vacuously."""
    _, _, tr_fresh, _, _ = both
    held = [i for i in GAP if tr_fresh['ball'][i]
            and next(iter(tr_fresh['ball'][i].values())).get('held_for')]
    assert held, 'expected the bounded hold to copy a box into the gap'


def test_a_held_box_never_becomes_observed(both):
    """The exact defect: a copied bbox reported as a measurement."""
    fresh, cached, tr_fresh, _, _ = both
    for i in GAP:
        entry = tr_fresh['ball'][i]
        if not entry:
            continue
        held_bbox = tuple(round(v, 4)
                          for v in next(iter(entry.values()))['bbox'])
        for label, rows in (('fresh', fresh), ('cached', cached)):
            state, bbox, _ = rows[i]
            assert not (state == OBSERVED and bbox == held_bbox), (
                f'{label} frame {i}: held box {held_bbox} reported as OBSERVED')


def test_no_output_confidence_was_invented_from_a_held_box(both):
    """The held copy carried conf 0.9 from the last real detection. No frame in
    the gap may report that as its own."""
    fresh, cached, *_ = both
    for rows in (fresh, cached):
        for i in GAP:
            state, _, conf = rows[i]
            assert state in (UNKNOWN, INTERPOLATED)
            if state == INTERPOLATED:
                assert conf is None, 'an estimate has no detector confidence'


def test_rescue_candidate_survives_the_cache(both):
    """The low-confidence candidate at RESCUE exists only in the candidate
    pool. If the cache drops it, this frame silently changes meaning."""
    fresh, cached, *_ = both
    assert fresh[RESCUE][0] == RECOVERED
    assert cached[RESCUE][0] == RECOVERED
    assert cached[RESCUE][2] == pytest.approx(0.15)


# ----------------------------------------------------------- cache format


def test_cache_payload_is_v2_and_carries_candidates(tmp_path):
    cache = tmp_path / 'tracks.pkl'
    run(tracker(tmp_path), cache, read=False)
    with open(cache, 'rb') as fh:
        blob = pickle.load(fh)
    assert blob['cache_format'] == FootballTracker.CACHE_FORMAT == 2
    assert set(blob) == {'cache_format', 'tracks', 'ball_candidates'}
    assert len(blob['ball_candidates']) == N_FRAMES
    assert blob['ball_candidates'][RESCUE][0]['confidence'] == pytest.approx(0.15)


def test_cache_hit_restores_candidates(tmp_path):
    cache = tmp_path / 'tracks.pkl'
    run(tracker(tmp_path), cache, read=False)
    t = tracker(tmp_path)
    assert t.ball_candidates is None
    t.get_object_tracks(frames(), read_from_cache=True, cache_path=str(cache))
    assert t.ball_candidates is not None
    assert len(t.ball_candidates) == N_FRAMES


# ------------------------------------------------- v1 backward compatibility


def test_a_v1_cache_is_ignored_rather_than_upgraded(tmp_path, capsys):
    """A v1 cache holds tracks only. It must be recomputed, never 'upgraded'
    by deriving candidates from tracks['ball'] -- those are held boxes."""
    cache = tmp_path / 'tracks.pkl'
    _, tracks = run(tracker(tmp_path), cache, read=False)
    with open(cache, 'wb') as fh:
        pickle.dump(tracks, fh)            # bare dict == the old v1 format

    t = tracker(tmp_path)
    out = t.get_object_tracks(frames(), read_from_cache=True,
                              cache_path=str(cache))
    assert 'stale cache' in capsys.readouterr().out.lower()
    assert t.ball_candidates is not None, 'must have recomputed, not given up'
    assert len(t.ball_candidates) == N_FRAMES
    assert len(out['ball']) == N_FRAMES


def test_v1_cache_result_equals_a_fresh_result(tmp_path):
    cache = tmp_path / 'tracks.pkl'
    fresh, tracks = run(tracker(tmp_path), cache, read=False)
    with open(cache, 'wb') as fh:
        pickle.dump(tracks, fh)
    recomputed, _ = run(tracker(tmp_path), cache, read=True)
    assert recomputed == fresh


def test_unpack_rejects_anything_that_is_not_v2():
    for blob in ({'players': []}, {'cache_format': 1, 'tracks': {}},
                 {'cache_format': 99, 'tracks': {}}, [], None):
        assert FootballTracker._unpack_cache(blob) == (None, None)


# -------------------------------- the fallback must stay gone


def test_selector_refuses_to_run_without_candidates(tmp_path):
    """Passing None must raise, not quietly reconstruct evidence from tracks."""
    t = tracker(tmp_path)
    ball = [{1: {'bbox': [10, 10, 20, 20], 'confidence': 0.9}}]
    with pytest.raises(ValueError, match='requires detector candidates'):
        t.apply_ball_temporal_selection(ball, candidates=None, fps=FPS,
                                        frame_width=W)


def test_selector_refuses_a_short_candidate_record(tmp_path):
    t = tracker(tmp_path)
    ball = [{}, {}, {}]
    with pytest.raises(ValueError, match='refusing to guess'):
        t.apply_ball_temporal_selection(ball, candidates=[[]], fps=FPS,
                                        frame_width=W)


def test_no_reconstruct_from_tracks_fallback_remains_in_the_source():
    import inspect
    src = inspect.getsource(FootballTracker.apply_ball_temporal_selection)
    body = src[src.index('n = len(ball_positions)'):]
    assert 'ball_positions[i].values()' not in body, (
        'ball_positions must never be read as candidate evidence')
