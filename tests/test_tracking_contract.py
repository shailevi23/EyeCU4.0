"""
docs/archive/TODO_legacy.md section 8 behavioural guarantees, exercised with a fake detector so the
tests are fast and deterministic and never load real weights.

Covered:
  * exactly one detector inference per processed frame
  * four detector classes survive end to end
  * the ball becomes unknown after the configured missing-frame gap
  * referees and goalkeepers are never assigned a team
"""

import numpy as np
import pytest

from conftest import FakeDetector, det
from trackers.football_tracker import TRACK_KEYS, FootballTracker
from trackers.team_assigner import TeamAssigner


def make_tracker(script, **kw):
    return FootballTracker(detector=FakeDetector(script), persist_cache=False, **kw)


def all_four_classes(frame_no):
    """Two players, a goalkeeper, a referee and the ball, stable across frames."""
    return [
        det('player', 100, 100),
        det('player', 200, 100),
        det('goalkeeper', 300, 100),
        det('referee', 400, 100),
        det('ball', 250, 150, w=6, h=6),
    ]


# --- one inference per processed frame -------------------------------------

def test_exactly_one_inference_per_processed_frame(frames):
    n = 25
    detector = FakeDetector(all_four_classes)
    tracker = FootballTracker(detector=detector, persist_cache=False)

    tracker.get_object_tracks(frames(n), read_from_cache=False, cache_path=None)

    assert detector.predict_calls == n, (
        f'expected {n} inferences for {n} frames, got {detector.predict_calls}'
    )
    assert detector.inference_count == n


def test_no_duplicate_detector_pass(frames):
    """The removed 'refresh detections' pass re-ran inference on every 20th
    frame. With 60 frames that was 3 extra calls."""
    n = 60
    detector = FakeDetector(all_four_classes)
    tracker = FootballTracker(detector=detector, persist_cache=False)
    tracker.get_object_tracks(frames(n), read_from_cache=False, cache_path=None)
    assert detector.predict_calls == n


# --- four classes preserved -------------------------------------------------

def test_four_track_keys_exist():
    assert TRACK_KEYS == ['players', 'goalkeepers', 'referees', 'ball']


def test_all_four_classes_survive_the_pipeline(frames):
    tracker = make_tracker(all_four_classes)
    tracks = tracker.get_object_tracks(frames(30), read_from_cache=False,
                                       cache_path=None)

    assert set(tracks) == set(TRACK_KEYS)
    for key in TRACK_KEYS:
        total = sum(len(f) for f in tracks[key])
        assert total > 0, f'{key} was lost between detector and tracks'


def test_goalkeeper_is_not_merged_into_players(frames):
    """The old detector rewrote goalkeeper -> player. It must stay separate."""
    tracker = make_tracker(all_four_classes)
    tracks = tracker.get_object_tracks(frames(30), read_from_cache=False,
                                       cache_path=None)
    assert sum(len(f) for f in tracks['goalkeepers']) > 0


# --- ball gap ---------------------------------------------------------------

def ball_only_on_first_frame(frame_no):
    people = [det('player', 100, 100), det('player', 200, 100)]
    if frame_no == 0:
        people.append(det('ball', 250, 150, w=6, h=6))
    return people


@pytest.mark.parametrize('max_ball_gap', [1, 3, 8])
def test_ball_becomes_unknown_after_max_gap(frames, max_ball_gap):
    n = 30
    tracker = make_tracker(ball_only_on_first_frame, max_ball_gap=max_ball_gap)
    tracks = tracker.get_object_tracks(frames(n), read_from_cache=False,
                                       cache_path=None)

    ball = tracks['ball']
    assert ball[0], 'ball should be present on the frame it was detected'

    # Held for exactly max_ball_gap frames after the last detection...
    for i in range(1, max_ball_gap + 1):
        assert ball[i], f'ball should still be held at frame {i}'
        assert all('held_for' in v for v in ball[i].values())

    # ...then unknown for the rest, not frozen in place forever.
    for i in range(max_ball_gap + 1, n):
        assert not ball[i], f'ball should be unknown at frame {i}'


def test_ball_hold_never_exceeds_configured_gap(frames):
    tracker = make_tracker(ball_only_on_first_frame, max_ball_gap=3)
    tracks = tracker.get_object_tracks(frames(40), read_from_cache=False,
                                       cache_path=None)
    holds = [v['held_for'] for f in tracks['ball'] for v in f.values()
             if 'held_for' in v]
    assert holds and max(holds) == 3


# --- referees / goalkeepers never get a team --------------------------------

def coloured_tracks(n_frames=10):
    """Two bright players, a dark referee and a goalkeeper, all stationary."""
    players, keepers, refs = [], [], []
    for _ in range(n_frames):
        players.append({
            1: {'bbox': [100.0, 100.0, 120.0, 140.0]},
            2: {'bbox': [300.0, 100.0, 320.0, 140.0]},
        })
        keepers.append({10: {'bbox': [500.0, 100.0, 520.0, 140.0]}})
        refs.append({20: {'bbox': [50.0, 100.0, 70.0, 140.0]}})
    return {'players': players, 'goalkeepers': keepers,
            'referees': refs, 'ball': [{} for _ in range(n_frames)]}


def coloured_frames(n_frames=10):
    out = []
    for _ in range(n_frames):
        f = np.zeros((360, 640, 3), dtype=np.uint8)
        f[100:140, 100:120] = (240, 240, 240)   # team A: white
        f[100:140, 300:320] = (60, 220, 220)    # team B: yellow
        f[100:140, 500:520] = (30, 120, 30)     # goalkeeper: green
        f[100:140, 50:70] = (20, 20, 20)        # referee: near-black
        out.append(f)
    return out


def test_referees_are_never_assigned_a_team():
    tracks = coloured_tracks()
    TeamAssigner(num_teams=2).assign_teams_to_tracks(coloured_frames(), tracks)

    for frame in tracks['referees']:
        for info in frame.values():
            assert 'team' not in info
            assert 'team_color' not in info


def test_goalkeepers_are_never_assigned_a_team():
    """A goalkeeper's kit deliberately differs from their own team's, so
    jersey-colour clustering must not label them."""
    tracks = coloured_tracks()
    TeamAssigner(num_teams=2).assign_teams_to_tracks(coloured_frames(), tracks)

    for frame in tracks['goalkeepers']:
        for info in frame.values():
            assert 'team' not in info


def test_outfield_players_do_get_a_team():
    """Guards the two tests above from passing because nothing is assigned."""
    tracks = coloured_tracks()
    TeamAssigner(num_teams=2).assign_teams_to_tracks(coloured_frames(), tracks)

    for frame in tracks['players']:
        for info in frame.values():
            assert info.get('team') in (1, 2)
