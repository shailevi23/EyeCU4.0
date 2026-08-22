"""
Goalkeeper possession gap: PlayerBallAssigner used to search only
tracks['players'] for the ball's nearest human, so a goalkeeper holding the
ball could never receive has_ball=True. Fixed by searching players AND
goalkeepers together (same geometry/max_distance, unchanged) for WHO
possesses the ball, while keeping team-control credit restricted to actual
field players -- goalkeepers are deliberately excluded from jersey
TeamAssigner and must not have a team fabricated from kit color.
"""
import numpy as np

from trackers.player_ball_assigner import PlayerBallAssigner

NEAR = [90, 90, 110, 130]     # feet near the ball
FAR = [500, 90, 520, 130]     # far away
BALL = [95, 120, 105, 130]


def test_a_player_nearest_ball_gets_has_ball():
    tracks = {
        'players': [{1: {'bbox': NEAR, 'team': 1}, 2: {'bbox': FAR, 'team': 2}}],
        'goalkeepers': [{}],
        'ball': [{1: {'bbox': BALL}}],
    }
    control = PlayerBallAssigner(max_distance=70).compute_team_ball_control(tracks)
    assert tracks['players'][0][1]['has_ball'] is True
    assert tracks['players'][0][2]['has_ball'] is False
    assert list(control) == [1], 'field player possession still credits their team'


def test_b_goalkeeper_nearest_ball_gets_has_ball():
    tracks = {
        'players': [{2: {'bbox': FAR, 'team': 2}}],
        'goalkeepers': [{9: {'bbox': NEAR}}],  # goalkeeper is the one near the ball
        'ball': [{1: {'bbox': BALL}}],
    }
    control = PlayerBallAssigner(max_distance=70).compute_team_ball_control(tracks)
    assert tracks['goalkeepers'][0][9]['has_ball'] is True
    assert tracks['players'][0][2]['has_ball'] is False
    assert list(control) == [0], 'goalkeeper possession must not be counted here'


def test_c_goalkeeper_possession_does_not_invent_a_team():
    tracks = {
        'players': [{2: {'bbox': FAR, 'team': 2}}],
        'goalkeepers': [{9: {'bbox': NEAR}}],
        'ball': [{1: {'bbox': BALL}}],
    }
    PlayerBallAssigner(max_distance=70).compute_team_ball_control(tracks)
    # A goalkeeper dict entry must never gain a 'team' key from this call --
    # TeamAssigner is the only thing allowed to write 'team', and it never
    # writes to tracks['goalkeepers'].
    assert 'team' not in tracks['goalkeepers'][0][9]


def test_d_no_ball_or_unresolved_ball_remains_unknown():
    tracks = {
        'players': [{1: {'bbox': NEAR, 'team': 1}}],
        'goalkeepers': [{9: {'bbox': FAR}}],
        'ball': [{}],  # unresolved -- BallTemporalSelector reported no ball this frame
    }
    control = PlayerBallAssigner(max_distance=70).compute_team_ball_control(tracks)
    assert list(control) == [0]
    # No ball this frame -> the possession pass never runs at all (matches
    # existing behavior for tracks['players']); has_ball is simply absent,
    # not a claimed False.
    assert 'has_ball' not in tracks['players'][0][1]
    assert 'has_ball' not in tracks['goalkeepers'][0][9]


def test_missing_goalkeepers_key_still_works():
    """Several existing possession tests build tracks with no 'goalkeepers'
    key at all -- must not KeyError."""
    tracks = {
        'players': [{1: {'bbox': NEAR, 'team': 1}}],
        'ball': [{1: {'bbox': BALL}}],
    }
    control = PlayerBallAssigner(max_distance=70).compute_team_ball_control(tracks)
    assert list(control) == [1]


def test_goalkeeper_beats_farther_player_for_possession():
    """When the goalkeeper is genuinely closer, they must win the nearest-
    human search over a farther field player -- not just be a fallback."""
    tracks = {
        'players': [{2: {'bbox': FAR, 'team': 2}}],
        'goalkeepers': [{9: {'bbox': NEAR}}],
        'ball': [{1: {'bbox': BALL}}],
    }
    PlayerBallAssigner(max_distance=70).compute_team_ball_control(tracks)
    assert tracks['goalkeepers'][0][9]['has_ball'] is True
