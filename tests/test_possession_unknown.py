"""Unknown ball position means unknown possession.

Possession used to carry the previous team's id forward through frames with no
ball. One observed frame followed by nineteen blind ones was reported as twenty
frames of possession and a 100%-0% split -- a confident claim built from a
single measurement.

Now an unresolved frame records 0, and 0 is excluded from the percentage
denominator. A ball the selector could resolve at all -- observed, recovered or
interpolated -- still yields a bbox and takes part normally; this only governs
frames where the evidence genuinely ran out.
"""

import numpy as np
import pytest

from trackers.ball_temporal import INTERPOLATED, OBSERVED, RECOVERED
from trackers.player_ball_assigner import PlayerBallAssigner

NEAR = [90, 90, 110, 130]      # team 1 player, feet near (100, 130)
FAR = [500, 90, 520, 130]      # team 2 player, far away
BALL = [95, 120, 105, 130]


def tracks(ball_frames, teams=(1, 2)):
    players = [{1: {'bbox': NEAR, 'team': teams[0]},
                2: {'bbox': FAR, 'team': teams[1]}}
               for _ in ball_frames]
    return {'players': players, 'ball': ball_frames}


def ball(state=OBSERVED, conf=0.9):
    e = {'bbox': list(BALL), 'state': state}
    if conf is not None:
        e['confidence'] = conf
    return {1: e}


def control(ball_frames):
    return PlayerBallAssigner(max_distance=70).compute_team_ball_control(
        tracks(ball_frames))


def pct(c):
    """The production denominator: known-possession frames only.
    Mirrors football_tracker.draw_team_ball_control()."""
    c = np.asarray(c)
    t1 = int((c == 1).sum())
    t2 = int((c == 2).sum())
    known = t1 + t2
    if known == 0:
        return 0.0, 0.0, 0
    return t1 / known, t2 / known, known


# ------------------------------------------------ the headline scenario


def test_one_observed_frame_then_nineteen_unknown():
    """Frame 1: Team 1. Frames 2-20: UNKNOWN."""
    c = control([ball()] + [{}] * 19)
    assert list(c) == [1] + [0] * 19
    t1, t2, known = pct(c)
    assert (t1, t2) == (1.0, 0.0)
    assert known == 1, 'one frame of evidence, not twenty'
    assert int((np.asarray(c) == 1).sum()) == 1, (
        'team 1 must not be credited with the nineteen blind frames')


# ------------------------------------------------------- placement cases


def test_unknown_at_the_start_of_a_clip():
    c = control([{}] * 3 + [ball()] * 2)
    assert list(c) == [0, 0, 0, 1, 1]


def test_unknown_at_the_end_of_a_clip():
    c = control([ball()] * 2 + [{}] * 3)
    assert list(c) == [1, 1, 0, 0, 0]
    assert pct(c)[2] == 2


def test_long_unknown_interval_between_two_known_possessions():
    """The gap belongs to neither side, and both keep only what they measured."""
    c = control([ball()] * 2 + [{}] * 10 + [ball()] * 2)
    assert list(c) == [1, 1] + [0] * 10 + [1, 1]
    t1, t2, known = pct(c)
    assert known == 4 and t1 == 1.0 and t2 == 0.0


def test_no_known_possession_at_all_does_not_divide_by_zero():
    c = control([{}] * 5)
    assert list(c) == [0] * 5
    assert pct(c) == (0.0, 0.0, 0)


def test_empty_clip():
    assert list(control([])) == []


# ------------------------------- resolved balls still participate normally


@pytest.mark.parametrize('state,conf', [(OBSERVED, 0.9),
                                        (RECOVERED, 0.15),
                                        (INTERPOLATED, None)])
def test_any_resolved_state_counts_as_evidence(state, conf):
    """The rule keys on whether a usable bbox exists, not on how it was
    obtained -- a recovered or interpolated ball is a position the selector
    was willing to stand behind."""
    c = control([ball(state, conf)] * 3)
    assert list(c) == [1, 1, 1]
    assert pct(c)[2] == 3


def test_a_mixed_sequence_counts_only_the_resolved_frames():
    c = control([ball(OBSERVED), {}, ball(INTERPOLATED, None), {},
                 ball(RECOVERED, 0.15)])
    assert list(c) == [1, 0, 1, 0, 1]
    assert pct(c)[2] == 3


# --------------------------------------- a resolved ball nobody is near


def test_a_ball_with_no_player_near_it_is_also_zero():
    """0 already meant 'nobody had it'. An unresolved ball reuses that value:
    both are 'not known to belong to a team', and neither should count."""
    far_ball = [{1: {'bbox': [300, 300, 310, 310], 'state': OBSERVED,
                     'confidence': 0.9}}]
    c = PlayerBallAssigner(max_distance=70).compute_team_ball_control(
        tracks(far_ball))
    assert list(c) == [0]
    assert pct(c)[2] == 0


# ------------------------------------------------- the old behaviour is gone


def test_previous_team_is_never_inherited():
    """Team 2 possesses, then the ball vanishes. Team 2 must not keep it."""
    c = control([{}, ball()] + [{}] * 4, )
    assert list(c) == [0, 1, 0, 0, 0, 0]


def test_carry_forward_is_absent_from_the_source():
    import inspect
    src = inspect.getsource(PlayerBallAssigner.compute_team_ball_control)
    assert 'team_ball_control[-1]' not in src, (
        'possession must not read its own previous value')
