"""
Player-Ball Association
Determines which player has possession of the ball
"""

import numpy as np
import math
from typing import Dict, List

def get_center_of_bbox(bbox):
    """
    Get center point of bounding box
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def measure_distance(p1, p2):
    """
    Calculate Euclidean distance between two points
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class PlayerBallAssigner:
    """
    Assigns ball possession to players
    """
    
    def __init__(self, max_distance: float = 70):
        """
        Initialize ball assigner
        
        Args:
            max_distance: Maximum pixel distance to consider ball possession
        """
        self.max_distance = max_distance
    
    def assign_ball_to_player(self, players: Dict, ball_bbox: List[float]) -> int:
        """
        Determine which player has possession of the ball
        
        Args:
            players: Dictionary of players with bounding boxes
            ball_bbox: Ball bounding box
            
        Returns:
            Player ID with ball possession, or -1 if none
        """
        # Get ball position
        ball_position = get_center_of_bbox(ball_bbox)
        
        # Track closest player
        minimum_distance = float('inf')
        assigned_player = -1
        
        # Check each player
        for player_id, player in players.items():
            player_bbox = player['bbox']
            
            # Check both left and right foot for distance to ball
            left_foot = (player_bbox[0], player_bbox[3])  # Bottom left
            right_foot = (player_bbox[2], player_bbox[3])  # Bottom right
            
            # Get minimum distance to either foot
            distance_left = measure_distance(left_foot, ball_position)
            distance_right = measure_distance(right_foot, ball_position)
            distance = min(distance_left, distance_right)
            
            # If within max distance and closer than previous best
            if distance < self.max_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player

    def _nearest_human(self, humans: Dict, ball_bbox: List[float]) -> int:
        """
        Same geometry/threshold as assign_ball_to_player, but over an
        arbitrary dict of human tracks (players AND goalkeepers combined).
        assign_ball_to_player itself is left untouched -- tools/eval_possession_val*.py
        call it directly, players-only, and must keep that exact behavior.
        """
        ball_position = get_center_of_bbox(ball_bbox)
        minimum_distance = float('inf')
        assigned_id = -1
        for human_id, human in humans.items():
            bbox = human['bbox']
            left_foot = (bbox[0], bbox[3])
            right_foot = (bbox[2], bbox[3])
            distance = min(measure_distance(left_foot, ball_position),
                          measure_distance(right_foot, ball_position))
            if distance < self.max_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_id = human_id
        return assigned_id

    def compute_team_ball_control(self, tracks: Dict) -> np.ndarray:
        """
        Calculate ball possession percentage for each team
        
        Args:
            tracks: Dictionary of tracking data with player-ball assignments
            
        Returns:
            Array of team IDs with ball control per frame
        """
        team_ball_control = []
        
        # Go through each frame
        num_frames = len(tracks['players'])
        for frame_idx in range(num_frames):
            if not tracks['ball'][frame_idx]:
                # No reliable ball position -> possession is UNKNOWN, recorded
                # as 0. This used to carry the previous team's id forward, which
                # turned one observed frame into an unbounded run of credited
                # possession: 1 frame of evidence followed by 19 unknown frames
                # reported as 20 frames of possession, and 100%-0%.
                #
                # A ball the selector could recover -- observed, recovered or
                # interpolated -- still produces a bbox and is handled below, so
                # this branch is reached only when the evidence genuinely ran
                # out. Unknown ball position means unknown possession.
                team_ball_control.append(0)
                continue
            
            # Get ball position
            ball_id = list(tracks['ball'][frame_idx].keys())[0]
            ball_bbox = tracks['ball'][frame_idx][ball_id]['bbox']

            players_here = tracks['players'][frame_idx]
            # Goalkeepers are deliberately excluded from jersey TeamAssigner
            # (their kit differs from their own team's), but that must not
            # also exclude them from WHO can possess the ball -- a keeper
            # holding the ball is a real, common event. .get(...) keeps this
            # safe for any caller/tracks dict that has no 'goalkeepers' key
            # at all (several existing possession tests build tracks that way).
            gk_list = tracks.get('goalkeepers')
            goalkeepers_here = gk_list[frame_idx] if gk_list and frame_idx < len(gk_list) else {}
            humans_here = {**players_here, **goalkeepers_here}
            assigned_human = self._nearest_human(humans_here, ball_bbox)

            # Mark possession in tracking data -- on whichever dict the
            # possessor actually lives in. A player and a goalkeeper never
            # share a track_id (both come off the same human association
            # pass, just bucketed by class), so this lookup is unambiguous.
            for player_id in players_here:
                players_here[player_id]['has_ball'] = (player_id == assigned_human)
            for gk_id in goalkeepers_here:
                goalkeepers_here[gk_id]['has_ball'] = (gk_id == assigned_human)

            # Team-control credit is a SEPARATE question from who possesses
            # the ball. Only a field player carries a 'team' (jersey
            # cluster) label; a goalkeeper possessor must not fabricate one
            # from kit color, so team control stays UNKNOWN (0) for a
            # goalkeeper-possession frame, exactly as it does when no one is
            # in range at all.
            team_id = 0  # No team by default
            if assigned_human != -1 and assigned_human in players_here \
                    and 'team' in players_here[assigned_human]:
                team_id = players_here[assigned_human]['team']

            team_ball_control.append(team_id)
        
        return np.array(team_ball_control)
