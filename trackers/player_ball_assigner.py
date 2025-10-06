"""
Player-Ball Association
Determines which player has possession of the ball
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional

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
            # Skip if no ball in this frame
            if not tracks['ball'][frame_idx]:
                # Use previous value if available
                if team_ball_control:
                    team_ball_control.append(team_ball_control[-1])
                else:
                    team_ball_control.append(0)  # No control
                continue
            
            # Get ball position
            ball_id = list(tracks['ball'][frame_idx].keys())[0]
            ball_bbox = tracks['ball'][frame_idx][ball_id]['bbox']
            
            # Find player with ball possession
            assigned_player = self.assign_ball_to_player(
                tracks['players'][frame_idx], ball_bbox)
            
            # Mark possession in tracking data
            for player_id in tracks['players'][frame_idx]:
                tracks['players'][frame_idx][player_id]['has_ball'] = (player_id == assigned_player)
            
            # Determine team with possession
            team_id = 0  # No team by default
            if assigned_player != -1 and 'team' in tracks['players'][frame_idx][assigned_player]:
                team_id = tracks['players'][frame_idx][assigned_player]['team']
            
            team_ball_control.append(team_id)
        
        return np.array(team_ball_control)

# Example usage
if __name__ == "__main__":
    import sys
    from trackers.video_utils import read_video
    from trackers.football_tracker import FootballTracker
    from trackers.team_assigner import TeamAssigner
    
    # Load video
    video_frames = read_video('input_video.mp4', max_frames=100)
    
    # Initialize tracker
    tracker = FootballTracker()
    
    # Get object tracks
    tracks = tracker.get_object_tracks(video_frames, read_from_cache=False)
    
    # Assign teams
    team_assigner = TeamAssigner()
    team_assigner.assign_teams_to_tracks(video_frames, tracks)
    
    # Assign ball possession
    ball_assigner = PlayerBallAssigner(max_distance=70)
    team_ball_control = ball_assigner.compute_team_ball_control(tracks)
    
    # Print results
    print("Team ball control:")
    for frame_idx in range(min(20, len(team_ball_control))):
        print(f"Frame {frame_idx}: Team {team_ball_control[frame_idx]}")