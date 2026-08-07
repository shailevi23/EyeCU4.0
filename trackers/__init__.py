"""
Football Analysis Trackers Package
"""

from trackers.football_tracker import FootballTracker
from trackers.roboflow_detector import RoboflowDetector
from trackers.team_assigner import TeamAssigner
from trackers.camera_movement import CameraMovementEstimator
from trackers.speed_distance import SpeedDistanceEstimator
from trackers.player_ball_assigner import PlayerBallAssigner
from trackers.video_utils import read_video, save_video, get_video_info

# trackers.football_analysis was a duplicate orchestrator and has been removed
# (TODO.md section 2). full_pipeline.py is the single orchestrator.

__all__ = [
    'FootballTracker',
    'RoboflowDetector',
    'TeamAssigner',
    'CameraMovementEstimator',
    'SpeedDistanceEstimator',
    'PlayerBallAssigner',
    'read_video',
    'save_video',
    'get_video_info',
]