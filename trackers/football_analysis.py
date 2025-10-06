"""
Advanced Football Tracking System
Main module that integrates all components
"""

import cv2
import numpy as np
import os
from pathlib import Path
import time
import pickle
from typing import Dict, List, Tuple, Optional
import argparse

# Import all component modules
from trackers.video_utils import read_video, save_video
from trackers.football_tracker import FootballTracker
from trackers.team_assigner import TeamAssigner
from trackers.camera_movement import CameraMovementEstimator
from trackers.speed_distance import SpeedDistanceEstimator
from trackers.player_ball_assigner import PlayerBallAssigner

def create_directories(base_dir: str) -> Dict[str, str]:
    """
    Create directory structure for output files
    
    Args:
        base_dir: Base output directory
        
    Returns:
        Dictionary of paths
    """
    # Create base directory
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    paths = {
        'cache': base_path / 'cache',
        'videos': base_path / 'videos',
        'stats': base_path / 'stats',
        'debug': base_path / 'debug'
    }
    
    for path in paths.values():
        path.mkdir(exist_ok=True)
    
    return {k: str(v) for k, v in paths.items()}

class FootballAnalysis:
    """
    Main class that integrates all tracking components
    """
    
    def __init__(self, 
                output_dir: str = 'football_analysis_output',
                model_path: str = 'yolov8s.pt',
                use_roboflow: bool = False,
                api_key: Optional[str] = None,
                use_cache: bool = True):
        """
        Initialize football analysis system
        
        Args:
            output_dir: Output directory
            model_path: Path to YOLO model
            use_roboflow: Whether to use Roboflow API
            api_key: Roboflow API key
            use_cache: Whether to use cached results
        """
        self.output_dir = output_dir
        self.model_path = model_path
        self.use_roboflow = use_roboflow
        self.api_key = api_key
        self.use_cache = use_cache
        
        # Create directories
        self.paths = create_directories(output_dir)
        
        # Initialize components
        print("Initializing Football Analysis components...")
        
        # Tracker (detection + tracking)
        print("  - Initializing Football Tracker...")
        self.tracker = FootballTracker(
            model_path=model_path,
            use_roboflow=use_roboflow,
            api_key=api_key,
            persist_cache=use_cache,
            cache_dir=self.paths['cache']
        )
        
        # Team assignment
        print("  - Initializing Team Assigner...")
        self.team_assigner = TeamAssigner(num_teams=2)
        
        # Player-ball association
        print("  - Initializing Player-Ball Assigner...")
        self.ball_assigner = PlayerBallAssigner(max_distance=70)
        
        print("Initialization complete!")
    
    def process_video(self, 
                     video_path: str,
                     max_frames: Optional[int] = None,
                     start_frame: int = 0,
                     fps: int = 30) -> Dict:
        """
        Process complete video
        
        Args:
            video_path: Path to input video
            max_frames: Maximum number of frames to process
            start_frame: Starting frame
            fps: Output video frame rate
            
        Returns:
            Processing results
        """
        start_time = time.time()
        
        # 1. Load video frames
        print(f"Loading video: {video_path}")
        frames = read_video(
            video_path,
            max_frames=max_frames,
            start_frame=start_frame
        )
        
        # 2. Get object tracks
        print("Detecting and tracking objects...")
        cache_path = os.path.join(self.paths['cache'], 'tracks.pkl')
        tracks = self.tracker.get_object_tracks(
            frames,
            read_from_cache=self.use_cache,
            cache_path=cache_path
        )
        
        # 3. Add position information to tracks
        print("Adding position information...")
        self.tracker.add_position_to_tracks(tracks)
        
        # 4. Interpolate ball positions
        print("Interpolating ball positions...")
        tracks["ball"] = self.tracker.interpolate_ball_positions(tracks["ball"])
        
        # 5. Estimate camera movement
        print("Estimating camera movement...")
        camera_estimator = CameraMovementEstimator(frames[0])
        camera_cache = os.path.join(self.paths['cache'], 'camera_movement.pkl')
        camera_movement = camera_estimator.get_camera_movement(
            frames,
            read_from_cache=self.use_cache,
            cache_path=camera_cache
        )
        
        # 6. Add adjusted positions
        print("Adding camera-adjusted positions...")
        camera_estimator.add_adjusted_positions_to_tracks(tracks, camera_movement)
        
        # 7. Assign teams to players
        print("Assigning teams...")
        self.team_assigner.assign_teams_to_tracks(frames, tracks)
        
        # 8. Calculate speed and distance
        print("Calculating speed and distance...")
        speed_estimator = SpeedDistanceEstimator(frame_rate=fps)
        speed_estimator.add_speed_and_distance_to_tracks(tracks)
        
        # 9. Determine ball possession and team control
        print("Analyzing ball possession...")
        team_ball_control = self.ball_assigner.compute_team_ball_control(tracks)
        
        # 10. Generate outputs
        print("Generating visualization...")
        output_frames = self.tracker.draw_annotations(
            frames, tracks, team_ball_control)
        
        # Add camera movement indicators
        output_frames = camera_estimator.draw_camera_movement(
            output_frames, camera_movement)
        
        # Add speed and distance
        output_frames = speed_estimator.draw_speed_and_distance(
            output_frames, tracks)
        
        # 11. Save output video
        output_video = os.path.join(self.paths['videos'], 'output.mp4')
        print(f"Saving output video to {output_video}...")
        save_video(output_frames, output_video, fps=fps)
        
        # 12. Generate statistics
        print("Generating player statistics...")
        player_stats = speed_estimator.get_player_statistics(tracks)
        
        # Save stats to file
        stats_file = os.path.join(self.paths['stats'], 'player_stats.pkl')
        with open(stats_file, 'wb') as f:
            pickle.dump(player_stats, f)
        
        # Print summary
        print("\nProcessing summary:")
        print(f"  - Frames processed: {len(frames)}")
        print(f"  - Players detected: {len(player_stats)}")
        print(f"  - Processing time: {time.time() - start_time:.2f} seconds")
        
        # Return results
        return {
            'tracks': tracks,
            'player_stats': player_stats,
            'team_ball_control': team_ball_control,
            'output_video': output_video,
            'processing_time': time.time() - start_time
        }

def main():
    """
    Main entry point when run as script
    """
    parser = argparse.ArgumentParser(description="Football Analysis System")
    parser.add_argument("--video", "-v", required=True, help="Path to input video")
    parser.add_argument("--output", "-o", default="football_analysis_output", help="Output directory")
    parser.add_argument("--model", "-m", default="yolov8s.pt", help="Path to YOLO model")
    parser.add_argument("--max-frames", "-f", type=int, help="Maximum frames to process")
    parser.add_argument("--start-frame", "-s", type=int, default=0, help="Starting frame")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--roboflow", action="store_true", help="Use Roboflow API")
    parser.add_argument("--api-key", help="Roboflow API key")
    
    args = parser.parse_args()
    
    # Initialize system
    analysis = FootballAnalysis(
        output_dir=args.output,
        model_path=args.model,
        use_roboflow=args.roboflow,
        api_key=args.api_key,
        use_cache=not args.no_cache
    )
    
    # Process video
    analysis.process_video(
        video_path=args.video,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        fps=args.fps
    )

if __name__ == "__main__":
    main()