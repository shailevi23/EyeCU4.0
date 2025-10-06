"""
Camera Movement Estimation using Optical Flow
Tracks camera movement between frames for accurate position tracking
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path

class CameraMovementEstimator:
    """
    Estimates camera movement between frames using optical flow
    """
    
    def __init__(self, reference_frame: np.ndarray, 
                minimum_distance: float = 5.0):
        """
        Initialize camera movement estimator
        
        Args:
            reference_frame: First frame of the sequence
            minimum_distance: Minimum distance to consider as camera movement
        """
        self.minimum_distance = minimum_distance
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Create feature detection mask (focus on field boundaries)
        first_frame_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_gray)
        
        # Focus on edges of frame (likely to have stable features)
        h, w = mask_features.shape
        mask_features[:, 0:int(w*0.02)] = 1  # Left edge (2%)
        mask_features[:, int(w*0.98):w] = 1  # Right edge (2%)
        mask_features[0:int(h*0.02), :] = 1  # Top edge (2%)
        mask_features[int(h*0.98):h, :] = 1  # Bottom edge (2%)
        
        # Feature detection parameters
        self.features_params = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )
    
    def add_adjusted_positions_to_tracks(self, 
                                        tracks: Dict, 
                                        camera_movement_per_frame: List) -> None:
        """
        Add camera-adjusted positions to tracking data
        
        Args:
            tracks: Dictionary of tracking data
            camera_movement_per_frame: List of [dx, dy] camera movements
        """
        # For each object type (players, ball, referees)
        for object_type, object_tracks in tracks.items():
            # For each frame
            for frame_num, frame_tracks in enumerate(object_tracks):
                # For each tracked object in frame
                for track_id, track_info in frame_tracks.items():
                    # Check if position exists
                    if 'position' not in track_info:
                        continue
                    
                    position = track_info['position']
                    
                    # Get camera movement for this frame
                    if frame_num < len(camera_movement_per_frame):
                        camera_movement = camera_movement_per_frame[frame_num]
                        
                        # Adjust position by subtracting camera movement
                        position_adjusted = (
                            position[0] - camera_movement[0],
                            position[1] - camera_movement[1]
                        )
                        
                        tracks[object_type][frame_num][track_id]['position_adjusted'] = position_adjusted
    
    def get_camera_movement(self, frames: List[np.ndarray], 
                           read_from_cache: bool = False,
                           cache_path: Optional[str] = None) -> List:
        """
        Calculate camera movement between frames
        
        Args:
            frames: List of video frames
            read_from_cache: Whether to read from cache if available
            cache_path: Path to cache file
            
        Returns:
            List of [dx, dy] camera movements per frame
        """
        # Check cache first
        if read_from_cache and cache_path and os.path.exists(cache_path):
            print(f"Loading camera movement from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Initialize camera movement list (first frame has no movement)
        camera_movement = [[0, 0]] * len(frames)
        
        # Convert first frame to grayscale
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        # Get good features to track from first frame
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features_params)
        
        # Process each subsequent frame
        for frame_num in range(1, len(frames)):
            # Convert current frame to grayscale
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )
            
            # Filter valid points
            if new_features is None or old_features is None:
                camera_movement[frame_num] = [0, 0]
                continue
                
            # Initialize max distance and movement
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0
            
            # Check movement for each tracked feature
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                # Check if this point is valid
                if status[i] == 1:  # Status 1 means the flow was found for this point
                    new_point = new.ravel()
                    old_point = old.ravel()
                    
                    # Calculate distance between old and new position
                    dx = old_point[0] - new_point[0]
                    dy = old_point[1] - new_point[1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # Keep track of maximum movement
                    if distance > max_distance:
                        max_distance = distance
                        camera_movement_x, camera_movement_y = dx, dy
            
            # Only consider movement if it exceeds minimum threshold
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                
                # Re-initialize features for next frame if significant movement
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features_params)
            else:
                camera_movement[frame_num] = [0, 0]
            
            # Update old_gray for next iteration
            old_gray = frame_gray.copy()
            
            # Progress update
            if frame_num % 50 == 0:
                print(f"Processed camera movement for frame {frame_num}/{len(frames)}")
        
        # Save to cache if requested
        if cache_path:
            print(f"Saving camera movement to cache: {cache_path}")
            # Make sure directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(camera_movement, f)
        
        return camera_movement
    
    def draw_camera_movement(self, frames: List[np.ndarray], 
                            camera_movement_per_frame: List) -> List[np.ndarray]:
        """
        Visualize camera movement on frames
        
        Args:
            frames: List of video frames
            camera_movement_per_frame: List of [dx, dy] camera movements
            
        Returns:
            List of annotated frames
        """
        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            # Create a copy of the frame
            frame = frame.copy()
            
            # Create semi-transparent overlay for text background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
            
            # Get camera movement for this frame
            if frame_num < len(camera_movement_per_frame):
                x_movement, y_movement = camera_movement_per_frame[frame_num]
                
                # Display camera movement
                cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            
            output_frames.append(frame)
        
        return output_frames

# Example usage
if __name__ == "__main__":
    import sys
    from trackers.video_utils import read_video
    
    # Load video
    video_frames = read_video('input_video.mp4', max_frames=100)
    
    # Initialize camera movement estimator
    estimator = CameraMovementEstimator(video_frames[0])
    
    # Get camera movement
    camera_movement = estimator.get_camera_movement(
        video_frames,
        read_from_cache=False,
        cache_path='cache/camera_movement.pkl'
    )
    
    # Visualize
    output_frames = estimator.draw_camera_movement(video_frames, camera_movement)
    
    # Display first few frames
    for i in range(min(5, len(output_frames))):
        cv2.imshow(f"Frame {i}", output_frames[i])
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()