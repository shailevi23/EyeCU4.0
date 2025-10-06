"""
Team Assignment based on Jersey Colors
Uses KMeans clustering to assign players to teams based on jersey colors
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import time

class TeamAssigner:
    """
    Assign players to teams based on jersey colors using KMeans clustering
    """
    
    def __init__(self, num_teams: int = 2, 
                use_top_half: bool = True):
        """
        Initialize team assigner
        
        Args:
            num_teams: Number of teams to detect
            use_top_half: Whether to use only top half of player for color detection
        """
        self.num_teams = num_teams
        self.use_top_half = use_top_half
        self.team_colors = {}  # Will contain team colors as RGB values
        self.player_team_dict = {}  # Maps player_id to team_id
        self.kmeans = None
    
    def get_clustering_model(self, image: np.ndarray) -> KMeans:
        """
        Get KMeans clustering model for an image
        
        Args:
            image: Input image
            
        Returns:
            Trained KMeans model
        """
        # Reshape the image to 2D array of pixels
        image_2d = image.reshape(-1, 3)
        
        # Perform K-means with specified clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        
        return kmeans
    
    def get_player_color(self, frame: np.ndarray, 
                        bbox: List[float]) -> np.ndarray:
        """
        Get dominant jersey color for a player
        
        Args:
            frame: Video frame
            bbox: Player bounding box [x1, y1, x2, y2]
            
        Returns:
            RGB color value as numpy array
        """
        # Extract player crop from frame
        x1, y1, x2, y2 = map(int, bbox)
        player_img = frame[y1:y2, x1:x2]
        
        if player_img.size == 0:
            # Return default color if crop is empty
            return np.array([128, 128, 128])
        
        # Use top half of player for better jersey color detection
        if self.use_top_half:
            h = player_img.shape[0]
            player_img = player_img[0:int(h/2), :]
        
        # Get clustering model
        try:
            kmeans = self.get_clustering_model(player_img)
            
            # Get cluster labels for each pixel
            labels = kmeans.labels_
            
            # Reshape labels to image shape
            h, w = player_img.shape[:2]
            clustered_image = labels.reshape(h, w)
            
            # Determine which cluster represents the jersey (not background)
            # Assume background appears in corners, jersey in middle
            corner_clusters = [
                clustered_image[0, 0],      # Top-left
                clustered_image[0, -1],     # Top-right
                clustered_image[-1, 0],     # Bottom-left
                clustered_image[-1, -1]     # Bottom-right
            ]
            
            # Most common cluster in corners is likely background
            background_cluster = max(set(corner_clusters), key=corner_clusters.count)
            jersey_cluster = 1 - background_cluster  # For binary clustering (2 clusters)
            
            # Get jersey color (centroid of jersey cluster)
            jersey_color = kmeans.cluster_centers_[jersey_cluster]
            
            return jersey_color
            
        except Exception as e:
            print(f"Error in color clustering: {str(e)}")
            return np.array([128, 128, 128])  # Default gray
    
    def assign_team_colors(self, frame: np.ndarray, 
                          player_detections: Dict) -> None:
        """
        Assign team colors based on player detections
        
        Args:
            frame: Video frame
            player_detections: Dictionary of player detections with bboxes
        """
        # Extract player colors
        player_colors = []
        
        for _, player_detection in player_detections.items():
            bbox = player_detection.get("bbox", [0, 0, 0, 0])
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        if len(player_colors) < self.num_teams:
            print(f"Warning: Found only {len(player_colors)} players, need at least {self.num_teams} for team assignment")
            return
        
        # Cluster player colors into teams
        kmeans = KMeans(n_clusters=self.num_teams, init="k-means++", n_init=10)
        kmeans.fit(np.array(player_colors))
        
        self.kmeans = kmeans
        
        # Store team colors
        for i in range(self.num_teams):
            # Convert from RGB float to BGR int for OpenCV
            color_rgb = kmeans.cluster_centers_[i].astype(np.uint8)
            color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
            self.team_colors[i+1] = color_bgr
        
        print(f"Team colors assigned: {self.team_colors}")
    
    def get_player_team(self, frame: np.ndarray, 
                       player_bbox: List[float], 
                       player_id: int) -> int:
        """
        Get team assignment for a player
        
        Args:
            frame: Video frame
            player_bbox: Player bounding box
            player_id: Player tracking ID
            
        Returns:
            Team ID (1-based)
        """
        # Check if we already know this player's team
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        # Extract player color and predict team
        if self.kmeans is None:
            # If we haven't assigned team colors yet, assign to team 1
            team_id = 1
        else:
            player_color = self.get_player_color(frame, player_bbox)
            
            # Predict team based on color
            team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
            team_id += 1  # Convert to 1-based indexing
        
        # Special case handling for goalkeepers or specific players if needed
        # (commented out as it depends on specific use case)
        # if player_id == 91:  # Example: always assign player 91 to team 1
        #     team_id = 1
        
        # Store assignment for future reference
        self.player_team_dict[player_id] = team_id
        
        return team_id
    
    def assign_teams_to_tracks(self, frames: List[np.ndarray], 
                              tracks: Dict) -> None:
        """
        Assign teams to all players in tracking data
        
        Args:
            frames: List of video frames
            tracks: Dictionary of tracking data
        """
        # First, determine team colors from first frame with enough players
        for frame_idx, frame in enumerate(frames):
            if len(tracks['players'][frame_idx]) >= self.num_teams:
                self.assign_team_colors(frame, tracks['players'][frame_idx])
                break
        
        # Then, assign teams to all tracked players
        for frame_idx, frame in enumerate(frames):
            # Skip if no players in this frame
            if frame_idx >= len(tracks['players']):
                continue
                
            for player_id, player_data in tracks['players'][frame_idx].items():
                bbox = player_data.get('bbox', [0, 0, 0, 0])
                team_id = self.get_player_team(frame, bbox, player_id)
                
                # Store team assignment and color in tracking data
                tracks['players'][frame_idx][player_id]['team'] = team_id
                tracks['players'][frame_idx][player_id]['team_color'] = self.team_colors.get(
                    team_id, (0, 255, 0))  # Default green if team color not found

# Example usage
if __name__ == "__main__":
    import sys
    from trackers.video_utils import read_video
    from trackers.football_tracker import FootballTracker
    
    # Load video
    video_frames = read_video('input_video.mp4', max_frames=100)
    
    # Initialize tracker
    tracker = FootballTracker()
    
    # Get object tracks
    tracks = tracker.get_object_tracks(video_frames, read_from_cache=False)
    
    # Assign teams
    team_assigner = TeamAssigner(num_teams=2)
    team_assigner.assign_teams_to_tracks(video_frames, tracks)
    
    # Check results
    for frame_idx in range(min(5, len(tracks['players']))):
        print(f"Frame {frame_idx}:")
        for player_id, player_data in tracks['players'][frame_idx].items():
            team = player_data.get('team', 'Unknown')
            print(f"  Player {player_id}: Team {team}")