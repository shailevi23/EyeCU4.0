"""
Speed and Distance Estimation for Football Players
Calculates player movement speed and total distance covered

⚠️ UNCALIBRATED — every km/h and metre figure produced here is unvalidated.

`pixels_per_meter` defaults to 12.0, which is a guess, not a measurement. It is
also treated as a single constant for the whole frame, which cannot be right for
broadcast footage: perspective makes a metre near the far touchline span far
fewer pixels than a metre near the camera, and the value changes again whenever
the camera zooms or pans.

Speeds are therefore usable only as relative comparisons within one clip, not as
real-world values. Calibration (pitch homography from known line markings) is a
separate, still-open task -- see docs/archive/TODO_legacy.md.
"""

import cv2
import numpy as np
from typing import Dict, List
import math

def measure_distance(p1, p2):
    """
    Calculate Euclidean distance between two points
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_foot_position(bbox):
    """
    Get player's foot position (bottom center of bbox)
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

class SpeedDistanceEstimator:
    """
    Estimates player speed and distance covered
    """
    
    def __init__(self, frame_rate: float = 24.0, 
                frame_window: int = 5,
                pixels_per_meter: float = 12.0,
                use_adjusted_positions: bool = True):
        """
        Initialize estimator
        
        Args:
            frame_rate: Video frame rate
            frame_window: Number of frames to use for speed calculation
            pixels_per_meter: Conversion factor from pixels to meters
            use_adjusted_positions: Whether to use camera-adjusted positions
        """
        self.frame_rate = frame_rate
        self.frame_window = frame_window
        self.pixels_per_meter = pixels_per_meter
        self.use_adjusted_positions = use_adjusted_positions
    
    def add_speed_and_distance_to_tracks(self, tracks: Dict) -> None:
        """
        Calculate speed and distance for all tracked objects
        
        Args:
            tracks: Dictionary of tracking data
        """
        total_distance = {}  # Keep track of cumulative distance
        
        # Process each object type
        for object_type, object_tracks in tracks.items():
            # Skip non-player objects
            if object_type not in ['players']:
                continue
                
            # Get total number of frames
            num_frames = len(object_tracks)
            
            # Process frames in windows
            for frame_num in range(0, num_frames, self.frame_window):
                # Determine last frame in window
                last_frame = min(frame_num + self.frame_window, num_frames - 1)
                
                # Process each tracked object
                for track_id, track_info in object_tracks[frame_num].items():
                    # Skip if not present in last frame of window
                    if track_id not in object_tracks[last_frame]:
                        continue
                    
                    # Get position (use adjusted if available)
                    if self.use_adjusted_positions and 'position_adjusted' in track_info:
                        start_position = track_info['position_adjusted']
                        end_position = object_tracks[last_frame][track_id]['position_adjusted']
                    elif 'position' in track_info:
                        start_position = track_info['position']
                        end_position = object_tracks[last_frame][track_id]['position']
                    else:
                        continue
                    
                    # Calculate distance in pixels
                    pixel_distance = measure_distance(start_position, end_position)
                    
                    # Convert to meters
                    distance_meters = pixel_distance / self.pixels_per_meter
                    
                    # Calculate time elapsed
                    time_elapsed = (last_frame - frame_num) / self.frame_rate  # seconds
                    
                    # Calculate speed (m/s and km/h)
                    speed_mps = distance_meters / time_elapsed if time_elapsed > 0 else 0
                    speed_kmh = speed_mps * 3.6  # Convert m/s to km/h
                    
                    # Track total distance
                    if object_type not in total_distance:
                        total_distance[object_type] = {}
                    
                    if track_id not in total_distance[object_type]:
                        total_distance[object_type][track_id] = 0
                    
                    total_distance[object_type][track_id] += distance_meters
                    
                    # Add to all frames in window
                    for i in range(frame_num, last_frame + 1):
                        if track_id in object_tracks[i]:
                            object_tracks[i][track_id]['speed'] = speed_kmh
                            object_tracks[i][track_id]['distance'] = total_distance[object_type][track_id]
    
    def draw_speed_and_distance(self, frames: List[np.ndarray], 
                               tracks: Dict,
                               show_speed: bool = False,
                               show_distance: bool = False,
                               compact_display: bool = True) -> List[np.ndarray]:
        """
        Visualize speed and distance on frames
        
        Args:
            frames: List of video frames
            tracks: Dictionary of tracking data
            show_speed: Whether to display speed information
            show_distance: Whether to display distance information
            compact_display: Use compact display format
            
        Returns:
            List of annotated frames
        """
        # If both displays are off, just return the original frames
        if not show_speed and not show_distance:
            return frames
            
        output_frames = []
        
        # Process each frame
        for frame_num, frame in enumerate(frames):
            # Create a copy of the frame
            frame_copy = frame.copy()
            
            # Draw speed and distance for players
            if frame_num < len(tracks['players']):
                for track_id, track_info in tracks['players'][frame_num].items():
                    # Skip if no speed/distance info
                    if 'speed' not in track_info or 'distance' not in track_info:
                        continue
                    
                    # Get values
                    speed = track_info['speed']
                    distance = track_info['distance']
                    bbox = track_info['bbox']
                    
                    # Skip if no information to show
                    if not show_speed and not show_distance:
                        continue
                    
                    # Get position for text (below feet)
                    position = get_foot_position(bbox)
                    position = (position[0], position[1] + 20)
                    
                    # Create text content based on what should be displayed
                    if show_speed and show_distance and compact_display:
                        info_text = f"{speed:.1f}km/h | {distance:.1f}m"
                    elif show_speed and show_distance:
                        # Non-compact: will be displayed on two lines
                        speed_text = f"{speed:.1f}km/h"
                        distance_text = f"{distance:.1f}m"
                        
                        # Draw speed on first line (smaller)
                        self._draw_text_with_background(
                            frame_copy, 
                            speed_text,
                            (position[0], position[1] - 15),  # Above distance
                            0.4,  # Smaller font
                            bg_alpha=0.5  # More transparent
                        )
                        
                        # Draw distance on second line
                        self._draw_text_with_background(
                            frame_copy,
                            distance_text,
                            position,
                            0.4,  # Smaller font
                            bg_alpha=0.5  # More transparent
                        )
                        
                        # Skip the rest of the loop since we've drawn both already
                        continue
                    elif show_speed:
                        info_text = f"{speed:.1f}km/h"
                    elif show_distance:
                        info_text = f"{distance:.1f}m"
                    else:
                        continue  # Nothing to show
                    
                    # Draw the text with a semi-transparent background
                    self._draw_text_with_background(frame_copy, info_text, position)
            
            output_frames.append(frame_copy)
        
        return output_frames
        
    def _draw_text_with_background(self, frame, text, position, 
                                 font_scale=0.4, 
                                 font_color=(255, 255, 255),
                                 bg_color=(0, 0, 0),
                                 font_thickness=1,
                                 bg_alpha=0.5):
        """
        Helper method to draw text with a semi-transparent background
        """
        # Get text size
        text_size = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness
        )[0]
        
        # Calculate background coordinates
        bg_coords = (
            position[0] - 3,
            position[1] - text_size[1] - 3,
            position[0] + text_size[0] + 3,
            position[1] + 3
        )
        
        # Create overlay for semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (bg_coords[0], bg_coords[1]),
            (bg_coords[2], bg_coords[3]),
            bg_color,
            -1
        )
        
        # Apply the overlay with transparency
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
        
        # Draw the text
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_color,
            font_thickness
        )
    
    def get_player_statistics(self, tracks: Dict) -> Dict:
        """
        Generate statistics for players
        
        Args:
            tracks: Dictionary of tracking data
            
        Returns:
            Dictionary of player statistics
        """
        stats = {}
        
        # Skip if no player tracks
        if 'players' not in tracks or not tracks['players']:
            return stats
        
        # Collect all track IDs
        all_track_ids = set()
        for frame_tracks in tracks['players']:
            all_track_ids.update(frame_tracks.keys())
        
        # Process each player
        for track_id in all_track_ids:
            max_speed = 0
            max_distance = 0
            appearances = 0
            
            # Find player's max speed and distance
            for frame_tracks in tracks['players']:
                if track_id in frame_tracks:
                    appearances += 1
                    track_info = frame_tracks[track_id]
                    
                    speed = track_info.get('speed', 0)
                    distance = track_info.get('distance', 0)
                    
                    max_speed = max(max_speed, speed)
                    max_distance = max(max_distance, distance)
            
            # Get team if available
            team = None
            for frame_tracks in tracks['players']:
                if track_id in frame_tracks and 'team' in frame_tracks[track_id]:
                    team = frame_tracks[track_id]['team']
                    break
            
            # Cast out of numpy types: track ids arrive from ByteTrack as int64,
            # which json.dump rejects as a dict key.
            stats[str(int(track_id))] = {
                'track_id': int(track_id),
                'team': int(team) if team is not None else None,
                # Marked on every record so these numbers cannot be lifted out
                # of the JSON and quoted as real speeds.
                'max_speed_kmh_UNCALIBRATED': float(max_speed),
                'total_distance_m_UNCALIBRATED': float(max_distance),
                'appearances': int(appearances)
            }

        return stats
