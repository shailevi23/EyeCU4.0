"""
Advanced Football Player Tracking using ByteTrack Algorithm
Dependencies: pip install supervision ultralytics numpy scipy opencv-python
"""

import cv2
import numpy as np
import supervision as sv
import os
from pathlib import Path
import pickle
import pandas as pd

# Import from local modules
from trackers.detector import CLASS_IDS, create_detector
from trackers.bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position

# tracks[] key for each detector class. Goalkeepers are kept separate from
# players so team assignment can exclude them (their kit deliberately differs
# from their own team's) -- see TODO.md section 5.
TRACK_KEY = {
    'player': 'players',
    'goalkeeper': 'goalkeepers',
    'referee': 'referees',
    'ball': 'ball',
}
TRACK_KEYS = list(TRACK_KEY.values())
CLASS_ID_TO_KEY = {CLASS_IDS[name]: key for name, key in TRACK_KEY.items()}


class FootballTracker:
    """
    Comprehensive football video analysis tracker based on ByteTrack
    Tracks players, goalkeepers, referees and the ball as distinct classes
    """

    def __init__(self, model_path='yolov8s.pt',
                use_roboflow=False,
                api_key=None,
                persist_cache=True,
                cache_dir='tracker_cache',
                confidence=0.25,
                imgsz=960,
                max_ball_gap=15,
                detector=None):
        """
        Initialize the football tracker

        Args:
            model_path: Path to the local YOLO model (the production detector)
            use_roboflow: Opt in to the hosted Roboflow detector (labelling/benchmark only)
            api_key: Roboflow API key; falls back to ROBOFLOW_API_KEY
            persist_cache: Whether to save cache to disk
            cache_dir: Directory for cache files
            confidence: Detection confidence threshold
            imgsz: Inference image size for the local detector
            max_ball_gap: How many consecutive frames the last known ball box may
                be held for while the ball is undetected. After that the ball is
                reported as unknown rather than frozen in place.
            detector: Pre-built detector to use instead of constructing one.
                Only for tests, which must not load real YOLO weights.
        """
        self.model_path = model_path
        self.max_ball_gap = max_ball_gap
        self.use_roboflow = use_roboflow
        self.api_key = api_key
        self.persist_cache = persist_cache
        self.cache_dir = Path(cache_dir)

        if persist_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.detector = detector if detector is not None else create_detector(
            model_path=model_path,
            use_roboflow=use_roboflow,
            api_key=api_key,
            confidence=confidence,
            imgsz=imgsz,
        )

        # Initialize tracker using supervision
        self.tracker = sv.ByteTrack()

        # Populated by get_object_tracks(); read by the pipeline's final report.
        self.tracks = None

        # Color map for visualization
        self.colors = {
            'player': (0, 255, 0),      # Green
            'goalkeeper': (0, 165, 255),  # Orange
            'ball': (0, 0, 255),        # Red
            'referee': (255, 255, 0),   # Yellow
            'team1': (255, 50, 50),     # Light red
            'team2': (50, 50, 255)      # Light blue
        }
        
    def add_position_to_tracks(self, tracks):
        """
        Add position information to tracked objects
        
        Args:
            tracks: Dictionary of tracked objects
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_type == 'ball':
                        # For ball, use center point
                        position = get_center_of_bbox(bbox)
                    else:
                        # For players and referees, use foot position (bottom center)
                        position = get_foot_position(bbox)
                    
                    tracks[object_type][frame_num][track_id]['position'] = position
    
    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolate missing ball positions
        
        Args:
            ball_positions: List of ball position dictionaries
            
        Returns:
            List of interpolated ball positions
        """
        # Extract ball bboxes
        ball_bboxes = []
        for frame_dict in ball_positions:
            bbox = None
            # Get the first ball entry (usually just one with ID 1)
            for ball_id, ball_info in frame_dict.items():
                bbox = ball_info.get('bbox', [])
                break
            ball_bboxes.append(bbox if bbox else [0, 0, 0, 0])
        
        # Create DataFrame for easier interpolation
        df_ball = pd.DataFrame(ball_bboxes, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Interpolate missing values
        df_ball = df_ball.interpolate(method='linear')
        df_ball = df_ball.fillna(method='bfill')  # Backward fill any remaining NaNs
        df_ball = df_ball.fillna(method='ffill')  # Forward fill any remaining NaNs
        
        # Convert back to original format
        interpolated_positions = []
        for i, bbox in enumerate(df_ball.values.tolist()):
            ball_id = list(ball_positions[i].keys())[0] if ball_positions[i] else 1
            interpolated_positions.append({ball_id: {"bbox": bbox}})
            
        return interpolated_positions
        
    def detect_objects_in_frames(self, frames):
        """
        Detect objects (players, ball, referees) in video frames
        
        Args:
            frames: List of video frames
            
        Returns:
            List of detections per frame
        """
        all_detections = []
        batch_size = 10  # Process in smaller batches to avoid memory issues
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_detections = []
            
            for frame_idx, frame in enumerate(batch_frames):
                # Get absolute frame index
                abs_idx = i + frame_idx
                detections = self.detector.detect(frame, abs_idx)
                batch_detections.append(detections)
                
                if abs_idx % 10 == 0:
                    print(f"Detected objects in frame {abs_idx}, found {len(detections)} objects")
            
            all_detections.extend(batch_detections)
            
        return all_detections
    
    def get_object_tracks(self, frames, read_from_cache=True, cache_path=None):
        """
        Get tracked objects from video frames
        
        Args:
            frames: List of video frames
            read_from_cache: Whether to read from cache if available
            cache_path: Path to cache file
            
        Returns:
            Dictionary of tracked objects
        """
        # Check cache first
        if read_from_cache and cache_path and os.path.exists(cache_path):
            print(f"Loading tracks from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.tracks = pickle.load(f)
            return self.tracks
        
        # Initialize tracks structure -- one list per detector class
        tracks = {key: [] for key in TRACK_KEYS}

        # Get detections for all frames
        detections_list = self.detect_objects_in_frames(frames)

        frames_since_ball = 0

        # Process each frame
        for frame_idx, frame_detections in enumerate(detections_list):
            # Prepare detection objects for supervision ByteTrack
            boxes = []
            class_ids = []
            confidences = []

            for det in frame_detections:
                class_name = det.get('class')
                if class_name not in CLASS_IDS:
                    continue  # detector already normalised; anything else is noise
                boxes.append(det['bbox'])
                class_ids.append(CLASS_IDS[class_name])
                confidences.append(det.get('confidence', 0.5))

            # Start this frame's slot for every class
            for key in TRACK_KEYS:
                tracks[key].append({})

            if boxes:
                detections = sv.Detections(
                    xyxy=np.array(boxes),
                    class_id=np.array(class_ids),
                    confidence=np.array(confidences)
                )

                # Update tracker
                tracked_detections = self.tracker.update_with_detections(detections)

                for i in range(len(tracked_detections.xyxy)):
                    key = CLASS_ID_TO_KEY.get(int(tracked_detections.class_id[i]))
                    if key is None:
                        continue
                    track_id = int(tracked_detections.tracker_id[i])
                    tracks[key][frame_idx][track_id] = {
                        "bbox": tracked_detections.xyxy[i].tolist(),
                        "confidence": float(tracked_detections.confidence[i]),
                    }

            # The ball moves erratically and ByteTrack often drops it, so take the
            # raw detection directly. ID 1 keeps it stable for downstream code.
            ball_found = False
            for det in frame_detections:
                if det.get('class') == 'ball':
                    tracks["ball"][frame_idx][1] = {
                        "bbox": det['bbox'],
                        "confidence": det.get('confidence', 0.5),
                    }
                    ball_found = True
                    break

            if ball_found:
                frames_since_ball = 0
            else:
                frames_since_ball += 1
                # Hold the last known box for a bounded number of frames only.
                # Copying it indefinitely used to freeze a stale ball on screen
                # for the rest of the video and corrupt possession statistics.
                if 0 < frames_since_ball <= self.max_ball_gap and frame_idx > 0:
                    for ball_id, ball_info in tracks["ball"][frame_idx - 1].items():
                        held = ball_info.copy()
                        held['held_for'] = frames_since_ball
                        tracks["ball"][frame_idx][ball_id] = held
                # Beyond max_ball_gap the ball is simply unknown: leave it empty.

        # Save to cache if requested
        if self.persist_cache and cache_path:
            print(f"Saving tracks to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(tracks, f)

        # Cached so generate_final_report() can write player_statistics.json.
        self.tracks = tracks
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draw ellipse under player/referee feet
        
        Args:
            frame: Frame to draw on
            bbox: Bounding box [x1, y1, x2, y2]
            color: Color tuple (B, G, R)
            track_id: Optional tracking ID to display
            
        Returns:
            Annotated frame
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw a more prominent ellipse
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=3,  # Increased thickness
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width//2
            x2_rect = x_center + rectangle_width//2
            y1_rect = (y2- rectangle_height//2) + 15
            y2_rect = (y2 + rectangle_height//2) + 15

            # Draw a bigger ID box
            cv2.rectangle(frame,
                        (int(x1_rect), int(y1_rect)),
                        (int(x2_rect), int(y2_rect)),
                        color,
                        cv2.FILLED)
                        
            # Add a black outline to make it more visible
            cv2.rectangle(frame,
                        (int(x1_rect), int(y1_rect)),
                        (int(x2_rect), int(y2_rect)),
                        (0, 0, 0),
                        1)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
                
            # Draw a white background for the player ID text for better visibility
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Slightly larger font
                (255, 255, 255),  # White outline
                3     # Thicker outline
            )
            
            # Draw the player ID in a contrasting color on top
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Slightly larger font
                (0, 0, 0),  # Black text
                1     # Regular thickness
            )

        return frame
    
    def draw_triangle(self, frame, bbox, color):
        """
        Draw triangle for ball
        
        Args:
            frame: Frame to draw on
            bbox: Bounding box [x1, y1, x2, y2]
            color: Color tuple (B, G, R)
            
        Returns:
            Annotated frame
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        # Make the triangle larger and more visible
        triangle_points = np.array([
            [x, y],
            [x-15, y-25],  # Increased size
            [x+15, y-25],  # Increased size
        ], np.int32)
        
        # Draw a filled triangle with the color
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        
        # Draw a slightly thicker black outline
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draw team ball control statistics
        
        Args:
            frame: Frame to draw on
            frame_num: Current frame number
            team_ball_control: Array of team IDs controlling ball
            
        Returns:
            Annotated frame
        """
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of times each team had ball control
        team_1_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        total_frames = team_1_frames + team_2_frames
        
        if total_frames > 0:
            team_1_pct = team_1_frames / total_frames
            team_2_pct = team_2_frames / total_frames
        else:
            team_1_pct = team_2_pct = 0.0

        cv2.putText(frame, f"Team 1 Ball Control: {team_1_pct*100:.2f}%",
                  (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2_pct*100:.2f}%",
                  (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame
    
    def draw_annotations(self, video_frames, tracks, team_ball_control=None):
        """
        Draw all annotations on video frames
        
        Args:
            video_frames: List of video frames
            tracks: Dictionary of tracked objects
            team_ball_control: Array of team IDs controlling ball
            
        Returns:
            List of annotated frames
        """
        output_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            # Add detection counts at the top
            player_count = len(tracks["players"][frame_num])
            keeper_count = len(tracks["goalkeepers"][frame_num])
            referee_count = len(tracks["referees"][frame_num])
            ball_count = len(tracks["ball"][frame_num])

            # Draw info bar
            cv2.rectangle(frame, (0, 0), (400, 105), (0, 0, 0), -1)
            cv2.putText(frame, f"Players: {player_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Goalkeepers: {keeper_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, self.colors['goalkeeper'], 2)
            cv2.putText(frame, f"Referees: {referee_count}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Ball: {ball_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

            # Draw players
            player_dict = tracks["players"][frame_num]
            for track_id, player in player_dict.items():
                # Get team color if available
                color = player.get("team_color", self.colors['player'])
                # Draw a more prominent bounding box
                bbox = player["bbox"]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                
                # Highlight player with ball
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))
            
            # Draw goalkeepers -- kept visually distinct from outfield players
            # because their kit deliberately differs from their own team's.
            for track_id, keeper in tracks["goalkeepers"][frame_num].items():
                bbox = keeper["bbox"]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                              self.colors['goalkeeper'], 2)
                frame = self.draw_ellipse(frame, bbox, self.colors['goalkeeper'], track_id)
                if keeper.get('has_ball', False):
                    frame = self.draw_triangle(frame, bbox, (0, 0, 255))

            # Draw referees
            referee_dict = tracks["referees"][frame_num]
            for track_id, referee in referee_dict.items():
                bbox = referee["bbox"]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 
                              self.colors['referee'], 2)
                frame = self.draw_ellipse(frame, referee["bbox"], self.colors['referee'], track_id)
            
            # Draw ball
            ball_dict = tracks["ball"][frame_num]
            for track_id, ball in ball_dict.items():
                bbox = ball["bbox"]
                # Draw a circle for ball for better visibility
                center = get_center_of_bbox(bbox)
                radius = max(5, int((bbox[2] - bbox[0]) / 2))
                cv2.circle(frame, (center[0], center[1]), radius, self.colors['ball'], -1)
                frame = self.draw_triangle(frame, ball["bbox"], self.colors['ball'])
            
            # Draw team ball control stats if available
            if team_ball_control is not None:
                frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            output_frames.append(frame)
        
        return output_frames
    
    # Utility functions
    # Helper functions are now imported from bbox_utils.py
