"""
core/pipeline.py
Main pipeline integrating all components
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import Optional, Dict, List
from tqdm import tqdm

from config import Config
from hybrid_assigner import HybridTeamAssigner
from jersey_ocr import PlayerIDManager
from event_detector import EventDetector
from highlight_generator import HighlightGenerator


class FootballAnalysisPipeline:
    """
    Main pipeline integrating:
    - Detection & Tracking (EyeCU4.0)
    - Hybrid Team Assignment (Hamza + EyeCU4.0)
    - Jersey OCR for Player IDs
    - Event Detection
    - Highlight Generation
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Initialize components
        self.team_assigner = HybridTeamAssigner(
            hamza_weight=self.config.team_assignment.hamza_weight,
            eyecu_weight=self.config.team_assignment.eyecu_weight,
            confidence_threshold=self.config.team_assignment.confidence_threshold
        )
        
        self.player_id_manager = PlayerIDManager(
            ocr_engine=self.config.ocr.ocr_engine,
            ocr_confidence=self.config.ocr.confidence_threshold
        )
        
        self.event_detector = EventDetector(fps=self.config.video.output_fps)
        
        self.highlight_generator = HighlightGenerator(
            pre_buffer=self.config.highlight.pre_event_buffer,
            post_buffer=self.config.highlight.post_event_buffer,
            min_duration=self.config.highlight.min_highlight_duration,
            max_duration=self.config.highlight.max_highlight_duration
        )
        
        # Placeholder for detector/tracker (to be integrated with EyeCU4.0)
        self.detector = None
        self.tracker = None
        
        # Processing state
        self.frames_processed = 0
        self.total_frames = 0
        self.video_path = None
        self.output_frames = []
    
    def initialize_detector_tracker(self):
        """Initialize YOLOv8 detector and tracker (EyeCU4.0 integration)"""
        # This should integrate with EyeCU4.0's detector and tracker
        # For MVP, this is a placeholder
        try:
            from ultralytics import YOLO
            self.detector = YOLO(self.config.detection.yolo_model)
        except ImportError:
            print("Warning: ultralytics not installed. Using mock detector.")
            self.detector = None
    
    def detect_and_track(self, frame: np.ndarray, frame_num: int) -> Dict:
        """
        Run detection and tracking on a frame
        
        Returns dict with:
            - players: List of player detections
            - ball: Ball detection
            - referees: Referee detections
        """
        # This should call EyeCU4.0's detection and tracking
        # Placeholder implementation
        
        if self.detector is None:
            return {'players': [], 'ball': None, 'referees': []}
        
        # Run YOLO detection
        results = self.detector(frame, verbose=False)
        
        detections = {
            'players': [],
            'ball': None,
            'referees': []
        }
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                x1, y1, x2, y2 = map(int, xyxy)
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_id': cls
                }
                
                if cls == self.config.detection.player_class_id:
                    detections['players'].append(detection)
                elif cls == self.config.detection.ball_class_id:
                    detections['ball'] = detection
                elif cls == self.config.detection.referee_class_id:
                    detections['referees'].append(detection)
        
        # TODO: Add tracking to maintain IDs across frames
        # This should use ByteTrack or similar from EyeCU4.0
        
        return detections
    
    def process_frame(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        """Process a single frame through the entire pipeline"""
        
        # Step 1: Detection and Tracking
        detections = self.detect_and_track(frame, frame_num)
        
        # Step 2: Process each player
        player_data = []
        
        for idx, player_det in enumerate(detections['players']):
            x1, y1, x2, y2 = player_det['bbox']
            
            # Extract player crop
            player_crop = frame[y1:y2, x1:x2]
            
            if player_crop.size == 0:
                continue
            
            # Assign track ID (placeholder - should come from tracker)
            track_id = idx  # This should be from actual tracker
            
            # Step 3: Team Assignment (Hybrid)
            team_id, team_conf = self.team_assigner.assign_team(
                player_crop, 
                track_id, 
                frame_num
            )
            
            # Step 4: Player ID via OCR
            player_id = self.player_id_manager.process_detection(
                track_id, 
                player_crop, 
                team_id, 
                frame_num
            )
            
            # Calculate position (pixel to field coordinates)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            position = self.pixel_to_field_coords(center_x, center_y, frame.shape)
            
            player_data.append({
                'track_id': track_id,
                'player_id': player_id,
                'team_id': team_id,
                'team_confidence': team_conf,
                'bbox': [x1, y1, x2, y2],
                'position': position
            })
        
        # Step 5: Event Detection
        ball_pos = None
        if detections['ball']:
            ball_bbox = detections['ball']['bbox']
            ball_x = (ball_bbox[0] + ball_bbox[2]) / 2
            ball_y = (ball_bbox[1] + ball_bbox[3]) / 2
            ball_pos = self.pixel_to_field_coords(ball_x, ball_y, frame.shape)
        
        events = self.event_detector.process_frame(frame_num, ball_pos, player_data)
        
        # Step 6: Visualization
        annotated_frame = self.draw_annotations(
            frame, 
            player_data, 
            detections['ball'],
            events
        )
        
        return annotated_frame
    
    def pixel_to_field_coords(self, x: float, y: float, frame_shape) -> tuple:
        """Convert pixel coordinates to field coordinates (meters)"""
        # Simplified conversion - should use homography for accuracy
        h, w = frame_shape[:2]
        
        # Normalize to [-1, 1]
        norm_x = (x / w) * 2 - 1
        norm_y = (y / h) * 2 - 1
        
        # Scale to field dimensions
        field_x = norm_x * (self.event_detector.field_width / 2)
        field_y = norm_y * (self.event_detector.field_height / 2)
        
        return (field_x, field_y)
    
    def draw_annotations(self, frame: np.ndarray, player_data: List[Dict],
                        ball_detection: Optional[Dict], events: List) -> np.ndarray:
        """Draw all annotations on frame"""
        annotated = frame.copy()
        
        # Draw players
        for player in player_data:
            x1, y1, x2, y2 = player['bbox']
            team_id = player['team_id']
            
            # Choose color based on team
            if team_id == 0:
                color = self.config.visualization.team_a_color
            elif team_id == 1:
                color = self.config.visualization.team_b_color
            else:
                color = (128, 128, 128)  # Gray for unknown
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 
                         self.config.visualization.bbox_thickness)
            
            # Draw player ID
            if player['player_id'] and self.config.visualization.show_player_ids:
                label = f"{player['player_id']}"
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           self.config.visualization.text_scale,
                           color,
                           self.config.visualization.text_thickness)
        
        # Draw ball
        if ball_detection:
            x1, y1, x2, y2 = ball_detection['bbox']
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(annotated, center, 10, 
                      self.config.visualization.ball_color, -1)
        
        # Draw events
        if events and self.config.visualization.show_events:
            for event in events:
                label = event.event_type.upper()
                cv2.putText(annotated, label, (50, 50),
                           cv2.FONT_HERSHEY_BOLD, 1.5,
                           (0, 255, 255), 3)
        
        return annotated
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Process entire video through pipeline
        
        Returns:
            Dictionary with processing statistics and results
        """
        self.video_path = video_path
        
        # Initialize detector/tracker
        self.initialize_detector_tracker()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Update event detector FPS
        self.event_detector.fps = fps
        
        # Setup output video writer
        if output_path is None:
            output_path = str(Path(self.config.output.output_dir) / 
                            self.config.output.tracked_video_dir / 
                            "tracked_output.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 
                             self.config.video.output_fps, 
                             (width, height))
        
        # Initialize team assignment with first frames
        initialization_crops = []
        initialization_positions = []
        
        frame_num = 0
        pbar = tqdm(total=self.total_frames, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if configured
            if frame_num % self.config.video.skip_frames != 0:
                frame_num += 1
                continue
            
            # Limit frames if configured
            if (self.config.video.max_frames and 
                self.frames_processed >= self.config.video.max_frames):
                break
            
            # Collect initialization data from first frames
            if frame_num < 30:
                detections = self.detect_and_track(frame, frame_num)
                for player_det in detections['players']:
                    x1, y1, x2, y2 = player_det['bbox']
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        initialization_crops.append(crop)
                        center_x = (x1 + x2) / 2
                        pos = self.pixel_to_field_coords(center_x, 0, frame.shape)
                        initialization_positions.append((crop, pos))
            
            # Initialize team assigner after collecting data
            if frame_num == 30 and initialization_crops:
                self.team_assigner.initialize(
                    initialization_crops,
                    initialization_positions
                )
            
            # Process frame
            processed_frame = self.process_frame(frame, frame_num)
            
            # Write output
            out.write(processed_frame)
            
            # Display if configured
            if self.config.video.display_realtime:
                cv2.imshow('Processing', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            self.frames_processed += 1
            frame_num += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        if self.config.video.display_realtime:
            cv2.destroyAllWindows()
        
        # Generate highlights
        print("\nGenerating highlights...")
        events = self.event_detector.get_all_events()
        highlight_dir = str(Path(self.config.output.output_dir) / 
                          self.config.output.highlights_dir)
        
        highlight_clips = self.highlight_generator.generate_highlights(
            output_path,
            events,
            highlight_dir,
            int(fps),
            self.total_frames
        )
        
        # Generate reports
        print("Generating reports...")
        self.generate_reports()
        
        # Return statistics
        return {
            'frames_processed': self.frames_processed,
            'total_frames': self.total_frames,
            'output_video': output_path,
            'highlights_generated': len(highlight_clips),
            'events_detected': len(events),
            'team_stats': self.team_assigner.get_team_stats(),
            'player_stats': self.player_id_manager.get_statistics(),
            'event_summary': self.event_detector.get_event_summary()
        }
    
    def generate_reports(self):
        """Generate all output reports"""
        report_dir = Path(self.config.output.output_dir) / self.config.output.reports_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Event report
        events = self.event_detector.get_all_events()
        event_report = {
            'events': [e.to_dict() for e in events],
            'summary': self.event_detector.get_event_summary()
        }
        
        with open(report_dir / 'events.json', 'w') as f:
            json.dump(event_report, f, indent=2)
        
        # Team assignment report
        team_report = self.team_assigner.get_team_stats()
        with open(report_dir / 'teams.json', 'w') as f:
            json.dump(team_report, f, indent=2)
        
        # Player ID report
        player_report = self.player_id_manager.get_statistics()
        player_report['players'] = {
            pid: self.player_id_manager.get_player_info(pid)
            for pid in self.player_id_manager.get_all_players()
        }
        
        with open(report_dir / 'players.json', 'w') as f:
            json.dump(player_report, f, indent=2, default=str)
        
        # Final summary
        summary = {
            'frames_processed': self.frames_processed,
            'video_path': self.video_path,
            'team_stats': team_report,
            'player_stats': player_report,
            'event_summary': self.event_detector.get_event_summary()
        }
        
        with open(report_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Reports saved to {report_dir}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.config.video.display_realtime:
            cv2.destroyAllWindows()