"""
Complete Integrated Pipeline for Football Player Analysis
This MVP ties together all modules into a working system
Using advanced tracking, team assignment, and speed estimation
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time

# Import original modules
from player_detection import PlayerDetector, VideoProcessor
from face_body_crop import FaceBodyExtractor
from mesh_reconstruction import PoseEstimator3D, MeshSimilarityCalculator
from player_tracking import PlayerTracker as LegacyPlayerTracker
from id_loss_handler import ReIdentificationSystem, InterruptionDetector
from db_handler import MatchRecorder
from eval_benchmark import TrackingEvaluator, PerformanceAnalyzer

# Import new advanced tracking modules
from trackers.football_tracker import FootballTracker
from trackers.team_assigner import TeamAssigner
from trackers.camera_movement import CameraMovementEstimator
from trackers.speed_distance import SpeedDistanceEstimator
from trackers.player_ball_assigner import PlayerBallAssigner
from trackers.video_utils import read_video, save_video


class FootballAnalysisPipeline:
    """
    Complete pipeline for football player analysis
    Integrates detection, tracking, re-identification, and database management
    """
    
    def __init__(self, 
                 yolo_model: str = 'yolov8x.pt',
                 output_dir: str = 'pipeline_output',
                 match_id: int = 1,
                 use_advanced_tracking: bool = True,
                 use_roboflow: bool = False,
                 api_key: Optional[str] = None,
                 use_cache: bool = True,
                 show_speed: bool = False,
                 show_distance: bool = False):
        """
        Initialize complete pipeline
        Args:
            yolo_model: Path to YOLO weights
            output_dir: Output directory for all results
            match_id: Unique match identifier
            use_advanced_tracking: Whether to use the advanced tracking system
            use_roboflow: Whether to use Roboflow API
            api_key: Roboflow API key
            use_cache: Whether to use cached results
            show_speed: Whether to display player speed
            show_distance: Whether to display player distance covered
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.match_id = match_id
        self.use_advanced_tracking = use_advanced_tracking
        self.use_roboflow = use_roboflow
        self.api_key = api_key
        self.use_cache = use_cache
        self.show_speed = show_speed
        self.show_distance = show_distance
        
        print("Initializing Football Analysis Pipeline...")
        
        # Set up tracker system
        if use_advanced_tracking:
            print("  - Using advanced tracking system...")
            
            # Create cache directory
            cache_dir = self.output_dir / 'cache'
            cache_dir.mkdir(exist_ok=True)
            
            # Initialize components
            self.adv_tracker = FootballTracker(
                model_path=yolo_model,
                use_roboflow=use_roboflow,
                api_key=api_key,
                persist_cache=use_cache,
                cache_dir=str(cache_dir)
            )
            
            self.team_assigner = TeamAssigner(num_teams=2)
            self.ball_assigner = PlayerBallAssigner(max_distance=70)
        else:
            print("  - Using legacy tracking system...")
            
            # Module 1: Detection
            print("  - Loading YOLO detector...")
            self.detector = PlayerDetector(yolo_model)
            
            # Module 2: Face/Body extraction
            print("  - Initializing face/body extractor...")
            self.face_body_extractor = FaceBodyExtractor(str(self.output_dir))
            
            # Module 3: 3D Pose/Mesh
            print("  - Initializing pose estimator...")
            self.pose_estimator = PoseEstimator3D(str(self.output_dir))
            self.mesh_calculator = MeshSimilarityCalculator()
            
            # Module 4: Tracking
            print("  - Setting up tracker...")
            self.tracker = LegacyPlayerTracker(max_age=30, min_hits=3, iou_threshold=0.3)
            
            # Module 5-6: Re-identification
            print("  - Initializing re-ID system...")
            self.reid_system = ReIdentificationSystem(
                mesh_weight=0.4,
                jersey_weight=0.4,
                proportion_weight=0.2,
                reid_threshold=0.7
            )
            self.interruption_detector = InterruptionDetector()
        
        # Module 7: Database
        print("  - Setting up database...")
        self.recorder = MatchRecorder(match_id, str(self.output_dir))
        
        # Module 8: Evaluation
        self.evaluator = TrackingEvaluator()
        self.analyzer = PerformanceAnalyzer(str(self.output_dir / 'evaluation'))
        
        # Pipeline state
        self.frame_count = 0
        self.processing_times = []
        self.annotated_frames = []
        
        print("Pipeline initialization complete!\n")
    
    def process_frame(self, frame: np.ndarray, frame_id: int,
                     save_visualizations: bool = True) -> Dict:
        """
        Process a single frame through the entire pipeline
        Args:
            frame: Input frame (BGR)
            frame_id: Frame number
            save_visualizations: Whether to save annotated frames
        Returns:
            Dictionary with all processing results
        """
        start_time = time.time()
        results = {'frame_id': frame_id, 'detections': []}
        
        if self.use_advanced_tracking:
            # Advanced tracking processes all frames at once, 
            # so this function just returns dummy results for compatibility
            results['detections'] = [{
                'player_id': 0,
                'tracking_id': 0,
                'bbox': [0, 0, 0, 0],
                'jersey_number': '',
                'is_reassigned': False,
                'crop_paths': {'body': None, 'face': None}
            }]
            
            # Return immediately, actual processing happens in process_video
            if save_visualizations:
                self.annotated_frames.append(frame.copy())
                
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            results['processing_time'] = processing_time
            
            return results
        else:
            # Legacy pipeline processing
            
            # STEP 1: Detect players
            detections = self._detect_players(frame, frame_id)
            
            # STEP 2: Track players across frames
            tracked_objects = self._track_players(detections)
            
            # STEP 3: Detect tracking interruptions
            interruptions = self._detect_interruptions(tracked_objects, frame_id)
            
            # STEP 4: Process each tracked player
            for tracked in tracked_objects:
                player_data = self._process_tracked_player(
                    frame, tracked, frame_id
                )
                
                if player_data:
                    results['detections'].append(player_data)
            
            # STEP 5: Visualize results
            if save_visualizations:
                annotated = self._visualize_results(frame, results['detections'])
                self.annotated_frames.append(annotated)
            
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            results['processing_time'] = processing_time
            
            return results
    
    def _detect_players(self, frame: np.ndarray, frame_id: int) -> List[Dict]:
        """Step 1: Detect players using YOLO"""
        # Use the PlayerDetector to detect players in the frame
        detections = self.detector.process_frame(frame, frame_id)
        
        # Process each detection to ensure it has all required fields
        processed_detections = []
        for detection in detections:
            # Make sure bbox exists and has the right format
            if 'bbox' not in detection:
                continue
                
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract crop from the frame
            if 'crop' not in detection or detection['crop'] is None or detection['crop'].size == 0:
                # Create crop if missing or empty
                try:
                    crop = frame[y1:y2, x1:x2].copy()
                    detection['crop'] = crop
                except:
                    # Skip if crop can't be extracted
                    continue
            
            # Calculate center point if missing
            if 'center' not in detection:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                detection['center'] = [center_x, center_y]
            
            # Ensure frame_id is set
            detection['frame_id'] = frame_id
            
            processed_detections.append(detection)
        
        return processed_detections
    
    def _track_players(self, detections: List[Dict]) -> List[Dict]:
        """Step 2: Track players across frames"""
        # Update tracker with new detections
        tracked_objects = self.tracker.update(detections)
        
        # Add original detection data to tracked objects for reference
        for tracked in tracked_objects:
            # Find the original detection data for this tracked object
            for det in detections:
                # Compare bounding boxes to match tracked object with original detection
                if np.allclose(tracked['bbox'], det['bbox']):
                    tracked['detection_data'] = det
                    tracked['confidence'] = det['confidence']  # Preserve confidence from detection
                    break
        
        return tracked_objects
    
    def _detect_interruptions(self, tracked_objects: List[Dict],
                             frame_id: int) -> List[tuple]:
        """Step 3: Detect tracking interruptions"""
        # Get current tracking IDs from tracked objects
        current_ids = [obj['tracking_id'] for obj in tracked_objects]
        
        # Initialize list for events
        events = []
        
        # Check for lost tracks in the tracker's history
        for track_id, track_data in self.tracker.track_history.items():
            # Skip active tracks
            if track_data['status'] == 'active':
                continue
            
            # If the track was active but is now lost and not in current_ids
            if track_data['status'] == 'lost' and track_id not in current_ids:
                # Check if this is a new lost event (last seen in recent frames)
                if frame_id - track_data['last_seen'] <= 5:  # Within 5 frames
                    events.append(('lost', track_id))
        
        # Handle lost tracks
        for event_type, track_id in events:
            if event_type == 'lost':
                self.reid_system.handle_tracking_interruption(track_id)

        return events
    
    def _process_tracked_player(self, frame: np.ndarray,
                                tracked_obj: Dict, 
                                frame_id: int) -> Optional[Dict]:
        """Step 4: Complete processing for a tracked player"""
        tracking_id = tracked_obj['tracking_id']
        det_data = tracked_obj.get('detection_data', {})
        player_crop = det_data.get('crop')
        
        if player_crop is None or player_crop.size == 0:
            return None
        
        # Extract face and body crops
        # crop_paths = self.face_body_extractor.save_crops(
        #     player_crop, tracking_id, frame_id
        # )
        crop_paths = {'body': None, 'face': None}
        
        # Extract 3D pose/mesh features
        # pose_data = self.pose_estimator.process_player_detection(
        #     player_crop, tracking_id, frame_id, 
        #     save_data=True, save_visualization=False
        # )
        pose_data = {
            'feature_vector': np.random.randn(128),
            'body_proportions': {
                'shoulder_to_torso_ratio': 0.45,
                'hip_to_torso_ratio': 0.38
            }
        }
        
        # Re-identification and player ID assignment
        reid_data = {
            'mesh_feature': pose_data.get('feature_vector', np.array([])),
            'jersey_number': det_data.get('jersey_number', ''),
            'jersey_confidence': det_data.get('jersey_confidence', 0.0),
            'body_proportions': pose_data.get('body_proportions', {})
        }
        
        # player_id, is_reassigned = self.reid_system.process_detection(
        #     tracking_id, reid_data, frame_id
        # )
        player_id = tracking_id  # Simplified for demo
        is_reassigned = False
        
        # Record to database
        detection_data = {
            'bbox': tracked_obj['bbox'],
            'confidence': tracked_obj['confidence'],
            'mesh_feature': reid_data['mesh_feature'],
            'body_crop': player_crop,
            'face_crop': None
        }
        
        # self.recorder.record_detection(
        #     player_id, tracking_id, frame_id, detection_data
        # )
        
        return {
            'player_id': player_id,
            'tracking_id': tracking_id,
            'bbox': tracked_obj['bbox'],
            'jersey_number': det_data.get('jersey_number', ''),
            'is_reassigned': is_reassigned,
            'crop_paths': crop_paths
        }
    
    def _visualize_results(self, frame: np.ndarray,
                          detections: List[Dict]) -> np.ndarray:
        """Step 5: Visualize tracking and re-ID results"""
        # First use the PlayerTracker to visualize the tracking information
        tracked_objects = []
        for det in detections:
            if 'tracking_id' in det and 'bbox' in det:
                tracked_objects.append({
                    'tracking_id': det['tracking_id'],
                    'bbox': det['bbox']
                })
        
        # Let the tracker visualize the tracks with motion trails
        vis = self.tracker.visualize_tracks(frame, tracked_objects, show_ids=True, show_trails=True)
        
        # Now add our additional player information (player IDs, jersey numbers, etc.)
        for det in detections:
            bbox = det['bbox']
            player_id = det.get('player_id', 'N/A')
            jersey = det.get('jersey_number', '')
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on player ID
            if player_id != 'N/A':
                np.random.seed(player_id)
                color = tuple(np.random.randint(0, 255, 3).tolist())
            else:
                color = (255, 255, 255)  # White for unassigned
            
            # Draw labels
            label = f"P{player_id}"
            if jersey:
                label += f" #{jersey}"
            if det.get('is_reassigned'):
                label += " (Re-ID)"
            
            cv2.putText(vis, label, (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add frame info
        info = f"Frame: {self.frame_count} | Players: {len(detections)}"
        cv2.putText(vis, info, (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis
    
    def process_video(self, video_path: str,
                     skip_frames: int = 1,
                     max_frames: Optional[int] = None,
                     display_results: bool = True) -> Dict:
        """
        Process complete video
        Args:
            video_path: Path to input video
            skip_frames: Process every nth frame
            max_frames: Maximum frames to process
            display_results: Show real-time visualization
        Returns:
            Processing statistics
        """
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"\nError: Video file not found: {video_path}")
            print("Please place a video file in the input-videos directory.")
            return {"error": "Video file not found"}
        
        # Use advanced tracking pipeline if enabled
        if self.use_advanced_tracking:
            return self._process_video_advanced(
                video_path, skip_frames, max_frames, display_results
            )
        else:
            return self._process_video_legacy(
                video_path, skip_frames, max_frames, display_results
            )
    
    def _process_video_advanced(self, video_path: str,
                               skip_frames: int = 1,
                               max_frames: Optional[int] = None,
                               display_results: bool = True) -> Dict:
        """Process video using advanced tracking system"""
        start_time = time.time()
        
        print(f"\nProcessing video: {video_path}")
        print(f"Processing every {skip_frames} frame(s)\n")
        
        # Create cache directory
        cache_dir = self.output_dir / 'cache'
        cache_dir.mkdir(exist_ok=True)
        
        # 1. Load video frames
        print("Loading video frames...")
        frames = read_video(
            video_path,
            max_frames=max_frames,
            start_frame=0  # Start from beginning
        )
        
        if skip_frames > 1:
            # Skip frames if needed
            frames = frames[::skip_frames]
            print(f"Skipped frames, now processing {len(frames)} frames")
        
        # Update frame count
        self.frame_count = len(frames)
        
        # 2. Get object tracks
        print("Detecting and tracking objects...")
        cache_path = str(cache_dir / 'tracks.pkl')
        tracks = self.adv_tracker.get_object_tracks(
            frames,
            read_from_cache=self.use_cache,
            cache_path=cache_path
        )
        
        # 3. Add position information to tracks
        print("Adding position information...")
        self.adv_tracker.add_position_to_tracks(tracks)
        
        # 4. Interpolate ball positions
        print("Interpolating ball positions...")
        tracks["ball"] = self.adv_tracker.interpolate_ball_positions(tracks["ball"])
        
        # 5. Estimate camera movement
        print("Estimating camera movement...")
        camera_estimator = CameraMovementEstimator(frames[0])
        camera_cache = str(cache_dir / 'camera_movement.pkl')
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
        fps = 30  # Assume 30fps if not known
        speed_estimator = SpeedDistanceEstimator(frame_rate=fps)
        speed_estimator.add_speed_and_distance_to_tracks(tracks)
        
        # 9. Determine ball possession and team control
        print("Analyzing ball possession...")
        team_ball_control = self.ball_assigner.compute_team_ball_control(tracks)
        
        # 10. Generate outputs
        print("Generating visualization...")
        
        # Force clear the cache to ensure fresh detections are visualized
        if self.adv_tracker.detector:
            self.adv_tracker.detector.detections_cache = {}
            
        # Make extra sure we're getting fresh detections
        for frame_idx, frame in enumerate(frames):
            if frame_idx % 20 == 0:
                # Detect every 20th frame to update cache with new detections
                _ = self.adv_tracker.detector.detect(frame, frame_idx)
                print(f"Refreshing detections for frame {frame_idx}")
        
        # Draw annotations with improved visualization
        output_frames = self.adv_tracker.draw_annotations(
            frames, tracks, team_ball_control)
        
        # Add camera movement indicators
        output_frames = camera_estimator.draw_camera_movement(
            output_frames, camera_movement)
        
        # Add speed and distance (only if enabled)
        output_frames = speed_estimator.draw_speed_and_distance(
            output_frames, 
            tracks,
            show_speed=self.show_speed,
            show_distance=self.show_distance,
            compact_display=True
        )
        
        # Store annotated frames
        self.annotated_frames = output_frames
        
        # Display results if requested
        if display_results:
            for i, frame in enumerate(output_frames):
                try:
                    cv2.imshow('Football Analysis', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopping (user interrupt)...")
                        break
                    
                    # Progress update
                    if i % 10 == 0:
                        progress = i / len(output_frames) * 100
                        print(f"Displaying: {i}/{len(output_frames)} ({progress:.1f}%)")
                except cv2.error as e:
                    # Silently continue if display is not supported
                    if i == 0:
                        print("\nWarning: Display not available. Processing without visual output.")
        
        # Calculate processing time
        total_time = time.time() - start_time
        fps_processing = len(frames) / total_time
        
        # Generate statistics
        stats = {
            'total_frames_processed': len(frames),
            'total_processing_time': total_time,
            'fps': fps_processing,
            'player_count': len(tracks['players'][0]) if tracks['players'] else 0,
            'advanced_tracking': True
        }
        
        # Generate detailed player statistics
        player_stats = speed_estimator.get_player_statistics(tracks)
        stats['player_stats'] = player_stats
        
        # Store processing times
        self.processing_times = [total_time / len(frames)] * len(frames)
        
        # Generate summary
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Frames processed: {stats['total_frames_processed']}")
        print(f"Total time: {stats['total_processing_time']:.2f}s")
        print(f"Processing FPS: {stats['fps']:.2f}")
        print(f"Players detected: {stats['player_count']}")
        print(f"{'='*60}\n")
        
        return stats
    
    def _process_video_legacy(self, video_path: str,
                             skip_frames: int = 1,
                             max_frames: Optional[int] = None,
                             display_results: bool = True) -> Dict:
        """Legacy video processing (frame-by-frame)"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return {"error": "Could not open video"}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing video: {video_path}")
        print(f"FPS: {fps:.2f} | Total frames: {total_frames}")
        print(f"Processing every {skip_frames} frame(s)\n")
        
        frame_idx = 0
        processed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % skip_frames == 0:
                # Process frame
                results = self.process_frame(
                    frame, 
                    frame_idx,
                    save_visualizations=True
                )
                
                processed_count += 1
                self.frame_count += 1
                
                # Progress update
                if processed_count % 10 == 0:
                    avg_time = np.mean(self.processing_times[-10:])
                    print(f"Processed {processed_count} frames | "
                          f"Avg time: {avg_time:.3f}s | "
                          f"Players detected: {len(results['detections'])}")
                
                # Display
                if display_results and self.annotated_frames:
                    try:
                        # Try to display if GUI support is available
                        cv2.imshow('Football Analysis', self.annotated_frames[-1])
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nStopping (user interrupt)...")
                            break
                    except cv2.error as e:
                        # Silently continue if display is not supported
                        # Only show the warning once
                        if processed_count == 1:
                            print("\nWarning: Display not available. Processing without visual output.")
                            print("Results will still be saved to output directory.")
                
                # Check max frames
                if max_frames and processed_count >= max_frames:
                    print(f"\nReached max frames limit ({max_frames})")
                    break
            
            frame_idx += 1
        
        cap.release()
        if display_results:
            cv2.destroyAllWindows()
        
        # Generate statistics
        stats = self._generate_statistics(processed_count)
        
        return stats
            
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return {"error": "Could not open video"}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing video: {video_path}")
        print(f"FPS: {fps:.2f} | Total frames: {total_frames}")
        print(f"Processing every {skip_frames} frame(s)\n")
        
        frame_idx = 0
        processed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % skip_frames == 0:
                # Process frame
                results = self.process_frame(
                    frame, 
                    frame_idx,
                    save_visualizations=True
                )
                
                processed_count += 1
                self.frame_count += 1
                
                # Progress update
                if processed_count % 10 == 0:
                    avg_time = np.mean(self.processing_times[-10:])
                    print(f"Processed {processed_count} frames | "
                          f"Avg time: {avg_time:.3f}s | "
                          f"Players detected: {len(results['detections'])}")
                
                # Display
                if display_results and self.annotated_frames:
                    try:
                        # Try to display if GUI support is available
                        cv2.imshow('Football Analysis', self.annotated_frames[-1])
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nStopping (user interrupt)...")
                            break
                    except cv2.error as e:
                        # Silently continue if display is not supported
                        # Only show the warning once
                        if processed_count == 1:
                            print("\nWarning: Display not available. Processing without visual output.")
                            print("Results will still be saved to output directory.")
                
                # Check max frames
                if max_frames and processed_count >= max_frames:
                    print(f"\nReached max frames limit ({max_frames})")
                    break
            
            frame_idx += 1
        
        cap.release()
        if display_results:
            cv2.destroyAllWindows()
        
        # Generate statistics
        stats = self._generate_statistics(processed_count)
        
        return stats
    
    def _generate_statistics(self, processed_frames: int) -> Dict:
        """Generate processing statistics"""
        stats = {
            'total_frames_processed': processed_frames,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'total_processing_time': sum(self.processing_times),
            'fps': 1 / np.mean(self.processing_times) if self.processing_times else 0,
            'advanced_tracking': self.use_advanced_tracking
        }
        
        # Save statistics
        stats_path = self.output_dir / 'processing_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Frames processed: {stats['total_frames_processed']}")
        print(f"Average time per frame: {stats['avg_processing_time']:.3f}s")
        print(f"Processing FPS: {stats['fps']:.2f}")
        print(f"Total time: {stats['total_processing_time']:.2f}s")
        print(f"{'='*60}\n")
        
        return stats
    
    def save_output_video(self, output_path: str = 'output_video.mp4',
                         fps: int = 30, save_frames: bool = True):
        """Save annotated frames as video"""
        if not self.annotated_frames:
            print("No frames to save")
            return
            
        try:
            # Make sure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            output_file = self.output_dir / output_path
            h, w = self.annotated_frames[0].shape[:2]
            
            # Use the improved video_utils.save_video function for better quality
            result = save_video(
                frames=self.annotated_frames,
                output_path=str(output_file),
                fps=fps,
                codec='XVID'  # Use XVID codec for better quality
            )
            
            # Save key frames as images
            if save_frames:
                visualizations_dir = self.output_dir / 'visualizations'
                os.makedirs(visualizations_dir, exist_ok=True)
                
                # Save every 10th frame or at least 10 frames total
                step = max(1, len(self.annotated_frames) // 10)
                for i, frame in enumerate(self.annotated_frames):
                    if i % step == 0:
                        frame_path = visualizations_dir / f"frame_{i:04d}.jpg"
                        cv2.imwrite(str(frame_path), frame)
                print(f"Saved {len(self.annotated_frames) // step} key frames to {visualizations_dir}")
            
            if result:
                print(f"Output video saved successfully to: {output_file}")
            else:
                print(f"Error: Failed to save output video to: {output_file}")
                
        except Exception as e:
            print(f"Error saving output video: {str(e)}")
    
    def generate_final_report(self):
        """Generate comprehensive match analysis report"""
        # report = self.recorder.generate_report()
        
        report = {
            'match_id': self.match_id,
            'frames_processed': self.frame_count,
            'processing_stats': self._generate_statistics(self.frame_count),
            'advanced_tracking': self.use_advanced_tracking
        }
        
        # Add player statistics if available
        if self.use_advanced_tracking and hasattr(self, 'adv_tracker'):
            try:
                # Create detailed player reports
                reports_dir = self.output_dir / 'reports'
                os.makedirs(reports_dir, exist_ok=True)
                
                # Generate player statistics
                speed_estimator = SpeedDistanceEstimator(frame_rate=30)
                player_stats = {}
                
                # If we have tracks available
                if hasattr(self.adv_tracker, 'tracks') and self.adv_tracker.tracks:
                    tracks = self.adv_tracker.tracks
                    player_stats = speed_estimator.get_player_statistics(tracks)
                    
                    # Save player statistics
                    players_report_path = reports_dir / 'player_statistics.json'
                    with open(players_report_path, 'w') as f:
                        json.dump(player_stats, f, indent=2)
                    print(f"Player statistics saved to {players_report_path}")
                    
                    # Add to main report
                    report['player_statistics'] = player_stats
            except Exception as e:
                print(f"Warning: Could not generate player statistics: {str(e)}")
        
        report_path = self.output_dir / 'final_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Final report saved to {report_path}")
        return report
    
    def cleanup(self):
        """Clean up resources"""
        # self.recorder.close()
        print("Pipeline cleanup complete")


# Main execution script
if __name__ == "__main__":
    # Configuration
    # Load Roboflow API key from .env file
    def load_roboflow_api_key(env_path='roboflow_api_key.env'):
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('ROBOTOFLOW_API_KEY='):
                        return line.strip().split('=', 1)[1]
        except Exception:
            return None

    roboflow_api_key = load_roboflow_api_key()

    CONFIG = {
        'video_path': 'input-videos/08fd33_4.mp4',
        'yolo_model': 'yolov8x.pt',  # Using the largest YOLOv8 model for better detection
        'output_dir': 'match_analysis_output',
        'match_id': 1,
        'skip_frames': 2,         # Process every 2nd frame for speed
        'max_frames': 200,        # Limit for testing
        'display': False,         # Set to False to avoid display issues
        'use_advanced': True,     # Use the new advanced tracking system
        'use_roboflow': True,    # Use Roboflow API for better detection
        'use_cache': False,       # Force recomputation of tracks
        'roboflow_api_key': roboflow_api_key, # Loaded from env file
        'show_speed': False,      # Hide speed information (prevent overlapping player IDs)
        'show_distance': False    # Hide distance information (prevent overlapping player IDs)
    }
    
    # Initialize pipeline
    pipeline = FootballAnalysisPipeline(
        yolo_model=CONFIG['yolo_model'],
        output_dir=CONFIG['output_dir'],
        match_id=CONFIG['match_id'],
        use_advanced_tracking=CONFIG['use_advanced'],
        use_roboflow=CONFIG['use_roboflow'],
        api_key=CONFIG['roboflow_api_key'],  # Use the API key from CONFIG
        use_cache=CONFIG['use_cache'],
        show_speed=CONFIG['show_speed'],     # Control speed display
        show_distance=CONFIG['show_distance'] # Control distance display
    )
    
    # Process video
    stats = pipeline.process_video(
        video_path=CONFIG['video_path'],
        skip_frames=CONFIG['skip_frames'],
        max_frames=CONFIG['max_frames'],
        display_results=CONFIG['display']
    )
    
    # Save output video
    pipeline.save_output_video('tracked_output.mp4', fps=15)
    
    # Generate final report
    report = pipeline.generate_final_report()
    
    # Cleanup
    pipeline.cleanup()
    
    print("\n✓ Pipeline execution complete!")
    print(f"✓ Output saved to: {CONFIG['output_dir']}")