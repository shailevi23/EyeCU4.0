"""
Complete Integrated Pipeline for Football Player Analysis
This MVP ties together all modules into a working system
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import time

# Import all modules (assuming they're in the same directory)
# from module1_detection import PlayerDetector, VideoProcessor
# from module2_crops import FaceBodyExtractor
# from module3_mesh import PoseEstimator3D, MeshSimilarityCalculator
# from module4_tracking import PlayerTracker
# from module5_6_reid import ReIdentificationSystem, InterruptionDetector
# from module7_database import MatchRecorder
# from module8_evaluation import TrackingEvaluator, PerformanceAnalyzer


class FootballAnalysisPipeline:
    """
    Complete pipeline for football player analysis
    Integrates detection, tracking, re-identification, and database management
    """
    
    def __init__(self, 
                 yolo_model: str = 'yolov8x.pt',
                 output_dir: str = 'pipeline_output',
                 match_id: int = 1):
        """
        Initialize complete pipeline
        Args:
            yolo_model: Path to YOLO weights
            output_dir: Output directory for all results
            match_id: Unique match identifier
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.match_id = match_id
        
        print("Initializing Football Analysis Pipeline...")
        
        # Module 1: Detection
        print("  - Loading YOLO detector...")
        # self.detector = PlayerDetector(yolo_model)
        
        # Module 2: Face/Body extraction
        print("  - Initializing face/body extractor...")
        # self.face_body_extractor = FaceBodyExtractor(output_dir)
        
        # Module 3: 3D Pose/Mesh
        print("  - Initializing pose estimator...")
        # self.pose_estimator = PoseEstimator3D(output_dir)
        # self.mesh_calculator = MeshSimilarityCalculator()
        
        # Module 4: Tracking
        print("  - Setting up tracker...")
        # self.tracker = PlayerTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Module 5-6: Re-identification
        print("  - Initializing re-ID system...")
        # self.reid_system = ReIdentificationSystem(
        #     mesh_weight=0.4,
        #     jersey_weight=0.4,
        #     proportion_weight=0.2,
        #     reid_threshold=0.7
        # )
        # self.interruption_detector = InterruptionDetector()
        
        # Module 7: Database
        print("  - Setting up database...")
        # self.recorder = MatchRecorder(match_id, output_dir)
        
        # Module 8: Evaluation
        # self.evaluator = TrackingEvaluator()
        # self.analyzer = PerformanceAnalyzer(output_dir / 'evaluation')
        
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
        # Placeholder - replace with actual detector
        # detections = self.detector.process_frame(frame, frame_id)
        
        # Simulated detections for demo
        detections = [
            {
                'bbox': [100, 100, 180, 300],
                'confidence': 0.9,
                'crop': frame[100:300, 100:180],
                'center': [140, 200],
                'jersey_number': '10',
                'jersey_confidence': 0.85,
                'frame_id': frame_id
            }
        ]
        return detections
    
    def _track_players(self, detections: List[Dict]) -> List[Dict]:
        """Step 2: Track players across frames"""
        # tracked = self.tracker.update(detections)
        
        # Simulated tracking for demo
        tracked = [
            {
                'tracking_id': 1,
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'detection_data': det
            }
            for det in detections
        ]
        return tracked
    
    def _detect_interruptions(self, tracked_objects: List[Dict],
                             frame_id: int) -> List[tuple]:
        """Step 3: Detect tracking interruptions"""
        current_ids = [obj['tracking_id'] for obj in tracked_objects]
        # events = self.interruption_detector.detect_interruptions(current_ids, frame_id)
        
        # Handle lost tracks
        # for event_type, track_id in events:
        #     if event_type == 'lost':
        #         self.reid_system.handle_tracking_interruption(track_id)
        
        return []  # events
    
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
        vis = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            player_id = det['player_id']
            tracking_id = det['tracking_id']
            jersey = det.get('jersey_number', '')
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on player ID
            np.random.seed(player_id)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Draw bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw labels
            label = f"P{player_id}"
            if jersey:
                label += f" #{jersey}"
            if det.get('is_reassigned'):
                label += " (Re-ID)"
            
            cv2.putText(vis, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw tracking ID (smaller)
            cv2.putText(vis, f"T{tracking_id}", (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
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
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return {}
        
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
                    cv2.imshow('Football Analysis', self.annotated_frames[-1])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopping (user interrupt)...")
                        break
                
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
            'fps': 1 / np.mean(self.processing_times) if self.processing_times else 0
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
                         fps: int = 30):
        """Save annotated frames as video"""
        if not self.annotated_frames:
            print("No frames to save")
            return
        
        output_file = self.output_dir / output_path
        h, w = self.annotated_frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_file), fourcc, fps, (w, h))
        
        for frame in self.annotated_frames:
            out.write(frame)
        
        out.release()
        print(f"Saved output video to {output_file}")
    
    def generate_final_report(self):
        """Generate comprehensive match analysis report"""
        # report = self.recorder.generate_report()
        
        report = {
            'match_id': self.match_id,
            'frames_processed': self.frame_count,
            'processing_stats': self._generate_statistics(self.frame_count)
        }
        
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
    CONFIG = {
        'video_path': 'soccer_match.mp4',
        'yolo_model': 'yolov8x.pt',
        'output_dir': 'match_analysis_output',
        'match_id': 1,
        'skip_frames': 2,  # Process every 2nd frame for speed
        'max_frames': 200,  # Limit for testing
        'display': True
    }
    
    # Initialize pipeline
    pipeline = FootballAnalysisPipeline(
        yolo_model=CONFIG['yolo_model'],
        output_dir=CONFIG['output_dir'],
        match_id=CONFIG['match_id']
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