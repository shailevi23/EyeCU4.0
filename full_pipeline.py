"""
Complete Integrated Pipeline for Football Player Analysis
This MVP ties together all modules into a working system
Using advanced tracking, team assignment, and speed estimation
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, Optional
import json
import time


# Import new advanced tracking modules
from trackers.football_tracker import FootballTracker
from trackers.team_assigner import TeamAssigner
from trackers.camera_movement import CameraMovementEstimator
from trackers.speed_distance import SpeedDistanceEstimator
from trackers.player_ball_assigner import PlayerBallAssigner
from trackers.video_utils import read_video, save_video, get_video_info
from trackers.cache_utils import compute_cache_key, cache_path_for
from trackers.detector import HUMAN_ACCEPT_CONF, HUMAN_CANDIDATE_CONF


class FootballAnalysisPipeline:
    """
    Complete pipeline for football player analysis
    Integrates detection, tracking, re-identification, and database management
    """
    
    def __init__(self, 
                 yolo_model: str = 'yolov8x.pt',
                 output_dir: str = 'pipeline_output',
                 match_id: int = 1,
                 use_roboflow: bool = False,  # opt-in only; local YOLO is the default path
                 api_key: Optional[str] = None,
                 use_cache: bool = True,
                 show_speed: bool = False,
                 show_distance: bool = False,
                 imgsz: int = 960,
                 confidence: float = 0.25,
                 max_ball_gap: int = 15,
                 human_candidate_pool: bool = False,
                 ball_candidate_pool: bool = True,
                 tracker_backend: str = 'cbiou'):
        """
        Initialize complete pipeline
        Args:
            yolo_model: Path to YOLO weights
            output_dir: Output directory for all results
            match_id: Unique match identifier
            use_roboflow: Opt in to the Roboflow cloud detector (labelling/benchmark
                only). Defaults to False so the pipeline runs fully locally.
            api_key: Roboflow API key
            use_cache: Whether to use cached results
            show_speed: Whether to display player speed
            show_distance: Whether to display player distance covered
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.match_id = match_id
        self.use_roboflow = use_roboflow
        # Kept for the cache key: every one of these changes the result.
        self.yolo_model = yolo_model
        self.imgsz = imgsz
        self.confidence = confidence
        self.max_ball_gap = max_ball_gap
        # Changes what the tracker sees, so it must be part of the cache key.
        self.human_candidate_pool = human_candidate_pool
        self.ball_candidate_pool = ball_candidate_pool
        # Set per run in _process_video_advanced().
        self.source_fps = None
        self.effective_fps = None
        self.cache_key = None
        self.api_key = api_key
        self.use_cache = use_cache
        self.show_speed = show_speed
        self.show_distance = show_distance
        
        print("Initializing Football Analysis Pipeline...")

        # Create cache directory
        cache_dir = self.output_dir / 'cache'
        cache_dir.mkdir(exist_ok=True)

        # Initialize components
        self.adv_tracker = FootballTracker(
            model_path=yolo_model,
            use_roboflow=use_roboflow,
            api_key=api_key,
            persist_cache=use_cache,
            cache_dir=str(cache_dir),
            confidence=confidence,
            imgsz=imgsz,
            max_ball_gap=max_ball_gap,
            human_candidate_pool=human_candidate_pool,
            ball_candidate_pool=ball_candidate_pool,
            tracker_backend=tracker_backend,
        )

        self.team_assigner = TeamAssigner(num_teams=2)
        self.ball_assigner = PlayerBallAssigner(max_distance=70)

        # Pipeline state
        self.frame_count = 0
        self.processing_times = []
        self.annotated_frames = []
        
        print("Pipeline initialization complete!\n")
    
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
        
        return self._process_video_advanced(
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
        
        # Read the real source frame rate. Speed/distance are meaningless if
        # this is guessed: every value scales linearly with it.
        info = get_video_info(video_path)
        source_fps = float(info.get('fps') or 0)
        if source_fps <= 0:
            source_fps = 30.0
            print("Warning: could not read source FPS; assuming 30.0")

        # Only every skip_frames-th frame reaches the estimator, so the interval
        # between the frames it sees is skip_frames times longer.
        effective_fps = source_fps / max(1, skip_frames)
        self.source_fps = source_fps
        self.effective_fps = effective_fps
        print(f"Source FPS: {source_fps:.2f}  ->  effective FPS: {effective_fps:.2f} "
              f"(skip_frames={skip_frames})")

        # Cache keys cover everything that changes the result, so a cache from a
        # different video/model/setting cannot be reused.
        cache_key = compute_cache_key(
            video_path=video_path,
            model_path=self.yolo_model,
            detector_settings={'imgsz': self.imgsz, 'confidence': self.confidence,
                               'use_roboflow': self.use_roboflow,
                               # The pool changes the detections available to
                               # association, so a cache built with it on must
                               # never be reused by a run with it off.
                               'human_candidate_pool': self.human_candidate_pool,
                               'human_candidate_conf': (HUMAN_CANDIDATE_CONF
                                                        if self.human_candidate_pool
                                                        else None),
                               'human_accept_conf': max(self.confidence,
                                                        HUMAN_ACCEPT_CONF)},
            # Read from the tracker rather than hard-coded. This is a CACHE KEY:
            # it said 'bytetrack' while the constructor defaulted to CBIoU, so a
            # cache built by one association backend could be silently reused by
            # the other. The ball candidate pool changes what the selector has to
            # work with, so it belongs in the key too.
            tracker_settings={'max_ball_gap': self.max_ball_gap,
                              'tracker': self.adv_tracker.tracker_backend,
                              'ball_candidate_pool': self.ball_candidate_pool,
                              'ball_temporal_selector': 'v1'},
            skip_frames=skip_frames,
            max_frames=max_frames,
        )
        self.cache_key = cache_key

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
        cache_path = cache_path_for(cache_dir, 'tracks', cache_key)
        tracks = self.adv_tracker.get_object_tracks(
            frames,
            read_from_cache=self.use_cache,
            cache_path=cache_path
        )
        
        # 3. Add position information to tracks
        print("Adding position information...")
        self.adv_tracker.add_position_to_tracks(tracks)
        
        # 4. Resolve the ball track temporally.
        # BallTemporalSelector is the authoritative ball post-processing step:
        # it tags every reported ball with its provenance and leaves a frame
        # empty when the evidence does not support one. Frames with no ball are
        # normal output, not an error, and every consumer below handles them.
        print("Resolving ball positions (temporal selector)...")
        frame_width = frames[0].shape[1] if len(frames) else None
        tracks["ball"] = self.adv_tracker.apply_ball_temporal_selection(
            tracks["ball"],
            candidates=getattr(self.adv_tracker, 'ball_candidates', None),
            fps=effective_fps,
            frame_width=frame_width,
        )
        
        # 5. Estimate camera movement
        print("Estimating camera movement...")
        camera_estimator = CameraMovementEstimator(frames[0])
        camera_cache = cache_path_for(cache_dir, 'camera_movement', cache_key)
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
        speed_estimator = SpeedDistanceEstimator(frame_rate=effective_fps)
        speed_estimator.add_speed_and_distance_to_tracks(tracks)
        
        # 9. Determine ball possession and team control
        print("Analyzing ball possession...")
        team_ball_control = self.ball_assigner.compute_team_ball_control(tracks)
        
        # 10. Generate outputs
        print("Generating visualization...")
        
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
    
    def _generate_statistics(self, processed_frames: int) -> Dict:
        """Generate processing statistics"""
        stats = {
            'total_frames_processed': processed_frames,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'total_processing_time': sum(self.processing_times),
            'fps': 1 / np.mean(self.processing_times) if self.processing_times else 0,
            'advanced_tracking': True
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
            'source_fps': self.source_fps,
            'effective_fps': self.effective_fps,
            'cache_key': self.cache_key,
            'processing_stats': self._generate_statistics(self.frame_count),
            'advanced_tracking': True
        }

        # Add per-track statistics if available
        if hasattr(self, 'adv_tracker'):
            try:
                # Create detailed player reports
                reports_dir = self.output_dir / 'reports'
                os.makedirs(reports_dir, exist_ok=True)

                # Reuse the run's real effective frame rate, not a guess.
                speed_estimator = SpeedDistanceEstimator(
                    frame_rate=self.effective_fps or 30.0)
                player_stats = {}
                
                # If we have tracks available
                if hasattr(self.adv_tracker, 'tracks') and self.adv_tracker.tracks:
                    tracks = self.adv_tracker.tracks
                    player_stats = speed_estimator.get_player_statistics(tracks)
                    
                    # These are TRACK ids, not people. Tracking currently
                    # fragments identities, so the count of entries here is an
                    # upper bound on the number of physical players, not a
                    # measurement of it. Reporting it as a player count would be
                    # wrong; the real count stays unresolved until tracking/ReID
                    # is stable (docs/archive/TODO_legacy.md section 6).
                    report['unique_track_ids'] = len(player_stats)
                    # pixels_per_meter is an unvalidated constant, so every
                    # speed/distance figure below is relative, not real-world.
                    report['speed_distance_calibrated'] = False

                    # Save per-track statistics
                    players_report_path = reports_dir / 'player_statistics.json'
                    with open(players_report_path, 'w') as f:
                        json.dump(player_stats, f, indent=2)
                    print(f"Per-track statistics saved to {players_report_path} "
                          f"({len(player_stats)} unique track ids -- track count, "
                          f"not player count)")

                    # Add to main report
                    report['track_statistics'] = player_stats
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
    CONFIG = {
        'video_path': 'input-videos/08fd33_4.mp4',
        'yolo_model': 'yolov8x.pt',  # Using the largest YOLOv8 model for better detection
        'output_dir': 'match_analysis_output',
        'match_id': 1,
        'skip_frames': 2,         # Process every 2nd frame for speed
        'max_frames': 200,        # Limit for testing
        'display': False,         # Set to False to avoid display issues
        'use_roboflow': False,   # local YOLO is the production path
        'use_cache': False,       # Force recomputation of tracks
        'roboflow_api_key': os.environ.get('ROBOFLOW_API_KEY'),  # never hardcode the key
        'show_speed': False,      # Hide speed information (prevent overlapping player IDs)
        'show_distance': False    # Hide distance information (prevent overlapping player IDs)
    }
    
    # Initialize pipeline
    pipeline = FootballAnalysisPipeline(
        yolo_model=CONFIG['yolo_model'],
        output_dir=CONFIG['output_dir'],
        match_id=CONFIG['match_id'],
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