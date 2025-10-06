"""
Test Script for Face/Body Crop and Mesh Reconstruction Pipeline
This script tests the integration between face_body_crop.py and mesh_reconstruction.py
"""

import cv2
import numpy as np
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import random

# Import the modules we want to test
from face_body_crop import FaceBodyExtractor
from mesh_reconstruction import PoseEstimator3D, MeshSimilarityCalculator

# Test configuration
TEST_CONFIG = {
    'input_video': 'input-videos/08fd33_4.mp4',  # Use the available video file
    'output_dir': 'test_mesh_output',
    'max_frames': 20,  # Process 20 frames for testing
    'skip_frames': 5,  # Process every 5th frame
    'threshold': 0.5,  # Detection confidence threshold
}

class MeshPipelineTester:
    """Test the mesh pipeline integration between face/body crop and mesh reconstruction"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize modules
        print(f"Initializing test modules...")
        self.face_body_extractor = FaceBodyExtractor(output_dir=str(self.output_dir))
        self.pose_estimator = PoseEstimator3D(output_dir=str(self.output_dir))
        self.similarity_calculator = MeshSimilarityCalculator()
        
        # Results storage
        self.detections = {}  # {frame_id: [{detection_data}]}
        self.crops = {}  # {tracking_id: {frame_id: crop}}
        self.pose_data = {}  # {tracking_id: {frame_id: pose_data}}
        self.crop_paths = {}  # {tracking_id: {frame_id: paths}}
        
    def extract_player_crops(self, frame: np.ndarray) -> List[Dict]:
        """Extract player crops using a simple detection approach (for testing)"""
        # For testing, we'll use a simplified approach to get player crops
        # In a real scenario, this would come from a proper player detector
        
        h, w = frame.shape[:2]
        crops = []
        
        # Create artificial player crops from random regions of the frame
        # This is just for testing - in real use, you'd have actual detections
        num_players = random.randint(2, 5)  # Random number of players
        
        for i in range(num_players):
            # Create random but realistic player bounding box
            player_h = random.randint(int(h*0.2), int(h*0.5))  # Player height ~20-50% of frame
            player_w = int(player_h * 0.4)  # Width about 40% of height
            
            # Random position
            x = random.randint(0, w - player_w)
            y = random.randint(int(h*0.3), h - player_h)  # Bottom half of frame more likely
            
            # Extract crop
            crop = frame[y:y+player_h, x:x+player_w].copy()
            
            # Create detection dict
            detection = {
                'tracking_id': i+1,  # Simple sequential IDs
                'bbox': [x, y, x+player_w, y+player_h],
                'confidence': random.uniform(0.6, 0.95),
                'crop': crop,
            }
            
            crops.append(detection)
            
        return crops
    
    def process_video(self):
        """Process video and run complete pipeline test"""
        # Open video file
        video_path = self.config['input_video']
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return False
        
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video")
            return False
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video stats: {total_frames} frames @ {fps:.1f} FPS")
        print(f"Processing every {self.config['skip_frames']} frame(s)")
        
        frame_idx = 0
        processed_count = 0
        
        # Prepare results visualization
        visualizations = []
        
        start_time = time.time()
        
        # Process video frames
        while cap.isOpened() and processed_count < self.config['max_frames']:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % self.config['skip_frames'] == 0:
                print(f"Processing frame {frame_idx}")
                
                # Step 1: Extract player crops
                detections = self.extract_player_crops(frame)
                self.detections[frame_idx] = detections
                
                # Step 2: Process each player
                for det in detections:
                    tracking_id = det['tracking_id']
                    player_crop = det['crop']
                    
                    # Save crop for visualization
                    if tracking_id not in self.crops:
                        self.crops[tracking_id] = {}
                    self.crops[tracking_id][frame_idx] = player_crop
                    
                    # Step 3: Save face/body crops
                    crop_paths = self.face_body_extractor.save_crops(
                        player_crop,
                        tracking_id,
                        frame_idx
                    )
                    
                    if tracking_id not in self.crop_paths:
                        self.crop_paths[tracking_id] = {}
                    self.crop_paths[tracking_id][frame_idx] = crop_paths
                    
                    # Step 4: Extract 3D pose/mesh data
                    pose_data = self.pose_estimator.process_player_detection(
                        player_crop,
                        tracking_id,
                        frame_idx,
                        save_data=True,
                        save_visualization=True
                    )
                    
                    if pose_data:
                        if tracking_id not in self.pose_data:
                            self.pose_data[tracking_id] = {}
                        self.pose_data[tracking_id][frame_idx] = pose_data
                
                processed_count += 1
                
                # Create visualization
                vis_frame = self.create_visualization(frame, frame_idx)
                visualizations.append(vis_frame)
                
                # Save visualization
                vis_path = self.output_dir / f"frame_{frame_idx:04d}_viz.jpg"
                cv2.imwrite(str(vis_path), vis_frame)
                
            frame_idx += 1
        
        cap.release()
        
        # Calculate processing time
        total_time = time.time() - start_time
        print(f"Processing complete: {processed_count} frames in {total_time:.2f}s")
        print(f"Average time per frame: {total_time/processed_count:.3f}s")
        
        # Run similarity analysis
        self.analyze_pose_similarity()
        
        return True
    
    def create_visualization(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        """Create visualization frame with bbox and pose overlay"""
        vis_frame = frame.copy()
        
        # Draw player detections
        if frame_id in self.detections:
            for det in self.detections[frame_id]:
                tracking_id = det['tracking_id']
                bbox = det['bbox']
                
                # Draw bounding box
                cv2.rectangle(vis_frame, 
                             (bbox[0], bbox[1]), 
                             (bbox[2], bbox[3]), 
                             (0, 255, 0), 2)
                
                # Draw ID
                cv2.putText(vis_frame, 
                           f"Player {tracking_id}", 
                           (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                
                # Check if pose was detected
                has_pose = (tracking_id in self.pose_data and 
                           frame_id in self.pose_data[tracking_id])
                
                # Check if face was detected
                has_face = (tracking_id in self.crop_paths and 
                           frame_id in self.crop_paths[tracking_id] and
                           self.crop_paths[tracking_id][frame_id].get('face') is not None)
                
                status = f"Pose: {'✓' if has_pose else '✗'}, Face: {'✓' if has_face else '✗'}"
                cv2.putText(vis_frame, 
                           status, 
                           (bbox[0], bbox[3] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, (255, 255, 255), 1)
        
        # Add frame info
        cv2.putText(vis_frame, 
                   f"Frame: {frame_id}", 
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 255), 2)
        
        return vis_frame
    
    def analyze_pose_similarity(self):
        """Analyze similarity between poses from same and different players"""
        print("\nAnalyzing pose similarity...")
        
        # Collect all poses
        all_poses = []
        for tracking_id, frames_data in self.pose_data.items():
            for frame_id, pose_data in frames_data.items():
                all_poses.append({
                    'tracking_id': tracking_id,
                    'frame_id': frame_id,
                    'pose_data': pose_data
                })
        
        if len(all_poses) < 2:
            print("Not enough pose data for similarity analysis")
            return
        
        # Compare poses of same player across frames
        print("\nSame player similarity across frames:")
        for tracking_id in self.pose_data.keys():
            frame_ids = sorted(self.pose_data[tracking_id].keys())
            if len(frame_ids) < 2:
                continue
                
            # Compare adjacent frames
            similarities = []
            for i in range(len(frame_ids) - 1):
                frame1, frame2 = frame_ids[i], frame_ids[i+1]
                pose1 = self.pose_data[tracking_id][frame1]
                pose2 = self.pose_data[tracking_id][frame2]
                
                # Calculate similarities
                feat_sim = self.similarity_calculator.feature_vector_similarity(
                    pose1['feature_vector'],
                    pose2['feature_vector']
                )
                
                prop_sim = self.similarity_calculator.body_proportion_similarity(
                    pose1['body_proportions'],
                    pose2['body_proportions']
                )
                
                avg_sim = (feat_sim + prop_sim) / 2
                similarities.append(avg_sim)
                
                print(f"  Player {tracking_id}: Frames {frame1}-{frame2}: " +
                     f"Feature: {feat_sim:.3f}, Proportion: {prop_sim:.3f}, " +
                     f"Average: {avg_sim:.3f}")
        
        # Compare poses between different players
        if len(self.pose_data.keys()) >= 2:
            print("\nDifferent players similarity:")
            player_ids = sorted(self.pose_data.keys())
            
            for i in range(len(player_ids)):
                for j in range(i+1, len(player_ids)):
                    id1, id2 = player_ids[i], player_ids[j]
                    
                    # Get a frame where both players have pose data
                    frames1 = set(self.pose_data[id1].keys())
                    frames2 = set(self.pose_data[id2].keys())
                    common_frames = frames1.intersection(frames2)
                    
                    if not common_frames:
                        # Try different frames if no common frame
                        frame1 = next(iter(frames1)) if frames1 else None
                        frame2 = next(iter(frames2)) if frames2 else None
                        
                        if frame1 is None or frame2 is None:
                            continue
                    else:
                        frame1 = frame2 = next(iter(common_frames))
                    
                    pose1 = self.pose_data[id1][frame1]
                    pose2 = self.pose_data[id2][frame2]
                    
                    # Calculate similarities
                    feat_sim = self.similarity_calculator.feature_vector_similarity(
                        pose1['feature_vector'],
                        pose2['feature_vector']
                    )
                    
                    prop_sim = self.similarity_calculator.body_proportion_similarity(
                        pose1['body_proportions'],
                        pose2['body_proportions']
                    )
                    
                    avg_sim = (feat_sim + prop_sim) / 2
                    
                    print(f"  Players {id1}-{id2}: " +
                         f"Feature: {feat_sim:.3f}, Proportion: {prop_sim:.3f}, " +
                         f"Average: {avg_sim:.3f}")
        
        # Generate summary
        self.generate_summary_report()
        
    def generate_summary_report(self):
        """Generate summary report of the test"""
        print("\nTest Summary:")
        print(f"Processed {len(self.detections)} frames")
        print(f"Detected {len(self.crops)} players")
        
        face_count = 0
        pose_count = 0
        for tracking_id in self.crop_paths:
            for frame_id, paths in self.crop_paths[tracking_id].items():
                if paths.get('face') is not None:
                    face_count += 1
                    
        for tracking_id in self.pose_data:
            pose_count += len(self.pose_data[tracking_id])
        
        print(f"Detected {face_count} faces")
        print(f"Generated {pose_count} pose reconstructions")
        
        # Save stats
        stats = {
            'frames_processed': len(self.detections),
            'players_detected': len(self.crops),
            'faces_detected': face_count,
            'poses_detected': pose_count
        }
        
        print("\nOutput saved to:", self.output_dir)
        
        # Return success
        return True


if __name__ == "__main__":
    # Run test
    print("\n" + "="*50)
    print("Mesh Pipeline Integration Test")
    print("="*50 + "\n")
    
    tester = MeshPipelineTester(TEST_CONFIG)
    tester.process_video()
    
    print("\nTest complete!")