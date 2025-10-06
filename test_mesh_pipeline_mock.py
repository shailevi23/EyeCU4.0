"""
Test Script for Face/Body Crop and Mesh Reconstruction Pipeline
With Mock Implementation for MediaPipe
"""

import cv2
import numpy as np
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import random
import json

# Note: We'll use mock implementations instead of direct imports
# from face_body_crop import FaceBodyExtractor
# from mesh_reconstruction import PoseEstimator3D, MeshSimilarityCalculator

# Test configuration
TEST_CONFIG = {
    'input_video': 'input-videos/08fd33_4.mp4',  # Use the available video file
    'output_dir': 'test_mesh_output',
    'max_frames': 20,  # Process 20 frames for testing
    'skip_frames': 5,  # Process every 5th frame
    'threshold': 0.5,  # Detection confidence threshold
}

class MockFaceBodyExtractor:
    """Mock implementation of FaceBodyExtractor that doesn't rely on MediaPipe"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = Path(output_dir)
        self.faces_dir = self.output_dir / 'faces'
        self.bodies_dir = self.output_dir / 'bodies'
        
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.bodies_dir.mkdir(parents=True, exist_ok=True)
        
        print("  - Using MockFaceBodyExtractor (no MediaPipe dependency)")
        
    def detect_face_in_crop(self, player_crop: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Mock face detection that looks at the top 1/3 of the image"""
        h, w = player_crop.shape[:2]
        
        # We'll just assume the face is in the upper third of the crop
        # About 1/3 of the time, we'll simulate no face detection
        if random.random() < 0.7:  # 70% chance of face detection
            face_h = int(h * 0.3)  # Face height is about 30% of crop
            face_w = int(face_h * 0.8)  # Typical face aspect ratio
            
            # Center in upper third
            x1 = max(0, (w - face_w) // 2)
            y1 = int(h * 0.05)  # 5% from top
            x2 = min(w, x1 + face_w)
            y2 = min(h, y1 + face_h)
            
            return (x1, y1, x2, y2)
        else:
            return None
    
    def expand_face_crop(self, face_bbox: Tuple[int, int, int, int], 
                         player_crop: np.ndarray,
                         expansion_factor: float = 1.5) -> Tuple[int, int, int, int]:
        """Expand face bounding box to include head/hair"""
        x1, y1, x2, y2 = face_bbox
        h, w = player_crop.shape[:2]
        
        # Calculate center and dimensions
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        fw, fh = x2 - x1, y2 - y1
        
        # Expand
        new_w = int(fw * expansion_factor)
        new_h = int(fh * expansion_factor)
        
        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(w, cx + new_w // 2)
        y2 = min(h, cy + new_h // 2)
        
        return (x1, y1, x2, y2)
    
    def save_crops(self, player_crop: np.ndarray, 
                   tracking_id: int,
                   frame_id: int,
                   save_body: bool = True,
                   save_face: bool = True) -> Dict[str, str]:
        """Save body and face crops with tracking information"""
        paths = {}
        
        # Save body crop
        if save_body:
            body_path = self.bodies_dir / f"player_{tracking_id}_frame_{frame_id}.jpg"
            cv2.imwrite(str(body_path), player_crop)
            paths['body'] = str(body_path)
        
        # Detect and save face crop
        if save_face:
            face_bbox = self.detect_face_in_crop(player_crop)
            
            if face_bbox:
                # Expand to include head
                expanded_bbox = self.expand_face_crop(face_bbox, player_crop)
                x1, y1, x2, y2 = expanded_bbox
                face_crop = player_crop[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    face_path = self.faces_dir / f"player_{tracking_id}_frame_{frame_id}.jpg"
                    cv2.imwrite(str(face_path), face_crop)
                    paths['face'] = str(face_path)
                    paths['face_bbox'] = expanded_bbox
            else:
                paths['face'] = None
                paths['face_bbox'] = None
        
        return paths


class MockPoseEstimator3D:
    """Mock implementation of PoseEstimator3D that doesn't rely on MediaPipe"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = Path(output_dir)
        self.mesh_dir = self.output_dir / 'meshes'
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        
        print("  - Using MockPoseEstimator3D (no MediaPipe dependency)")
    
    def extract_pose_landmarks(self, player_crop: np.ndarray) -> Optional[Dict]:
        """Mock extraction of 3D pose landmarks"""
        h, w = player_crop.shape[:2]
        
        # We'll simulate successful pose detection 80% of the time
        if random.random() < 0.8:  # 80% chance of pose detection
            # Generate mock landmarks for a standing person
            # These are normalized coordinates (0-1 range)
            
            # Generate 33 landmarks (MediaPipe Pose has 33 landmarks)
            num_landmarks = 33
            
            # Create basic pose structure with randomized but realistic positions
            landmarks_2d = []
            landmarks_3d = []
            visibility = []
            
            # Head (top)
            landmarks_2d.append([0.5, 0.05])
            
            # Face landmarks
            for i in range(4):
                x = 0.5 + random.uniform(-0.05, 0.05)
                y = 0.1 + i * 0.02
                landmarks_2d.append([x, y])
            
            # Shoulders
            landmarks_2d.append([0.3, 0.2])  # Left shoulder
            landmarks_2d.append([0.7, 0.2])  # Right shoulder
            
            # Elbows
            landmarks_2d.append([0.2, 0.35])  # Left elbow
            landmarks_2d.append([0.8, 0.35])  # Right elbow
            
            # Wrists
            landmarks_2d.append([0.15, 0.5])  # Left wrist
            landmarks_2d.append([0.85, 0.5])  # Right wrist
            
            # Hands (fingers)
            for i in range(8):
                if i < 4:  # Left hand
                    x = 0.15 + random.uniform(-0.05, 0.05)
                    y = 0.55 + i * 0.02
                else:  # Right hand
                    x = 0.85 + random.uniform(-0.05, 0.05)
                    y = 0.55 + (i-4) * 0.02
                landmarks_2d.append([x, y])
            
            # Hip
            landmarks_2d.append([0.4, 0.5])  # Left hip
            landmarks_2d.append([0.6, 0.5])  # Right hip
            
            # Knees
            landmarks_2d.append([0.4, 0.7])  # Left knee
            landmarks_2d.append([0.6, 0.7])  # Right knee
            
            # Ankles
            landmarks_2d.append([0.4, 0.9])  # Left ankle
            landmarks_2d.append([0.6, 0.9])  # Right ankle
            
            # Feet
            landmarks_2d.append([0.35, 0.95])  # Left foot
            landmarks_2d.append([0.65, 0.95])  # Right foot
            
            # Fill remaining landmarks with reasonable positions
            while len(landmarks_2d) < num_landmarks:
                x = random.uniform(0.3, 0.7)
                y = random.uniform(0.2, 0.9)
                landmarks_2d.append([x, y])
            
            # Create 3D landmarks (2D + random depth)
            for lm in landmarks_2d:
                # Add random depth (z) value
                z = random.uniform(-0.2, 0.2)
                landmarks_3d.append([lm[0], lm[1], z])
                
                # Visibility score (higher for central landmarks)
                x_center_dist = abs(lm[0] - 0.5)
                vis_score = max(0.2, 1.0 - x_center_dist * 2)
                visibility.append(vis_score)
            
            # Create world landmarks (real-world 3D coordinates)
            world_landmarks = []
            for lm in landmarks_3d:
                # Scale to meters and center at origin
                wx = (lm[0] - 0.5) * 1.0  # ~1m width
                wy = (lm[1] - 0.5) * 1.8  # ~1.8m height
                wz = lm[2] * 0.5  # ~0.5m depth
                world_landmarks.append([wx, wy, wz])
            
            pose_data = {
                'landmarks_2d': np.array(landmarks_2d),
                'landmarks_3d': np.array(landmarks_3d),
                'visibility': np.array(visibility),
                'world_landmarks': np.array(world_landmarks)
            }
            
            # Add segmentation mask (mock)
            segmentation = np.zeros((h, w), dtype=np.uint8)
            # Create a simple silhouette in the middle
            x1, y1 = int(w * 0.3), 0
            x2, y2 = int(w * 0.7), h
            cv2.rectangle(segmentation, (x1, y1), (x2, y2), 255, -1)
            pose_data['segmentation'] = segmentation
            
            return pose_data
        
        return None
    
    def compute_body_proportions(self, landmarks_3d: np.ndarray) -> Dict:
        """Compute body proportions from landmarks"""
        # MediaPipe Pose landmark indices (approximate)
        LEFT_SHOULDER = 11 % len(landmarks_3d)
        RIGHT_SHOULDER = 12 % len(landmarks_3d)
        LEFT_HIP = 23 % len(landmarks_3d)
        RIGHT_HIP = 24 % len(landmarks_3d)
        LEFT_KNEE = 25 % len(landmarks_3d)
        RIGHT_KNEE = 26 % len(landmarks_3d)
        LEFT_ANKLE = 27 % len(landmarks_3d)
        RIGHT_ANKLE = 28 % len(landmarks_3d)
        
        def euclidean(p1, p2):
            return np.linalg.norm(landmarks_3d[p1] - landmarks_3d[p2])
        
        proportions = {
            'shoulder_width': euclidean(LEFT_SHOULDER, RIGHT_SHOULDER),
            'hip_width': euclidean(LEFT_HIP, RIGHT_HIP),
            'torso_height': euclidean(LEFT_SHOULDER, LEFT_HIP),
            'left_leg_length': euclidean(LEFT_HIP, LEFT_KNEE) + euclidean(LEFT_KNEE, LEFT_ANKLE),
            'right_leg_length': euclidean(RIGHT_HIP, RIGHT_KNEE) + euclidean(RIGHT_KNEE, RIGHT_ANKLE),
        }
        
        # Compute ratios (more robust to scale changes)
        torso = proportions['torso_height']
        if torso > 0:
            proportions['shoulder_to_torso_ratio'] = proportions['shoulder_width'] / torso
            proportions['hip_to_torso_ratio'] = proportions['hip_width'] / torso
        
        return proportions
    
    def create_feature_vector(self, pose_data: Dict) -> np.ndarray:
        """Create compact feature vector for similarity comparison"""
        landmarks_3d = pose_data['landmarks_3d']
        visibility = pose_data['visibility']
        
        # Get body proportions
        proportions = self.compute_body_proportions(landmarks_3d)
        
        # Create feature vector
        features = []
        
        # Add visible landmark positions (normalized)
        for i, (lm, vis) in enumerate(zip(landmarks_3d, visibility)):
            if vis > 0.5:  # Only use visible landmarks
                features.extend(lm)
        
        # Add body proportions
        features.extend([
            proportions['shoulder_width'],
            proportions['hip_width'],
            proportions['torso_height'],
            proportions.get('shoulder_to_torso_ratio', 0),
            proportions.get('hip_to_torso_ratio', 0),
        ])
        
        return np.array(features)
    
    def visualize_pose(self, player_crop: np.ndarray, 
                      pose_data: Dict,
                      output_path: Optional[str] = None) -> np.ndarray:
        """Visualize pose landmarks on image"""
        annotated = player_crop.copy()
        h, w = annotated.shape[:2]
        
        landmarks_2d = pose_data['landmarks_2d']
        visibility = pose_data['visibility']
        
        # Draw connections (simulating a stick figure)
        # Define some key connections for a human pose
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6),
            (6, 8), (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 9), (0, 10)
        ]
        
        # Only use connections that are within the landmarks array bounds
        max_idx = len(landmarks_2d) - 1
        valid_connections = [(a, b) for a, b in connections 
                            if a <= max_idx and b <= max_idx]
        
        # Draw connections
        for connection in valid_connections:
            start_idx, end_idx = connection
            if visibility[start_idx] > 0.5 and visibility[end_idx] > 0.5:
                start_point = (int(landmarks_2d[start_idx][0] * w),
                             int(landmarks_2d[start_idx][1] * h))
                end_point = (int(landmarks_2d[end_idx][0] * w),
                           int(landmarks_2d[end_idx][1] * h))
                cv2.line(annotated, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for i, (lm, vis) in enumerate(zip(landmarks_2d, visibility)):
            if vis > 0.5:
                point = (int(lm[0] * w), int(lm[1] * h))
                cv2.circle(annotated, point, 4, (0, 0, 255), -1)
        
        if output_path:
            cv2.imwrite(output_path, annotated)
        
        return annotated
    
    def save_pose_data(self, pose_data: Dict, tracking_id: int, frame_id: int) -> str:
        """Save pose data to disk"""
        # Save visualization
        vis_path = self.mesh_dir / f"player_{tracking_id}_frame_{frame_id}_viz.jpg"
        
        if 'player_crop' in pose_data:
            player_crop = pose_data['player_crop']
            self.visualize_pose(player_crop, pose_data, str(vis_path))
        
        # Save simplified pose data as JSON (instead of pickle)
        data_to_save = {
            'tracking_id': tracking_id,
            'frame_id': frame_id,
            'body_proportions': pose_data['body_proportions'],
            'feature_vector': pose_data['feature_vector'].tolist() if isinstance(pose_data['feature_vector'], np.ndarray) else pose_data['feature_vector'],
        }
        
        json_path = self.mesh_dir / f"player_{tracking_id}_frame_{frame_id}_pose.json"
        with open(json_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        return str(vis_path)
    
    def process_player_detection(self, player_crop: np.ndarray,
                                tracking_id: int,
                                frame_id: int,
                                save_data: bool = True,
                                save_visualization: bool = False) -> Optional[Dict]:
        """Complete processing pipeline for single player"""
        # Extract pose
        pose_data = self.extract_pose_landmarks(player_crop)
        
        if pose_data is None:
            return None
        
        # Add metadata
        pose_data['tracking_id'] = tracking_id
        pose_data['frame_id'] = frame_id
        pose_data['player_crop'] = player_crop  # Store for visualization
        
        # Compute features
        pose_data['feature_vector'] = self.create_feature_vector(pose_data)
        pose_data['body_proportions'] = self.compute_body_proportions(
            pose_data['landmarks_3d']
        )
        
        # Save
        if save_data or save_visualization:
            pose_path = self.save_pose_data(pose_data, tracking_id, frame_id)
            pose_data['pose_path'] = pose_path
        
        return pose_data


class MeshSimilarityCalculator:
    """Calculate similarity between 3D poses/meshes"""
    
    @staticmethod
    def feature_vector_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between feature vectors"""
        # Handle different lengths by using minimum size
        min_len = min(len(feat1), len(feat2))
        f1 = feat1[:min_len]
        f2 = feat2[:min_len]
        
        # Cosine similarity
        dot_product = np.dot(f1, f2)
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, (similarity + 1) / 2))  # Normalize to [0, 1]
    
    @staticmethod
    def body_proportion_similarity(props1: Dict, props2: Dict) -> float:
        """Compare body proportions"""
        keys = ['shoulder_to_torso_ratio', 'hip_to_torso_ratio']
        
        differences = []
        for key in keys:
            if key in props1 and key in props2:
                diff = abs(props1[key] - props2[key])
                differences.append(diff)
        
        if not differences:
            return 0.5
        
        # Convert difference to similarity (smaller diff = higher similarity)
        avg_diff = np.mean(differences)
        similarity = np.exp(-avg_diff * 5)  # Exponential decay
        
        return similarity


class MeshPipelineTester:
    """Test the mesh pipeline integration between face/body crop and mesh reconstruction"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize modules with mock implementations
        print(f"Initializing test modules...")
        self.face_body_extractor = MockFaceBodyExtractor(output_dir=str(self.output_dir))
        self.pose_estimator = MockPoseEstimator3D(output_dir=str(self.output_dir))
        self.similarity_calculator = MeshSimilarityCalculator()
        
        # Results storage
        self.detections = {}  # {frame_id: [{detection_data}]}
        self.crops = {}  # {tracking_id: {frame_id: crop}}
        self.pose_data = {}  # {tracking_id: {frame_id: pose_data}}
        self.crop_paths = {}  # {tracking_id: {frame_id: paths}}
        
    def extract_player_crops(self, frame: np.ndarray) -> List[Dict]:
        """Extract player crops using a simple detection approach (for testing)"""
        h, w = frame.shape[:2]
        crops = []
        
        # Create artificial player crops from random regions of the frame
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
        
        # Save as JSON
        stats_path = self.output_dir / 'test_summary.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\nOutput saved to:", self.output_dir)
        print(f"Summary statistics saved to: {stats_path}")
        
        # Return success
        return True


if __name__ == "__main__":
    # Run test
    print("\n" + "="*50)
    print("Mesh Pipeline Integration Test (Mock Version)")
    print("="*50 + "\n")
    
    tester = MeshPipelineTester(TEST_CONFIG)
    tester.process_video()
    
    print("\nTest complete!")