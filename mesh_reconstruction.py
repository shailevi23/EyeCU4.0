"""
Module 3: 3D Avatar/Mesh Reconstruction
Dependencies: pip install mediapipe torch numpy opencv-python
For advanced SMPL: pip install smplx chumpy
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, Optional, List
import json
import pickle

class PoseEstimator3D:
    """Extract 3D pose and mesh representations for players"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = Path(output_dir)
        self.mesh_dir = self.output_dir / 'meshes'
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe Pose (provides 3D landmarks)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 0, 1, or 2 (2 is most accurate)
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_pose_landmarks(self, player_crop: np.ndarray) -> Optional[Dict]:
        """
        Extract 3D pose landmarks from player crop
        Args:
            player_crop: Player image crop
        Returns:
            Dictionary with 2D and 3D landmarks, or None
        """
        rgb = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            return None
        
        h, w = player_crop.shape[:2]
        
        # Extract landmarks in multiple formats
        landmarks_2d = []
        landmarks_3d = []
        visibility = []
        
        for lm in results.pose_landmarks.landmark:
            # 2D normalized coordinates
            landmarks_2d.append([lm.x, lm.y])
            
            # 3D coordinates (z is depth relative to hips)
            landmarks_3d.append([lm.x, lm.y, lm.z])
            
            # Visibility score
            visibility.append(lm.visibility)
        
        pose_data = {
            'landmarks_2d': np.array(landmarks_2d),
            'landmarks_3d': np.array(landmarks_3d),
            'visibility': np.array(visibility),
            'world_landmarks': self._extract_world_landmarks(results)
        }
        
        # Extract body segmentation mask
        if results.segmentation_mask is not None:
            pose_data['segmentation'] = results.segmentation_mask
        
        return pose_data
    
    def _extract_world_landmarks(self, results) -> Optional[np.ndarray]:
        """Extract world landmarks (real-world 3D coordinates in meters)"""
        if not results.pose_world_landmarks:
            return None
        
        world_lm = []
        for lm in results.pose_world_landmarks.landmark:
            world_lm.append([lm.x, lm.y, lm.z])
        
        return np.array(world_lm)
    
    def compute_body_proportions(self, landmarks_3d: np.ndarray) -> Dict:
        """
        Compute body proportions from landmarks (useful for re-identification)
        Args:
            landmarks_3d: 3D landmark array
        Returns:
            Dictionary of body measurements
        """
        # MediaPipe Pose landmark indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        
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
        """
        Create compact feature vector for similarity comparison
        Args:
            pose_data: Pose data dictionary
        Returns:
            Feature vector (1D numpy array)
        """
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
    
    def save_pose_data(self, pose_data: Dict, tracking_id: int, frame_id: int) -> str:
        """
        Save pose data to disk
        Args:
            pose_data: Pose data dictionary
            tracking_id: Player tracking ID
            frame_id: Frame number
        Returns:
            Path to saved file
        """
        filename = f"player_{tracking_id}_frame_{frame_id}_pose.pkl"
        filepath = self.mesh_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(pose_data, f)
        
        return str(filepath)
    
    def load_pose_data(self, filepath: str) -> Dict:
        """Load pose data from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def visualize_pose(self, player_crop: np.ndarray, 
                      pose_data: Dict,
                      output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize pose landmarks on image
        Args:
            player_crop: Original image
            pose_data: Pose data with landmarks
            output_path: Optional path to save visualization
        Returns:
            Annotated image
        """
        annotated = player_crop.copy()
        h, w = annotated.shape[:2]
        
        landmarks_2d = pose_data['landmarks_2d']
        visibility = pose_data['visibility']
        
        # Draw landmarks
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        
        # Convert to MediaPipe format for drawing
        landmark_list = mp_pose.PoseLandmark
        connections = mp_pose.POSE_CONNECTIONS
        
        # Draw connections
        for connection in connections:
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
    
    def process_player_detection(self, player_crop: np.ndarray,
                                tracking_id: int,
                                frame_id: int,
                                save_data: bool = True,
                                save_visualization: bool = False) -> Optional[Dict]:
        """
        Complete processing pipeline for single player
        Args:
            player_crop: Player image crop
            tracking_id: Tracking ID
            frame_id: Frame number
            save_data: Whether to save pose data
            save_visualization: Whether to save visualization
        Returns:
            Pose data dictionary with paths
        """
        # Extract pose
        pose_data = self.extract_pose_landmarks(player_crop)
        
        if pose_data is None:
            return None
        
        # Add metadata
        pose_data['tracking_id'] = tracking_id
        pose_data['frame_id'] = frame_id
        
        # Compute features
        pose_data['feature_vector'] = self.create_feature_vector(pose_data)
        pose_data['body_proportions'] = self.compute_body_proportions(
            pose_data['landmarks_3d']
        )
        
        # Save
        if save_data:
            pose_path = self.save_pose_data(pose_data, tracking_id, frame_id)
            pose_data['pose_path'] = pose_path
        
        if save_visualization:
            vis_path = self.mesh_dir / f"player_{tracking_id}_frame_{frame_id}_viz.jpg"
            self.visualize_pose(player_crop, pose_data, str(vis_path))
            pose_data['visualization_path'] = str(vis_path)
        
        return pose_data


class MeshSimilarityCalculator:
    """Calculate similarity between 3D poses/meshes"""
    
    @staticmethod
    def feature_vector_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute cosine similarity between feature vectors
        Args:
            feat1, feat2: Feature vectors
        Returns:
            Similarity score [0, 1]
        """
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
        """
        Compare body proportions
        Args:
            props1, props2: Body proportion dictionaries
        Returns:
            Similarity score [0, 1]
        """
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


# Example usage
if __name__ == "__main__":
    estimator = PoseEstimator3D(output_dir='output')
    
    # Process player
    player_crop = cv2.imread('player_crop.jpg')
    pose_data = estimator.process_player_detection(
        player_crop,
        tracking_id=1,
        frame_id=100,
        save_visualization=True
    )
    
    if pose_data:
        print(f"Extracted pose with {len(pose_data['landmarks_3d'])} landmarks")
        print(f"Body proportions: {pose_data['body_proportions']}")
    
    # Compare two poses
    calculator = MeshSimilarityCalculator()
    
    # Example comparison
    pose_data2 = estimator.process_player_detection(
        player_crop,
        tracking_id=2,
        frame_id=101
    )
    
    if pose_data and pose_data2:
        feat_sim = calculator.feature_vector_similarity(
            pose_data['feature_vector'],
            pose_data2['feature_vector']
        )
        prop_sim = calculator.body_proportion_similarity(
            pose_data['body_proportions'],
            pose_data2['body_proportions']
        )
        print(f"Feature similarity: {feat_sim:.3f}")
        print(f"Proportion similarity: {prop_sim:.3f}")