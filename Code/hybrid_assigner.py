"""
team_assignment/hybrid_assigner.py
Hybrid team assignment combining Hamza's color clustering and EyeCU4.0 approach
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from collections import defaultdict

class ColorClusteringAssigner:
    """Hamza's color clustering-based team assignment"""
    
    def __init__(self, n_clusters=3, color_space='HSV', roi_ratio=0.6):
        self.n_clusters = n_clusters
        self.color_space = color_space
        self.roi_ratio = roi_ratio
        self.team_colors = None
        
    def extract_jersey_roi(self, bbox_crop: np.ndarray) -> np.ndarray:
        """Extract ROI focusing on jersey area (upper torso)"""
        h, w = bbox_crop.shape[:2]
        
        # Focus on upper-middle section (jersey area)
        start_y = int(h * 0.15)
        end_y = int(h * self.roi_ratio)
        start_x = int(w * 0.2)
        end_x = int(w * 0.8)
        
        roi = bbox_crop[start_y:end_y, start_x:end_x]
        return roi
    
    def convert_color_space(self, image: np.ndarray) -> np.ndarray:
        """Convert image to specified color space"""
        if self.color_space == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == 'LAB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return image
    
    def get_dominant_color(self, roi: np.ndarray) -> np.ndarray:
        """Extract dominant color using K-means"""
        converted = self.convert_color_space(roi)
        pixels = converted.reshape(-1, 3)
        
        # Filter out very dark/bright pixels (likely shadows/highlights)
        mask = (pixels[:, -1] > 20) & (pixels[:, -1] < 240)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) < 10:
            filtered_pixels = pixels
        
        kmeans = KMeans(n_clusters=min(3, len(filtered_pixels)), 
                       random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)
        
        # Return the most common cluster center
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_idx = labels[np.argmax(counts)]
        return kmeans.cluster_centers_[dominant_idx]
    
    def initialize_team_colors(self, player_crops: List[np.ndarray]):
        """Initialize team colors from first frame detections"""
        dominant_colors = []
        
        for crop in player_crops:
            roi = self.extract_jersey_roi(crop)
            if roi.size > 0:
                color = self.get_dominant_color(roi)
                dominant_colors.append(color)
        
        if len(dominant_colors) < 2:
            return None
        
        # Cluster all dominant colors to find team colors
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(dominant_colors)
        self.team_colors = kmeans.cluster_centers_
        
        return self.team_colors
    
    def assign_team(self, bbox_crop: np.ndarray) -> Tuple[int, float]:
        """Assign team based on color similarity"""
        if self.team_colors is None:
            return -1, 0.0
        
        roi = self.extract_jersey_roi(bbox_crop)
        if roi.size == 0:
            return -1, 0.0
        
        player_color = self.get_dominant_color(roi)
        
        # Find closest team color
        distances = [np.linalg.norm(player_color - tc) for tc in self.team_colors]
        team_id = np.argmin(distances)
        
        # Confidence based on distance (normalized)
        max_dist = 255 * np.sqrt(3)  # Max possible distance
        confidence = 1.0 - (distances[team_id] / max_dist)
        
        return team_id, confidence


class EyeCUTeamAssigner:
    """EyeCU4.0 team assignment approach"""
    
    def __init__(self):
        self.team_templates = {}
        self.assignments = defaultdict(lambda: {'team': -1, 'confidence': 0.0})
    
    def extract_features(self, bbox_crop: np.ndarray) -> np.ndarray:
        """Extract color histogram features"""
        # Convert to HSV
        hsv = cv2.cvtColor(bbox_crop, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram
        h_hist = cv2.calcHist([hsv], [0], None, [30], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        
        # Normalize
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        
        return np.concatenate([h_hist, s_hist])
    
    def initialize_templates(self, crops_with_positions: List[Tuple[np.ndarray, Tuple[int, int]]]):
        """Initialize team templates using field position"""
        left_crops = []
        right_crops = []
        
        # Split by field position
        for crop, (x, y) in crops_with_positions:
            if x < 0:  # Left side
                left_crops.append(crop)
            else:
                right_crops.append(crop)
        
        if left_crops:
            features = [self.extract_features(c) for c in left_crops]
            self.team_templates[0] = np.mean(features, axis=0)
        
        if right_crops:
            features = [self.extract_features(c) for c in right_crops]
            self.team_templates[1] = np.mean(features, axis=0)
    
    def assign_team(self, bbox_crop: np.ndarray, track_id: int) -> Tuple[int, float]:
        """Assign team using template matching"""
        if not self.team_templates:
            return -1, 0.0
        
        features = self.extract_features(bbox_crop)
        
        # Compare with templates
        similarities = {}
        for team_id, template in self.team_templates.items():
            similarity = cv2.compareHist(
                features.reshape(-1, 1), 
                template.reshape(-1, 1),
                cv2.HISTCMP_CORREL
            )
            similarities[team_id] = max(0, similarity)
        
        if not similarities:
            return -1, 0.0
        
        team_id = max(similarities, key=similarities.get)
        confidence = similarities[team_id]
        
        # Update persistent assignment
        self.assignments[track_id] = {'team': team_id, 'confidence': confidence}
        
        return team_id, confidence


class HybridTeamAssigner:
    """Hybrid team assignment with arbitration"""
    
    def __init__(self, hamza_weight=0.5, eyecu_weight=0.5, confidence_threshold=0.6):
        self.color_assigner = ColorClusteringAssigner()
        self.eyecu_assigner = EyeCUTeamAssigner()
        self.hamza_weight = hamza_weight
        self.eyecu_weight = eyecu_weight
        self.confidence_threshold = confidence_threshold
        
        self.track_history = defaultdict(lambda: {
            'team': -1,
            'confidence': 0.0,
            'last_updated': 0,
            'ocr_number': None
        })
    
    def initialize(self, player_crops: List[np.ndarray], 
                   crops_with_positions: List[Tuple[np.ndarray, Tuple[int, int]]]):
        """Initialize both assigners"""
        self.color_assigner.initialize_team_colors(player_crops)
        self.eyecu_assigner.initialize_templates(crops_with_positions)
    
    def assign_team(self, bbox_crop: np.ndarray, track_id: int, 
                    frame_num: int, ocr_number: Optional[int] = None) -> Tuple[int, float]:
        """
        Assign team using hybrid approach with arbitration
        
        Returns:
            (team_id, confidence) tuple
        """
        # Get assignments from both methods
        hamza_team, hamza_conf = self.color_assigner.assign_team(bbox_crop)
        eyecu_team, eyecu_conf = self.eyecu_assigner.assign_team(bbox_crop, track_id)
        
        # Check if both agree
        if hamza_team == eyecu_team and hamza_team != -1:
            # Both agree - high confidence
            combined_conf = (hamza_conf * self.hamza_weight + 
                           eyecu_conf * self.eyecu_weight)
            final_team = hamza_team
        
        elif hamza_team == -1 and eyecu_team != -1:
            # Only EyeCU has result
            final_team = eyecu_team
            combined_conf = eyecu_conf * 0.8
        
        elif hamza_team != -1 and eyecu_team == -1:
            # Only Hamza has result
            final_team = hamza_team
            combined_conf = hamza_conf * 0.8
        
        else:
            # Disagreement - use weighted voting
            hamza_score = hamza_conf * self.hamza_weight
            eyecu_score = eyecu_conf * self.eyecu_weight
            
            if hamza_score > eyecu_score:
                final_team = hamza_team
                combined_conf = hamza_score
            else:
                final_team = eyecu_team
                combined_conf = eyecu_score
        
        # Check against track history
        if track_id in self.track_history:
            hist = self.track_history[track_id]
            
            # If low confidence, use historical assignment
            if combined_conf < self.confidence_threshold and hist['confidence'] > 0.5:
                final_team = hist['team']
                combined_conf = hist['confidence'] * 0.9
            
            # If OCR number matches historical assignment, boost confidence
            if ocr_number and ocr_number == hist['ocr_number']:
                combined_conf = min(1.0, combined_conf * 1.2)
        
        # Update track history
        if combined_conf >= self.confidence_threshold:
            self.track_history[track_id] = {
                'team': final_team,
                'confidence': combined_conf,
                'last_updated': frame_num,
                'ocr_number': ocr_number
            }
        
        return final_team, combined_conf
    
    def get_team_stats(self) -> Dict:
        """Get statistics about team assignments"""
        teams = [t['team'] for t in self.track_history.values() if t['team'] != -1]
        if not teams:
            return {}
        
        from collections import Counter
        team_counts = Counter(teams)
        
        return {
            'total_players': len(self.track_history),
            'assigned_players': len(teams),
            'team_distribution': dict(team_counts),
            'avg_confidence': np.mean([t['confidence'] for t in self.track_history.values()])
        }