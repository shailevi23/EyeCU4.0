"""
Module 2: Face and Body Crop Saving
Dependencies: pip install mediapipe opencv-python
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

class FaceBodyExtractor:
    """Extract and save face and body crops from player detections"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = Path(output_dir)
        self.faces_dir = self.output_dir / 'faces'
        self.bodies_dir = self.output_dir / 'bodies'
        
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.bodies_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe Face Detection
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=1,  # 1 for full range (better for sports)
            min_detection_confidence=0.5
        )
        
    def detect_face_in_crop(self, player_crop: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face within player crop
        Args:
            player_crop: Cropped player image
        Returns:
            Face bbox as (x1, y1, x2, y2) or None
        """
        rgb = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb)
        
        if not results.detections:
            return None
        
        # Get first (most confident) detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        h, w = player_crop.shape[:2]
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        return (x1, y1, x2, y2)
    
    def expand_face_crop(self, face_bbox: Tuple[int, int, int, int], 
                        player_crop: np.ndarray,
                        expansion_factor: float = 1.5) -> Tuple[int, int, int, int]:
        """
        Expand face bounding box to include head/hair
        Args:
            face_bbox: Original face bbox
            player_crop: Player crop image
            expansion_factor: How much to expand (1.5 = 150% of original)
        Returns:
            Expanded bbox
        """
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
        """
        Save body and face crops with tracking information
        Args:
            player_crop: Full player crop
            tracking_id: Unique tracking ID for player
            frame_id: Frame number
            save_body: Whether to save body crop
            save_face: Whether to save face crop
        Returns:
            Dictionary with saved file paths
        """
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
    
    def process_detections_batch(self, detections: Dict[int, Dict],
                                 frame_id: int) -> Dict[int, Dict]:
        """
        Process batch of detections for a frame
        Args:
            detections: Dict mapping tracking_id -> detection data
            frame_id: Current frame number
        Returns:
            Updated detections with crop paths
        """
        for track_id, det in detections.items():
            if 'crop' in det and det['crop'] is not None:
                paths = self.save_crops(
                    det['crop'],
                    track_id,
                    frame_id,
                    save_body=True,
                    save_face=True
                )
                det.update(paths)
        
        return detections
    
    def create_face_gallery(self, tracking_ids: list = None, 
                           max_per_player: int = 5) -> None:
        """
        Create visual gallery of faces for each player
        Args:
            tracking_ids: List of IDs to include (None = all)
            max_per_player: Maximum faces per player
        """
        from collections import defaultdict
        
        # Organize faces by tracking ID
        faces_by_id = defaultdict(list)
        
        for face_path in sorted(self.faces_dir.glob("player_*_frame_*.jpg")):
            parts = face_path.stem.split('_')
            track_id = int(parts[1])
            
            if tracking_ids is None or track_id in tracking_ids:
                faces_by_id[track_id].append(str(face_path))
        
        # Create gallery for each player
        gallery_dir = self.output_dir / 'galleries'
        gallery_dir.mkdir(exist_ok=True)
        
        for track_id, face_paths in faces_by_id.items():
            # Sample faces
            sampled = face_paths[:max_per_player]
            
            # Load images
            images = [cv2.imread(p) for p in sampled]
            
            if not images:
                continue
            
            # Resize to same height
            h = 200
            resized = []
            for img in images:
                aspect = img.shape[1] / img.shape[0]
                w = int(h * aspect)
                resized.append(cv2.resize(img, (w, h)))
            
            # Concatenate horizontally
            gallery = np.hstack(resized)
            
            # Save
            gallery_path = gallery_dir / f"player_{track_id}_gallery.jpg"
            cv2.imwrite(str(gallery_path), gallery)
            print(f"Created gallery for player {track_id}: {gallery_path}")


# Example usage
if __name__ == "__main__":
    extractor = FaceBodyExtractor(output_dir='output')
    
    # Example: process single detection
    player_crop = cv2.imread('player_crop.jpg')
    paths = extractor.save_crops(
        player_crop,
        tracking_id=1,
        frame_id=100
    )
    print(f"Saved crops: {paths}")
    
    # Create face galleries
    extractor.create_face_gallery(max_per_player=5)