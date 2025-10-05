"""
Module 1: Player Detection & Jersey Number Recognition
Dependencies: pip install ultralytics opencv-python easyocr torch
"""

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from pathlib import Path
from typing import List, Dict, Tuple
import json

class PlayerDetector:
    """Handles player detection and jersey number recognition"""
    
    def __init__(self, yolo_model='yolov8x.pt', conf_threshold=0.5):
        """
        Initialize detector
        Args:
            yolo_model: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(yolo_model)
        self.conf_threshold = conf_threshold
        self.ocr_reader = easyocr.Reader(['en'], gpu=True)
        
    def detect_players(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect players in frame using YOLO
        Args:
            frame: Input video frame (BGR)
        Returns:
            List of detection dictionaries with bbox, confidence, crop
        """
        results = self.model(frame, classes=[0], conf=self.conf_threshold)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Crop player region
                player_crop = frame[y1:y2, x1:x2]
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'crop': player_crop,
                    'center': [(x1+x2)//2, (y1+y2)//2]
                })
        
        return detections
    
    def extract_jersey_number(self, player_crop: np.ndarray) -> Tuple[str, float]:
        """
        Extract jersey number from player crop using OCR
        Args:
            player_crop: Cropped player image
        Returns:
            (jersey_number, confidence) tuple
        """
        h, w = player_crop.shape[:2]
        
        # Focus on torso region (middle 40-70% height, center 60% width)
        torso = player_crop[int(h*0.4):int(h*0.7), int(w*0.2):int(w*0.8)]
        
        # Preprocessing for better OCR
        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        
        # Try multiple preprocessing strategies
        preprocessed = [
            gray,
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
        ]
        
        best_number = ""
        best_conf = 0.0
        
        for img in preprocessed:
            results = self.ocr_reader.readtext(img, allowlist='0123456789')
            
            for (bbox, text, conf) in results:
                # Filter for jersey number characteristics (1-2 digits)
                if text.isdigit() and 1 <= len(text) <= 2 and conf > best_conf:
                    best_number = text
                    best_conf = conf
        
        return best_number, best_conf
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> List[Dict]:
        """
        Complete processing pipeline for a single frame
        Args:
            frame: Input frame
            frame_id: Frame number/timestamp
        Returns:
            List of processed player detections
        """
        detections = self.detect_players(frame)
        
        for det in detections:
            jersey_num, jersey_conf = self.extract_jersey_number(det['crop'])
            det['jersey_number'] = jersey_num
            det['jersey_confidence'] = jersey_conf
            det['frame_id'] = frame_id
        
        return detections


class VideoProcessor:
    """Handles video ingestion and frame extraction"""
    
    def __init__(self, video_path: str, output_dir: str = 'output'):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def process_video(self, detector: PlayerDetector, 
                     skip_frames: int = 1,
                     max_frames: int = None) -> List[List[Dict]]:
        """
        Process video and detect players
        Args:
            detector: PlayerDetector instance
            skip_frames: Process every nth frame
            max_frames: Maximum frames to process (None = all)
        Returns:
            List of detections per frame
        """
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        all_detections = []
        frame_id = 0
        processed = 0
        
        print(f"Processing video: {self.video_path}")
        print(f"FPS: {fps}, Skip: {skip_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_id % skip_frames == 0:
                detections = detector.process_frame(frame, frame_id)
                all_detections.append(detections)
                
                processed += 1
                if processed % 10 == 0:
                    print(f"Processed {processed} frames, found {len(detections)} players")
                
                if max_frames and processed >= max_frames:
                    break
            
            frame_id += 1
        
        cap.release()
        print(f"Complete! Processed {processed} frames")
        return all_detections
    
    def save_detections(self, all_detections: List[List[Dict]], 
                       save_crops: bool = True):
        """Save detection results to disk"""
        # Create directories
        crops_dir = self.output_dir / 'crops'
        crops_dir.mkdir(exist_ok=True)
        
        # Prepare serializable data
        results = []
        for frame_dets in all_detections:
            frame_data = []
            for i, det in enumerate(frame_dets):
                det_copy = {
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'jersey_number': det['jersey_number'],
                    'jersey_confidence': det['jersey_confidence'],
                    'frame_id': det['frame_id'],
                    'center': det['center']
                }
                
                # Save crop
                if save_crops and det['crop'] is not None:
                    crop_path = crops_dir / f"frame_{det['frame_id']}_det_{i}.jpg"
                    cv2.imwrite(str(crop_path), det['crop'])
                    det_copy['crop_path'] = str(crop_path)
                
                frame_data.append(det_copy)
            results.append(frame_data)
        
        # Save JSON
        json_path = self.output_dir / 'detections.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved detections to {json_path}")


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = PlayerDetector(yolo_model='yolov8x.pt', conf_threshold=0.5)
    
    # Process video
    processor = VideoProcessor('your_video.mp4', output_dir='output')
    detections = processor.process_video(detector, skip_frames=5, max_frames=100)
    
    # Save results
    processor.save_detections(detections, save_crops=True)