"""
player_id/jersey_ocr.py
OCR for extracting jersey numbers from player crops
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import re
from collections import defaultdict, Counter

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

# Check for CUDA availability
CUDA_AVAILABLE = False
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass


class JerseyOCR:
    """Extract jersey numbers using OCR"""
    
    def __init__(self, engine='paddleocr', confidence_threshold=0.5, 
                 number_range=(1, 99)):
        self.engine = engine
        self.confidence_threshold = confidence_threshold
        self.number_range = number_range
        
        # Initialize OCR engine
        if engine == 'paddleocr' and PADDLEOCR_AVAILABLE:
            # Use GPU only if CUDA is available
            try:
                # Try with new API (without show_log parameter)
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en', 
                                  use_gpu=CUDA_AVAILABLE)
            except ValueError:
                # Try with older API
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
                
            print(f"PaddleOCR initialized with GPU support: {CUDA_AVAILABLE}")
        elif engine == 'tesseract' and TESSERACT_AVAILABLE:
            self.tesseract_config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
        else:
            raise RuntimeError(f"OCR engine '{engine}' not available")
    
    def preprocess_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Focus on chest/back area where numbers are
        h, w = crop.shape[:2]
        
        # Extract upper torso region
        y1, y2 = int(h * 0.2), int(h * 0.7)
        x1, x2 = int(w * 0.25), int(w * 0.75)
        roi = crop[y1:y2, x1:x2]
        
        if roi.size == 0:
            return crop
        
        # Resize for better OCR
        scale = 3
        roi = cv2.resize(roi, (roi.shape[1] * scale, roi.shape[0] * scale))
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques
        processed_images = []
        
        # 1. Simple threshold
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(thresh1)
        
        # 2. Adaptive threshold
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        processed_images.append(thresh2)
        
        # 3. Bilateral filter + threshold
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(thresh3)
        
        # 4. Edge enhancement
        edges = cv2.Canny(gray, 50, 150)
        processed_images.append(edges)
        
        return processed_images
    
    def extract_number_tesseract(self, image: np.ndarray) -> Tuple[Optional[int], float]:
        """Extract number using Tesseract"""
        if not TESSERACT_AVAILABLE:
            return None, 0.0
        
        try:
            # Try OCR on preprocessed images
            results = []
            
            preprocessed = self.preprocess_for_ocr(image)
            for img in preprocessed:
                text = pytesseract.image_to_string(img, config=self.tesseract_config)
                data = pytesseract.image_to_data(img, config=self.tesseract_config, 
                                                output_type=pytesseract.Output.DICT)
                
                # Extract numbers with confidence
                for i, txt in enumerate(data['text']):
                    if txt.strip().isdigit():
                        conf = float(data['conf'][i]) / 100.0
                        num = int(txt.strip())
                        if self.number_range[0] <= num <= self.number_range[1]:
                            results.append((num, conf))
            
            if results:
                # Return most confident result
                results.sort(key=lambda x: x[1], reverse=True)
                return results[0]
            
            return None, 0.0
            
        except Exception as e:
            return None, 0.0
    
    def extract_number_paddle(self, image: np.ndarray) -> Tuple[Optional[int], float]:
        """Extract number using PaddleOCR"""
        if not PADDLEOCR_AVAILABLE:
            return None, 0.0
        
        try:
            results = []
            preprocessed = self.preprocess_for_ocr(image)
            
            for img in preprocessed:
                result = self.ocr.ocr(img, cls=True)
                
                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0]
                        conf = line[1][1]
                        
                        # Extract digits
                        digits = re.findall(r'\d+', text)
                        for digit_str in digits:
                            num = int(digit_str)
                            if self.number_range[0] <= num <= self.number_range[1]:
                                results.append((num, conf))
            
            if results:
                # Return most confident result
                results.sort(key=lambda x: x[1], reverse=True)
                return results[0]
            
            return None, 0.0
            
        except Exception as e:
            return None, 0.0
    
    def extract_number(self, crop: np.ndarray) -> Tuple[Optional[int], float]:
        """Extract jersey number from player crop"""
        if self.engine == 'paddleocr':
            return self.extract_number_paddle(crop)
        elif self.engine == 'tesseract':
            return self.extract_number_tesseract(crop)
        return None, 0.0


class PlayerIDManager:
    """Manage persistent player IDs using OCR and tracking"""
    
    def __init__(self, ocr_engine='paddleocr', ocr_confidence=0.5):
        self.ocr = JerseyOCR(engine=ocr_engine, confidence_threshold=ocr_confidence)
        
        # Track ID -> Player ID mapping
        self.track_to_player = {}
        
        # Player ID info: {player_id: {track_ids, jersey_number, team, confidence}}
        self.player_info = {}
        
        # History of OCR detections per track
        self.ocr_history = defaultdict(list)
        
        # Interpolation data
        self.last_seen = {}
    
    def process_detection(self, track_id: int, crop: np.ndarray, 
                         team_id: int, frame_num: int) -> Optional[int]:
        """
        Process a detection and assign/update player ID
        
        Args:
            track_id: Tracking ID from tracker
            crop: Player bounding box crop
            team_id: Assigned team
            frame_num: Current frame number
            
        Returns:
            player_id if successfully assigned, None otherwise
        """
        # Check if track already has player ID
        if track_id in self.track_to_player:
            player_id = self.track_to_player[track_id]
            self.last_seen[player_id] = frame_num
            return player_id
        
        # Try OCR every N frames (expensive operation)
        if frame_num % 15 == 0:  # Every 15 frames (~0.5 seconds at 30fps)
            number, confidence = self.ocr.extract_number(crop)
            
            if number and confidence >= self.ocr.confidence_threshold:
                self.ocr_history[track_id].append((number, confidence, frame_num))
                
                # Create player ID from team and number
                player_id = f"{team_id}_{number:02d}"
                
                # Check if player ID already exists
                if player_id in self.player_info:
                    # Merge track IDs for same player
                    self.player_info[player_id]['track_ids'].add(track_id)
                    self.track_to_player[track_id] = player_id
                else:
                    # Create new player
                    self.player_info[player_id] = {
                        'track_ids': {track_id},
                        'jersey_number': number,
                        'team': team_id,
                        'confidence': confidence,
                        'first_seen': frame_num
                    }
                    self.track_to_player[track_id] = player_id
                
                self.last_seen[player_id] = frame_num
                return player_id
        
        # Use historical OCR if available
        if track_id in self.ocr_history and self.ocr_history[track_id]:
            # Get most common number from history
            numbers = [n for n, c, f in self.ocr_history[track_id]]
            if numbers:
                most_common = Counter(numbers).most_common(1)[0][0]
                player_id = f"{team_id}_{most_common:02d}"
                
                if player_id not in self.player_info:
                    self.player_info[player_id] = {
                        'track_ids': {track_id},
                        'jersey_number': most_common,
                        'team': team_id,
                        'confidence': 0.7,
                        'first_seen': frame_num
                    }
                
                self.track_to_player[track_id] = player_id
                self.last_seen[player_id] = frame_num
                return player_id
        
        return None
    
    def interpolate_missing_ids(self, max_gap_frames: int = 30):
        """Interpolate player IDs for gaps in tracking"""
        # For tracks that lost and regained the same player
        # Based on spatial proximity and team
        pass
    
    def get_player_id(self, track_id: int) -> Optional[str]:
        """Get player ID for a track"""
        return self.track_to_player.get(track_id)
    
    def get_player_info(self, player_id: str) -> Optional[dict]:
        """Get information about a player"""
        return self.player_info.get(player_id)
    
    def get_all_players(self, team_id: Optional[int] = None) -> List[str]:
        """Get all player IDs, optionally filtered by team"""
        if team_id is None:
            return list(self.player_info.keys())
        return [pid for pid, info in self.player_info.items() 
                if info['team'] == team_id]
    
    def get_statistics(self) -> dict:
        """Get OCR and player ID statistics"""
        total_tracks = len(self.track_to_player)
        total_players = len(self.player_info)
        
        # Count successful OCR detections
        ocr_detections = sum(len(hist) for hist in self.ocr_history.values())
        
        # Average confidence
        avg_conf = np.mean([info['confidence'] 
                           for info in self.player_info.values()]) if self.player_info else 0
        
        return {
            'total_tracks': total_tracks,
            'total_players': total_players,
            'ocr_detections': ocr_detections,
            'avg_confidence': avg_conf,
            'assignment_rate': total_tracks / max(total_players, 1)
        }