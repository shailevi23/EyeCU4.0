"""
Roboflow API Integration for Football Player Detection
Dependencies: pip install roboflow ultralytics

IMPORTANT: Setting up Roboflow API access
-----------------------------------------
1. Obtain your API key from: https://app.roboflow.com/settings/api
2. Set it as an environment variable:
   - PowerShell: $env:ROBOFLOW_API_KEY = "RF-your-api-key"
   - Bash: export ROBOFLOW_API_KEY="RF-your-api-key"
3. Or pass it directly to the RoboflowDetector constructor

NOTE: API keys must start with "RF-" (new format) or "rf_" (older format)
"""

import cv2
import numpy as np
from roboflow import Roboflow
from typing import List, Dict, Optional, Union
import os
from pathlib import Path
import time

# Quick way to test if API key works
def test_roboflow_key(api_key=None):
    """Test if Roboflow API key is valid"""
    key = api_key or os.environ.get('ROBOFLOW_API_KEY')
    if not key:
        print("No API key found. Set ROBOFLOW_API_KEY environment variable or pass key as argument.")
        return False
    
    try:
        import requests
        resp = requests.get("https://api.roboflow.com/user/verify", 
                           headers={"Authorization": f"Bearer {key}"})
        if resp.status_code == 200:
            print("✅ API key validated successfully!")
            return True
        else:
            print(f"❌ API key validation failed: {resp.status_code} - {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ API key test failed with error: {e}")
        return False

# Uncomment to test your API key directly
# if __name__ == "__main__":
#     test_roboflow_key()

class RoboflowDetector:
    """
    Integration with Roboflow API for football player detection
    """
    def _format_api_key(self, key):
        """Attempt to format the API key correctly or provide guidance"""
        if not key:
            return None
            
        # For our specific API key, we know it works without the prefix
        # The test showed that direct API calls fail with RF- prefix, but the Roboflow client works
        # So we'll keep the original format and let the client handle it
        print(f"DEBUG: Using API key: '{key[:5]}...' (length: {len(key)})")
        
        return key
            
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_id: str = "football-players-detection-3zvbc-3u4yx/1",  # Update this to your actual project ID
                 confidence: float = 0.5,
                 use_local: bool = True,
                 local_model: str = 'yolov8s.pt'):
        """
        Initialize Roboflow API client
        
        Args:
            api_key: Roboflow API key (if None, will look for ROBOFLOW_API_KEY env var)
            model_id: Roboflow model ID (format: "project/version")
            confidence: Confidence threshold for detections
            use_local: Whether to use a local YOLOv8 model as fallback
            local_model: Path to local YOLO model weights
        """
        self.use_local = use_local
        self.confidence = confidence
        self.local_model = local_model
        
        # Get API key from parameter or environment
        raw_api_key = api_key or os.environ.get('ROBOFLOW_API_KEY')
        self.api_key = self._format_api_key(raw_api_key)
        
        # Debug API key information
        if self.api_key:
            print(f"DEBUG: API key provided: {self.api_key[:5]}... (length: {len(self.api_key)})")
            
            # Validate API key format (typically RF-... or rf_...)
            if self.api_key.startswith(('RF-', 'rf_')):
                print("DEBUG: API key format appears valid")
            else:
                print("DEBUG: WARNING - API key format may be invalid (should start with 'RF-' or 'rf_')")
                print("DEBUG: Your key starts with '" + self.api_key[:5] + "...' - This does not match the expected format")
                print("DEBUG: You may need to obtain a valid API key from https://roboflow.com/")
                print("DEBUG: API key format example: 'RF-abcdef123456' or 'rf_abcdef123456'...")
                
                # If we attempted to fix the key, note that
                if raw_api_key != self.api_key:
                    print(f"DEBUG: Attempted to fix API key format: '{raw_api_key[:5]}...' -> '{self.api_key[:5]}...'")
        else:
            print("DEBUG: No API key provided")
            
        self.model_id = model_id
        print(f"DEBUG: Model ID: {self.model_id}")
        self.detections_cache = {}
        
        # Initialize counters to track API usage
        self.api_request_count = 0
        self.api_success_count = 0
        self.local_model_count = 0
        self.last_stats_time = time.time()
        
        # Initialize both options
        if use_local:
            try:
                from ultralytics import YOLO
                self.local_detector = YOLO(local_model)
                print(f"Loaded local model from {local_model}")
            except Exception as e:
                print(f"Error loading local model: {str(e)}")
                self.local_detector = None
        else:
            self.local_detector = None
        
        # Initialize Roboflow API if key is available
        if self.api_key:
            try:
                print(f"Initializing Roboflow API client...")
                project_id, version_id = model_id.split('/')
                
                
                # Direct API key validation test
                try:
                    import requests
                    test_url = "https://api.roboflow.com/user/verify"
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    response = requests.get(test_url, headers=headers, timeout=5)
                    
                    if response.status_code == 200:
                        print("API key validated successfully!")
                    else:
                        print(f"API key validation failed")
                except Exception:
                    pass
                
                # Get Roboflow version
                try:
                    import roboflow
                    import pkg_resources
                    try:
                        roboflow_version = pkg_resources.get_distribution("roboflow").version
                        print(f"Using Roboflow version: {roboflow_version}")
                    except:
                        pass
                except Exception:
                    pass
                
                # Create Roboflow instance properly - api_key should be a string
                print("Creating Roboflow instance...")
                self.rf = Roboflow(api_key=str(self.api_key))
                
                # Get workspace (handle API changes in different versions)
                try:
                    # Newer Roboflow API versions
                    self.workspace = self.rf.workspace()
                except Exception:
                    try:
                        # Try getting default workspace
                        workspaces = self.rf.workspaces()
                        if workspaces and len(workspaces) > 0:
                            self.workspace = workspaces[0]
                        else:
                            raise Exception("No workspaces found")
                    except Exception as e:
                        print(f"Failed to access workspace: {e}")
                        raise
                
                # Access project and model
                self.project = self.rf.workspace("shai-1c9ud").project("football-players-detection-3zvbc-3u4yx")        
                self.version = self.project.version(str(version_id))
                self.model = self.version.model

                if self.model is None:
                    raise RuntimeError(f"❌ Failed to load model version {version_id} for project {project_id}")
                else:
                    print(f"✅ Successfully loaded Roboflow model: {project_id}/{version_id}")

                
                if self.model:
                    print(f"Successfully connected to Roboflow API for model {model_id}")
                else:
                    print(f"WARNING: Failed to load model {model_id} - will use local model")
                    
                # Check if predict method exists
                if not hasattr(self.model, 'predict'):
                    print("WARNING - model does not have predict method!")
                    
                # Test the model with a simple predict call to verify it's working
                try:
                    print("Testing Roboflow API connection...")
                    
                    # Create a small test image
                    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                    # Draw a rectangle to make it more meaningful
                    cv2.rectangle(test_img, (30, 30), (70, 70), (255, 255, 255), 2)
                    test_path = "test_roboflow.jpg"
                    cv2.imwrite(test_path, test_img)
                    
                    if os.path.exists(test_path):
                        # Try API call
                        test_result = self.model.predict(test_path, confidence=0.1)
                        
                        # Check if we got predictions
                        has_predictions = False
                        if hasattr(test_result, 'predictions') and len(test_result.predictions) > 0:
                            has_predictions = True
                        elif isinstance(test_result, list) and len(test_result) > 0:
                            has_predictions = True
                            
                        if has_predictions:
                            print("Roboflow API test successful")
                        else:
                            print("Roboflow API connected but returned no predictions")
                        
                        os.remove(test_path)
                    else:
                        print("Failed to create test image")
                except Exception as test_err:
                    print(f"DEBUG: Test predict failed: {test_err}")
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"Error initializing Roboflow API: {str(e)}")
                self.model = None
                
                # Fall back to local model if available
                if not self.local_detector and self.use_local:
                    try:
                        from ultralytics import YOLO
                        self.local_detector = YOLO(local_model)
                        print(f"Falling back to local model from {local_model}")
                    except Exception as local_err:
                        print(f"Could not initialize local model: {local_err}")
                
                print("\n" + "="*70)
                print("ROBOFLOW API CONNECTION FAILED")
                print("LIKELY CAUSE: Invalid API key format or authentication issue")
                print("TROUBLESHOOTING STEPS:")
                print("1. Verify your API key starts with 'RF-' or 'rf_'")
                print("2. Check that you're using a valid key from your Roboflow account")
                print("3. Get a valid API key from: https://app.roboflow.com/settings/api")
                print("4. Set environment variable: $env:ROBOFLOW_API_KEY='your-key-here'")
                print("="*70 + "\n")
                        
                print("NOTE: The system will continue using the local YOLO model only")
        else:
            print("No Roboflow API key provided, using local model only")
            self.model = None
    
    def detect(self, image: np.ndarray, 
              frame_id: Optional[int] = None) -> List[Dict]:
        """
        Detect players, referees, and ball in an image
        
        Args:
            image: Input image (BGR format)
            frame_id: Optional frame ID for caching
        
        Returns:
            List of detection dictionaries
        """
        # Check cache if frame_id is provided
        if frame_id is not None and frame_id in self.detections_cache:
            return self.detections_cache[frame_id]
            
        start_time = time.time()
        detections = []
        
        # First try Roboflow API if available
        if self.model:
            # Track API attempt
            self.api_request_count += 1
            
            try:
                # Convert image to format needed by Roboflow
                # Need to save image temporarily as Roboflow Python API
                # doesn't support direct numpy array input
                temp_path = f"temp_frame_{int(time.time())}.jpg"
                cv2.imwrite(temp_path, image)
                
                # Call Roboflow API
                try:
                    # Check what API endpoint is being used
                    api_endpoint = None
                    if hasattr(self.model, 'api_url'):
                        api_endpoint = self.model.api_url
                    elif hasattr(self.model, 'endpoint'):
                        api_endpoint = self.model.endpoint
                    elif hasattr(self.model, '_api_endpoint'):
                        api_endpoint = self.model._api_endpoint
                        
                    response = self.model.predict(temp_path, confidence=self.confidence)
                    # Track successful API call
                    self.api_success_count += 1
                except Exception as predict_error:
                    print(f"DEBUG: Predict call failed: {predict_error}")
                    print(f"DEBUG: Error type: {type(predict_error)}")
                    
                    # Check for common network errors
                    error_str = str(predict_error).lower()
                    if "timeout" in error_str or "connection" in error_str or "network" in error_str:
                        print("DEBUG: LIKELY NETWORK CONNECTIVITY ISSUE")
                        # Test basic internet connectivity
                        try:
                            import requests
                            test_response = requests.get("https://www.google.com", timeout=5)
                            print(f"DEBUG: Internet connectivity test: {test_response.status_code}")
                        except Exception as conn_err:
                            print(f"DEBUG: Internet connectivity test failed: {conn_err}")
                    
                    import traceback
                    traceback.print_exc()
                    raise
                    
                os.remove(temp_path)
                
                # Process predictions
                if hasattr(response, 'predictions'):
                    raw_preds = response.predictions
                elif isinstance(response, list):
                    raw_preds = response
                else:
                    raw_preds = []
                
                # Convert to standard format
                for pred in raw_preds:
                    # Handle both dictionary and object formats
                    if isinstance(pred, dict):
                        # Dictionary format (older API versions)
                        class_name = pred.get('class', 'unknown')
                        x = pred.get('x', 0)
                        y = pred.get('y', 0)
                        width = pred.get('width', 0)
                        height = pred.get('height', 0)
                        confidence = pred.get('confidence', 0.0)
                    else:
                        # Object format (newer API versions)
                        try:
                            # Check if json_prediction is available
                            if hasattr(pred, 'json_prediction'):
                                # Get prediction data from json_prediction
                                pred_data = pred.json_prediction
                                if isinstance(pred_data, dict):
                                    # Extract values from json_prediction dictionary
                                    class_name = pred_data.get('class', 'unknown')
                                    x = pred_data.get('x', 0)
                                    y = pred_data.get('y', 0)
                                    width = pred_data.get('width', 0)
                                    height = pred_data.get('height', 0)
                                    confidence = pred_data.get('confidence', 0.0)
                                else:
                                    raise ValueError("Invalid json_prediction format")
                            # If json_prediction not available, try json method
                            elif hasattr(pred, 'json') and callable(pred.json):
                                # Call json() method to get prediction data
                                pred_data = pred.json()
                                if isinstance(pred_data, dict):
                                    # Extract values from json dictionary
                                    class_name = pred_data.get('class', 'unknown')
                                    x = pred_data.get('x', 0)
                                    y = pred_data.get('y', 0)
                                    width = pred_data.get('width', 0)
                                    height = pred_data.get('height', 0)
                                    confidence = pred_data.get('confidence', 0.0)
                                else:
                                    raise ValueError("Invalid json result format")
                            else:
                                # Try direct attribute access as last resort
                                class_name = getattr(pred, 'class_name', 'unknown') if hasattr(pred, 'class_name') else getattr(pred, 'class', 'unknown')
                                x = getattr(pred, 'x', 0)
                                y = getattr(pred, 'y', 0)
                                width = getattr(pred, 'width', 0)
                                height = getattr(pred, 'height', 0)
                                confidence = getattr(pred, 'confidence', 0.0)
                        except Exception:
                            # Try to get any useful information
                            try:
                                # If the object can be converted to string, check for useful info
                                pred_str = str(pred)
                                if 'class' in pred_str:
                                    # Try to extract data from string representation
                                    import re
                                    class_match = re.search(r"class['\"]?\s*:\s*['\"]([^'\"]+)['\"]", pred_str)
                                    if class_match:
                                        class_name = class_match.group(1)
                                    else:
                                        class_name = 'unknown'
                                else:
                                    class_name = 'unknown'
                                # Set default values
                                x, y, width, height = 0, 0, 0, 0
                                confidence = 0.0
                            except:
                                # Fallback to minimal values
                                class_name = 'unknown'
                                x, y, width, height = 0, 0, 0, 0
                                confidence = 0.0
                    
                    if class_name.lower() in ['goalkeeper']:
                        class_name = 'player'  # Normalize goalkeeper as player
                    
                    # Convert from center format to x1y1x2y2
                    x1 = int(x - width/2)
                    y1 = int(y - height/2)
                    x2 = int(x + width/2)
                    y2 = int(y + height/2)
                    
                    # Ensure within image boundaries
                    h, w = image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Extract crop
                    crop = image[y1:y2, x1:x2].copy()
                    
                    # Add detection
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,  # Use the extracted confidence value
                        'class': class_name,
                        'crop': crop,
                        'center': [int((x1+x2)/2), int((y1+y2)/2)]
                    })
                
            except Exception as e:
                print(f"Error using Roboflow API: {str(e)}")
                print("Falling back to local model")
                
        # Use local model if Roboflow failed or wasn't available
        if not detections and self.local_detector:
            # Update counter for local model usage
            self.local_model_count += 1
            
            try:
                # Lower the confidence threshold for detection to capture more objects
                results = self.local_detector(image, conf=max(0.1, self.confidence - 0.15))
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Get class name
                        if hasattr(r, 'names'):
                            class_name = r.names[class_id]
                        else:
                            if class_id == 0:
                                class_name = 'player'
                            elif class_id == 1:
                                class_name = 'referee'
                            elif class_id == 2:
                                class_name = 'ball'
                            else:
                                class_name = f'class_{class_id}'
                        
                        # Map common YOLO classes to football classes
                        if class_name.lower() in ['goalkeeper', 'person', 'player', 'football player']:
                            class_name = 'player'
                        elif class_name.lower() in ['sports ball', 'ball', 'football']:
                            class_name = 'ball'
                        
                        # Crop player region
                        crop = image[y1:y2, x1:x2].copy()
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': class_name,
                            'crop': crop,
                            'center': [(x1+x2)//2, (y1+y2)//2]
                        })
            
            except Exception as e:
                print(f"Error using local model: {str(e)}")
        
        # Cache results
        if frame_id is not None:
            self.detections_cache[frame_id] = detections
            
        elapsed = time.time() - start_time
        
        # Print statistics periodically (every 10 seconds)
        self._print_usage_statistics()
        
        return detections
    
    def _print_usage_statistics(self):
        """Print API vs local model usage statistics"""
        current_time = time.time()
        # Print stats every 60 seconds instead of 10
        if current_time - self.last_stats_time >= 60:
            total_frames = self.api_success_count + self.local_model_count
            if total_frames > 0:
                api_percent = (self.api_success_count / total_frames) * 100
                local_percent = (self.local_model_count / total_frames) * 100
                
                print("\n" + "="*50)
                print(f"DETECTOR USAGE STATISTICS:")
                print(f"API requests: {self.api_request_count}")
                print(f"API successful: {self.api_success_count} ({api_percent:.1f}%)")
                print(f"Local model: {self.local_model_count} ({local_percent:.1f}%)")
                print(f"Total frames processed: {total_frames}")
                print("="*50 + "\n")
                
            self.last_stats_time = current_time
    
    def extract_jersey_number(self, player_crop: np.ndarray) -> tuple:
        """
        Extract jersey number from player crop using OCR
        
        Args:
            player_crop: Cropped player image
        
        Returns:
            (jersey_number, confidence) tuple
        """
        # This requires importing easyocr which is already in the player_detection.py
        try:
            import easyocr
            
            # Initialize OCR reader if needed
            if not hasattr(self, 'ocr_reader'):
                self.ocr_reader = easyocr.Reader(['en'], gpu=True)
            
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
            
        except Exception as e:
            print(f"Error extracting jersey number: {str(e)}")
            return "", 0.0
    
    def process_video_batch(self, video_path: str, 
                           start_frame: int = 0, 
                           max_frames: Optional[int] = None,
                           skip_frames: int = 1) -> List[List[Dict]]:
        """
        Process a batch of frames from a video
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            max_frames: Maximum number of frames to process
            skip_frames: Process every nth frame
            
        Returns:
            List of detection lists per frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Skip to start_frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
        all_detections = []
        processed_count = 0
        frame_idx = start_frame
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % skip_frames == 0:
                print(f"Processing frame {frame_idx}...")
                detections = self.detect(frame, frame_idx)
                
                # Add jersey numbers for players
                for det in detections:
                    if det['class'] == 'player':
                        jersey_num, jersey_conf = self.extract_jersey_number(det['crop'])
                        det['jersey_number'] = jersey_num
                        det['jersey_confidence'] = jersey_conf
                    
                all_detections.append(detections)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} frames, found {len(detections)} objects")
                    
                if max_frames and processed_count >= max_frames:
                    break
                    
            frame_idx += 1
            
        cap.release()
        print(f"Completed batch processing: {processed_count} frames")
        return all_detections

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = RoboflowDetector(
        api_key=None,  # Use environment variable or pass key directly
        model_id="football-players-detection/1",
        confidence=0.5,
        use_local=True
    )
    
    # Process a single image
    image = cv2.imread("test_image.jpg")
    detections = detector.detect(image)
    print(f"Found {len(detections)} objects")
    
    # Visualize results
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{det['class']} {det['confidence']:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite("output.jpg", image)