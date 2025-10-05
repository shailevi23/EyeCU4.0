# Football Player Analysis Pipeline - Complete Setup Guide

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv football_env
source football_env/bin/activate  # On Windows: football_env\Scripts\activate

# Install dependencies
pip install ultralytics opencv-python easyocr torch torchvision
pip install mediapipe numpy scipy filterpy lap
pip install sqlite3 pickle5 matplotlib
```

### Basic Usage

```python
from integrated_pipeline import FootballAnalysisPipeline

# Initialize pipeline
pipeline = FootballAnalysisPipeline(
    yolo_model='yolov8x.pt',
    output_dir='match_output',
    match_id=1
)

# Process video
stats = pipeline.process_video(
    video_path='your_match.mp4',
    skip_frames=2,
    max_frames=500,
    display_results=True
)

# Save results
pipeline.save_output_video('output.mp4')
pipeline.generate_final_report()
pipeline.cleanup()
```

## 📋 Module Overview

### **Module 1: Player Detection & Jersey Number Recognition**
**File:** `module1_detection.py`

**Key Classes:**
- `PlayerDetector`: YOLO-based detection + OCR
- `VideoProcessor`: Video ingestion and frame processing

**Key Features:**
- Multi-strategy OCR preprocessing for jersey numbers
- Confidence scoring for detections
- Batch processing with frame skipping

**Usage:**
```python
detector = PlayerDetector('yolov8x.pt', conf_threshold=0.5)
detections = detector.process_frame(frame, frame_id)
# Returns: List of dicts with bbox, confidence, jersey_number
```

---

### **Module 2: Face and Body Crop Extraction**
**File:** `module2_crops.py`

**Key Classes:**
- `FaceBodyExtractor`: MediaPipe face detection + crop management

**Key Features:**
- Automatic face detection within player crops
- Expandable face bounding boxes (includes hair/head)
- Gallery generation for visual inspection
- Organized file system storage

**Usage:**
```python
extractor = FaceBodyExtractor(output_dir='output')
paths = extractor.save_crops(player_crop, tracking_id=1, frame_id=100)
extractor.create_face_gallery(max_per_player=5)
```

---

### **Module 3: 3D Mesh Reconstruction**
**File:** `module3_mesh.py`

**Key Classes:**
- `PoseEstimator3D`: MediaPipe Pose for 3D landmarks
- `MeshSimilarityCalculator`: Feature comparison

**Key Features:**
- 33 3D landmarks per player
- Body proportion calculations (ratios for re-ID)
- Feature vector generation for similarity matching
- Pose visualization

**Usage:**
```python
estimator = PoseEstimator3D(output_dir='output')
pose_data = estimator.process_player_detection(
    player_crop, tracking_id=1, frame_id=100
)
# Returns: landmarks_3d, body_proportions, feature_vector
```

**Body Measurements Extracted:**
- Shoulder width, hip width, torso height
- Leg lengths (left/right)
- Shoulder-to-torso ratio, hip-to-torso ratio

---

### **Module 4: Player Tracking**
**File:** `module4_tracking.py`

**Key Classes:**
- `KalmanBoxTracker`: Kalman filter for smooth tracking
- `PlayerTracker`: Multi-object tracker (simplified BoT-SORT)

**Key Features:**
- Kalman filtering for prediction
- Hungarian algorithm for data association
- Track history management
- Visualization with trails

**Parameters:**
```python
tracker = PlayerTracker(
    max_age=30,        # Max frames without detection
    min_hits=3,        # Min detections to confirm track
    iou_threshold=0.3  # IoU threshold for matching
)
```

**Usage:**
```python
tracked_objects = tracker.update(detections)
vis_frame = tracker.visualize_tracks(frame, tracked_objects)
```

---

### **Module 5-6: Re-Identification System**
**File:** `module5_6_reid.py`

**Key Classes:**
- `PlayerDatabase`: Identity storage and management
- `ReIdentificationSystem`: Multi-cue similarity matching
- `InterruptionDetector`: Track loss/recovery detection

**Key Features:**
- **Combined similarity scoring:**
  - 40% mesh/pose similarity
  - 40% jersey number matching
  - 20% body proportions
- Configurable re-ID threshold (default: 0.7)
- Automatic new player creation
- Track interruption handling

**Usage:**
```python
reid_system = ReIdentificationSystem(
    mesh_weight=0.4,
    jersey_weight=0.4,
    proportion_weight=0.2,
    reid_threshold=0.7
)

player_id, is_reassigned = reid_system.process_detection(
    tracking_id, detection_data, frame_id
)
```

**Re-ID Decision Logic:**
```
1. Check if tracking_id has existing player_id
2. If lost/new → compute similarity with all recent players
3. If similarity > threshold → reassign to best match
4. Else → create new player identity
```

---

### **Module 7: Database & Record Keeping**
**File:** `module7_database.py`

**Key Classes:**
- `MatchDatabase`: SQLite database manager
- `FileSystemManager`: Organized file storage
- `MatchRecorder`: High-level recording interface

**Database Schema:**
- **players**: Player identities and metadata
- **detections**: Frame-by-frame detections
- **features**: Serialized feature vectors
- **crops**: File paths to saved images
- **tracking_history**: Tracking ID assignments
- **reid_events**: Re-identification events log

**Usage:**
```python
recorder = MatchRecorder(match_id=1, output_dir='output')
recorder.record_detection(player_id, tracking_id, frame_id, data)
report = recorder.generate_report()
```

**Export Options:**
- JSON export of entire database
- SQLite queries for custom analysis
- Automatic file organization

---

### **Module 8: Evaluation & Benchmarking**
**File:** `module8_evaluation.py`

**Key Classes:**
- `TrackingEvaluator`: MOTA, IDF1, ID switches
- `ReIDEvaluator`: Re-ID accuracy metrics
- `BenchmarkLoader`: Dataset loading utilities
- `PerformanceAnalyzer`: Report generation

**Metrics Computed:**
- **MOTA** (Multiple Object Tracking Accuracy)
- **IDF1** (ID F1 Score)
- **ID Switches**: Identity swap count
- **Precision/Recall**: Detection accuracy
- **Re-ID Accuracy**: Successful re-identifications

**Usage:**
```python
evaluator = TrackingEvaluator()
evaluator.add_ground_truth(frame_id, gt_annotations)
evaluator.add_predictions(frame_id, predictions)

metrics = evaluator.compute_metrics()
# Returns: MOTA, IDF1, ID_Switches, Precision, Recall
```

---

## 🔧 Configuration Tips

### Performance Tuning

**For Speed:**
```python
CONFIG = {
    'yolo_model': 'yolov8n.pt',  # Nano model
    'skip_frames': 5,              # Process every 5th frame
    'conf_threshold': 0.6,         # Higher confidence
    'iou_threshold': 0.4           # More lenient matching
}
```

**For Accuracy:**
```python
CONFIG = {
    'yolo_model': 'yolov8x.pt',  # Extra-large model
    'skip_frames': 1,              # Process all frames
    'conf_threshold': 0.3,         # Lower confidence
    'reid_threshold': 0.75,        # Stricter re-ID
    'max_age': 50                  # Longer track persistence
}
```

### Re-ID Weight Tuning

```python
# If jersey numbers are very reliable
reid_system = ReIdentificationSystem(
    mesh_weight=0.2,
    jersey_weight=0.6,
    proportion_weight=0.2
)

# If players change jerseys / numbers unreliable
reid_system = ReIdentificationSystem(
    mesh_weight=0.5,
    jersey_weight=0.1,
    proportion_weight=0.4
)
```

---

## 📊 Output Structure

```
match_output/
├── match_1.db                    # SQLite database
├── faces/                        # Face crops
│   └── player_1_frame_100.jpg
├── bodies/                       # Body crops
│   └── player_1_frame_100.jpg
├── meshes/                       # Pose data
│   └── player_1_frame_100.pkl
├── tracking_videos/              # Annotated videos
│   └── tracked_output.mp4
├── reports/                      # Analysis reports
│   ├── match_1_report.json
│   └── processing_stats.json
└── evaluation/                   # Evaluation results
    └── evaluation_report.json
```

---

## 🎯 Common Use Cases

### 1. Process Single Match
```python
pipeline = FootballAnalysisPipeline(match_id=1)
pipeline.process_video('match.mp4', max_frames=None)  # Full video
pipeline.save_output_video()
pipeline.generate_final_report()
```

### 2. Real-Time Analysis (Live Stream)
```python
cap = cv2.VideoCapture(0)  # Or RTSP stream
while cap.isOpened():
    ret, frame = cap.read()
    results = pipeline.process_frame(frame, frame_id)
    # Display or stream results
```

### 3. Batch Processing Multiple Matches
```python
for match_id, video_path in enumerate(video_list):
    pipeline = FootballAnalysisPipeline(match_id=match_id)
    pipeline.process_video(video_path)
    pipeline.cleanup()
```

### 4. Re-ID Only (Skip Tracking)
```python
reid_system = ReIdentificationSystem()
for detection in detections:
    player_id, _ = reid_system.process_detection(
        tracking_id, detection_data, frame_id
    )
```

---

## 🐛 Debugging & Troubleshooting

### Common Issues

**1. Low Detection Rate**
- Lower `conf_threshold` in PlayerDetector
- Check video quality/resolution
- Ensure YOLO model is properly loaded

**2. Jersey Number OCR Fails**
- Adjust torso region in `extract_jersey_number()`
- Try different preprocessing strategies
- Check jersey visibility in crops

**3. Too Many ID Switches**
- Increase `reid_threshold` (more strict)
- Adjust similarity weights
- Increase `max_age` in tracker

**4. Slow Processing**
- Use smaller YOLO model (yolov8n)
- Increase `skip_frames`
- Reduce pose model complexity
- Disable visualization during processing

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Process with debug info
results = pipeline.process_frame(frame, frame_id)
print(f"Detections: {len(results['detections'])}")
print(f"Processing time: {results['processing_time']:.3f}s")
```

---

## 📈 Performance Benchmarks

**Expected Performance (GPU):**
- Detection (YOLOv8x): ~30ms per frame
- Pose estimation: ~50ms per player
- Tracking: ~5ms per frame
- Re-ID: ~10ms per player
- **Total: ~100-200ms per frame** (5-10 FPS)

**Optimization Targets:**
- Real-time (30 FPS): Use YOLOv8n + skip tracking
- High accuracy: Use YOLOv8x + all modules
- Balanced: YOLOv8m + skip frames=2

---

## 🔬 Advanced Features

### Custom Similarity Functions

```python
class CustomReID(ReIdentificationSystem):
    def compute_custom_similarity(self, feat1, feat2):
        # Add your custom similarity logic
        return custom_score
```

### Database Queries

```python
# Query all detections for player
cursor = recorder.db.conn.cursor()
cursor.execute("""
    SELECT frame_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2
    FROM detections
    WHERE player_id = ?
    ORDER BY frame_id
""", (player_id,))
```

### Export for ML Training

```python
# Export features for training custom re-ID model
features = []
labels = []
for player_id in database.players:
    player = database.players[player_id]
    features.extend(player.mesh_features)
    labels.extend([player_id] * len(player.mesh_features))

# Train custom classifier
# model.fit(features, labels)
```

---

## 📚 References & Datasets

**Datasets for Testing:**
- **SoccerNet**: https://www.soccer-net.org/
- **SportsReid**: Sports-specific re-ID dataset
- **MOT Challenge**: Multi-object tracking benchmark

**Key Papers:**
- BoT-SORT: "BoT-SORT: Robust Associations Multi-Pedestrian Tracking"
- SMPL: "Keep it SMPL: Automatic Estimation of 3D Human Pose"
- DeepSORT: "Simple Online and Realtime Tracking with a Deep Association Metric"

---

## 🤝 Contributing

To extend the pipeline:
1. Add new module following the pattern
2. Integrate into `FootballAnalysisPipeline`
3. Update evaluation metrics
4. Add tests with sample data

---

## ⚡ Next Steps

1. **Test with sample video**: Start with 100 frames
2. **Tune parameters**: Adjust thresholds for your use case
3. **Evaluate results**: Use Module 8 for metrics
4. **Scale up**: Process full matches
5. **Customize**: Add team classification, action recognition, etc.

**Happy Analyzing! ⚽**