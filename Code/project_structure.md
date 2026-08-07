"""
MVP Football Analysis System - Project Structure
Integrates EyeCU4.0 with Hamza's features + Jersey OCR + Event Detection
"""

# Directory Structure:
"""
football_analysis_mvp/
├── main.py                          # Entry point - Streamlit app
├── run_pipeline.py                  # CLI interface
├── requirements.txt
├── README.md
├── config/
│   ├── __init__.py
│   └── config.py                    # Configuration settings
├── core/
│   ├── __init__.py
│   ├── detector.py                  # YOLOv8 + Roboflow detection
│   ├── tracker.py                   # Multi-object tracking (EyeCU4.0)
│   └── pipeline.py                  # Main pipeline orchestrator
├── team_assignment/
│   ├── __init__.py
│   ├── color_clustering.py          # Hamza's color clustering
│   ├── eyecu_assigner.py           # EyeCU4.0 team assignment
│   └── hybrid_assigner.py          # Arbitration logic
├── player_id/
│   ├── __init__.py
│   ├── jersey_ocr.py               # OCR for jersey numbers
│   └── id_manager.py               # Persistent ID tracking
├── events/
│   ├── __init__.py
│   ├── event_detector.py           # Goal, proximity, action detection
│   └── highlight_generator.py      # Create highlight clips
├── utils/
│   ├── __init__.py
│   ├── video_utils.py              # Video I/O operations
│   ├── visualization.py            # Drawing overlays
│   └── export.py                   # JSON/CSV exports
├── models/
│   └── yolov8x.pt                  # YOLO weights
├── input_videos/
├── output/
│   ├── tracked_videos/
│   ├── highlights/
│   ├── reports/
│   └── cache/
└── tests/
    ├── __init__.py
    └── test_pipeline.py
"""

# Key Dependencies:
"""
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
pytesseract>=0.3.10
paddleocr>=2.7.0
streamlit>=1.28.0
supervision>=0.16.0
torch>=2.0.0
torchvision>=0.15.0
"""