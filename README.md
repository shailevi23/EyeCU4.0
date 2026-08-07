# EyeCU 4.0 Football Analysis Pipeline

## Overview
This is an advanced football analysis pipeline that performs player detection, tracking, team assignment, and speed/distance calculation from football match videos. The system uses state-of-the-art computer vision techniques to identify players, track their movements, and analyze their performance.

## Features
- Player detection using YOLOv8 and Roboflow API
- Advanced multi-object tracking
- Team assignment for player identification
- Speed and distance calculation
- Camera movement estimation
- Ball detection and tracking
- Visualization options

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup
1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the YOLOv8 model (if not using Roboflow API exclusively)
   ```
   # Models are downloaded automatically when first used
   ```

## Usage

### Quick Start
For a quick test of the pipeline, use one of the provided scripts:
- Windows CMD: `quick_test.bat`
- PowerShell: `.\quick_test.ps1`

### Command Line Interface
The `run_pipeline.py` script provides a command-line interface for running the football analysis pipeline:

```
python run_pipeline.py --input INPUT_VIDEO --output OUTPUT_FILENAME [OPTIONS]
```

#### Required Arguments
- `--input`: Path to input video file

#### Optional Arguments
- `--output`: Output video filename (default: tracked_output.mp4)
- `--output-dir`: Output directory for all results (default: match_analysis_output)
- `--max-frames`: Maximum frames to process (default: all frames)
- `--skip-frames`: Process every N frames (default: 2)
- `--display`: Display real-time visualization (default: False)
- `--use-cache`: Use cached detections/tracks if available (default: False)
- `--use-roboflow`: Opt in to the Roboflow cloud detector (default: off — local YOLO only)
- `--api-key`: Roboflow API key (defaults to the `ROBOFLOW_API_KEY` environment variable)
- `--yolo-model`: YOLO model path (default: yolov8x.pt)
- `--show-speed`: Show player speed in visualization
- `--show-distance`: Show distance traveled in visualization
- `--fps`: Output video FPS (default: 15)

### Examples

Process a complete video with advanced tracking:
```
python run_pipeline.py --input input-videos/match.mp4 --output tracked_output.mp4
```

Process with limited frames for faster testing:
```
python run_pipeline.py --input input-videos/match.mp4 --max-frames 100 --skip-frames 3
```

Show player speed and distance in visualization:
```
python run_pipeline.py --input input-videos/match.mp4 --show-speed --show-distance
```

## Output Structure
The pipeline generates output in the specified directory (default: match_analysis_output):

- `tracked_output.mp4`: Main output video with tracking visualization
- `visualizations/`: Selected frames saved as images
- `reports/`: Analysis reports and statistics in JSON format
- `cache/`: Cached detections and intermediate results
- `final_report.json`: Summary of processing statistics and results

## Advanced Usage

### Direct API Access
You can also use the `FootballAnalysisPipeline` class directly in your Python code:

```python
from full_pipeline import FootballAnalysisPipeline

# Initialize the pipeline
pipeline = FootballAnalysisPipeline(
    yolo_model="yolov8x.pt",
    output_dir="my_results",
    use_advanced_tracking=True,
    use_roboflow=True
)

# Process a video
stats = pipeline.process_video(
    video_path="my_video.mp4",
    skip_frames=2,
    max_frames=None,  # Process all frames
    display_results=True
)

# Save the results
pipeline.save_output_video("output.mp4", fps=30)
pipeline.generate_final_report()
pipeline.cleanup()
```

## Customization
- Modify tracking parameters in `trackers/football_tracker.py`
- Adjust team assignment in `trackers/team_assigner.py`
- Configure speed calculation in `trackers/speed_distance.py`

## Troubleshooting
- **No detection results**: Verify your Roboflow API key and connection
- **Slow processing**: Increase skip_frames value or use a smaller YOLO model
- **Memory errors**: Reduce max_frames or batch size in detector