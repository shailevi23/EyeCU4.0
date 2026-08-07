# Quick Start Guide

Get started with Football Analysis MVP in 5 minutes!

## 🚀 Quick Installation

```bash
# 1. Clone and setup
git clone <repository-url>
cd football_analysis_mvp
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## ⚡ Quick Run

### Option 1: Command Line (Fastest)

```bash
# Basic analysis
python run_pipeline.py --input your_video.mp4

# With highlights
python run_pipeline.py \
    --input your_video.mp4 \
    --generate-highlights \
    --show-player-ids
```

### Option 2: Web Interface (Easiest)

```bash
# Launch web app
streamlit run main.py

# Then:
# 1. Upload your video
# 2. Adjust settings in sidebar
# 3. Click "Start Analysis"
# 4. Download results
```

## 📝 Example Workflows

### Workflow 1: Quick Match Analysis

```bash
# Fast analysis with highlights
python run_pipeline.py \
    --input match.mp4 \
    --skip-frames 3 \
    --max-frames 1000 \
    --generate-highlights
```

**Output:**
- `output/tracked_videos/tracked_output.mp4` - Annotated video
- `output/highlights/` - Highlight clips
- `output/reports/` - JSON reports

### Workflow 2: Full Match Analysis

```bash
# Complete analysis with all features
python run_pipeline.py \
    --input full_match.mp4 \
    --skip-frames 2 \
    --team-method hybrid \
    --ocr-engine paddleocr \
    --detect-goals \
    --detect-shots \
    --detect-passes \
    --generate-highlights \
    --show-player-ids \
    --show-speed
```

### Workflow 3: Team Assignment Focus

```bash
# Focus on accurate team assignment
python run_pipeline.py \
    --input match.mp4 \
    --team-method hybrid \
    --hamza-weight 0.6 \
    --eyecu-weight 0.4 \
    --skip-frames 1
```

### Workflow 4: Event Detection Only

```bash
# Quick event detection
python run_pipeline.py \
    --input match.mp4 \
    --skip-frames 5 \
    --detect-goals \
    --detect-shots \
    --generate-highlights
```

## 🎯 Understanding Output

### 1. Tracked Video
Location: `output/tracked_videos/tracked_output.mp4`

Features:
- Colored bounding boxes (team colors)
- Player IDs (jersey numbers)
- Real-time event annotations
- Speed/distance (if enabled)

### 2. Highlight Clips
Location: `output/highlights/`

Files:
- `highlight_000_goal_frame1234.mp4` - Video clip
- `highlight_000_goal_frame1234_metadata.json` - Clip info

### 3. Reports
Location: `output/reports/`

Files:
- `summary.json` - Overall statistics
- `events.json` - All detected events
- `teams.json` - Team assignment stats
- `players.json` - Player identification info

## 📊 Reading Reports

### Check Event Summary
```bash
cat output/reports/events.json | python -m json.tool
```

Example output:
```json
{
  "events": [
    {
      "event_type": "goal",
      "timestamp": 41.1,
      "player_id": "0_10",
      "team_id": 0
    }
  ],
  "summary": {
    "total_events": 45,
    "goals": 2,
    "shots": 8,
    "passes": 35
  }
}
```

### Check Player Stats
```bash
cat output/reports/players.json | python -m json.tool
```

### View Processing Summary
```bash
cat output/reports/summary.json | python -m json.tool
```

## 🔧 Common Customizations

### Adjust Team Colors

Edit `config/config.py`:
```python
config.visualization.team_a_color = (255, 0, 0)  # Blue in BGR
config.visualization.team_b_color = (0, 0, 255)  # Red in BGR
```

### Change Event Sensitivity

```python
# More sensitive goal detection
config.event_detection.goal_zone_depth = 7.0  # Default: 5.0

# Require higher shot speed
config.event_detection.shot_speed_threshold = 20.0  # Default: 15.0
```

### Modify Highlight Length

```bash
# Longer highlights
python run_pipeline.py \
    --input match.mp4 \
    --highlight-buffer-pre 5.0 \
    --highlight-buffer-post 10.0
```

## 🎨 Visualization Options

### Minimal (Fast)
```bash
python run_pipeline.py \
    --input match.mp4 \
    --skip-frames 5
```

### Standard
```bash
python run_pipeline.py \
    --input match.mp4 \
    --show-player-ids
```

### Full (Detailed)
```bash
python run_pipeline.py \
    --input match.mp4 \
    --show-player-ids \
    --show-speed \
    --show-distance \
    --display  # Real-time preview
```

## 💡 Performance Tips

### For Speed
```bash
# Process fewer frames
python run_pipeline.py --input match.mp4 --skip-frames 5

# Limit total frames
python run_pipeline.py --input match.mp4 --max-frames 500

# Use smaller YOLO model
python run_pipeline.py --input match.mp4 --yolo-model yolov8n.pt
```

### For Accuracy
```bash
# Process more frames
python run_pipeline.py --input match.mp4 --skip-frames 1

# Use hybrid team assignment
python run_pipeline.py --input match.mp4 --team-method hybrid

# Use PaddleOCR
python run_pipeline.py --input match.mp4 --ocr-engine paddleocr
```

## 🐛 Quick Troubleshooting

### GPU Not Detected
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory
```bash
# Use smaller batch / more skipping
python run_pipeline.py --input match.mp4 --skip-frames 5

# Or use smaller model
python run_pipeline.py --input match.mp4 --yolo-model yolov8s.pt
```

### PaddleOCR Error
```bash
# Install PaddlePaddle
pip install paddlepaddle-gpu  # GPU version
# OR
pip install paddlepaddle  # CPU version
```

### Poor Results
```bash
# Try different team assignment
python run_pipeline.py --input match.mp4 --team-method color

# Adjust confidence thresholds
python run_pipeline.py --input match.mp4 --confidence 0.5 --ocr-confidence 0.6
```

## 📱 Using Streamlit App

### Launch
```bash
streamlit run main.py
```

### Workflow
1. **Upload Video**: Click "Browse files" and select your video
2. **Configure**: Adjust settings in left sidebar
   - Processing options
   - Team assignment method
   - OCR settings
   - Event detection
   - Visualization preferences
3. **Process**: Click "Start Analysis" button
4. **Review**: Browse results in tabs:
   - Overview: Summary and downloads
   - Teams: Team assignment stats
   - Players: Player identification
   - Events: Detected events and highlights
5. **Download**: Get tracked video, reports, and highlights

## 🎓 Learning Path

### Beginner
1. Run basic analysis with default settings
2. View tracked video output
3. Check summary report

### Intermediate
1. Experiment with different team assignment methods
2. Adjust event detection thresholds
3. Generate and review highlight clips

### Advanced
1. Integrate with EyeCU4.0 tracking
2. Customize event detection logic
3. Extend with new analysis features
4. Optimize for your specific use case

## 📚 Next Steps

- Read full [README.md](README.md) for detailed documentation
- Check [config/config.py](config/config.py) for all options
- Review example outputs in `output/` directory
- Explore API in `core/pipeline.py`

## 💬 Getting Help

1. Check troubleshooting section above
2. Review full README.md
3. Check configuration in config.py
4. Open GitHub issue with:
   - Error message
   - Command used
   - Video properties
   - System info

---

**Happy analyzing! ⚽**