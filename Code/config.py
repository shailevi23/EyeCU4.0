"""
config/config.py
Configuration settings for the Football Analysis MVP
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional

# Check for CUDA availability
CUDA_AVAILABLE = False
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass

@dataclass
class DetectionConfig:
    """Detection model configuration"""
    yolo_model: str = "yolov8x.pt"
    use_roboflow: bool = True
    roboflow_api_key: Optional[str] = None
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.5
    
    # Class IDs
    player_class_id: int = 0
    ball_class_id: int = 1
    referee_class_id: int = 2
    goalkeeper_class_id: int = 3

@dataclass
class TrackingConfig:
    """Tracking configuration"""
    tracker_type: str = "bytetrack"  # or "botsort"
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3

@dataclass
class TeamAssignmentConfig:
    """Team assignment configuration"""
    use_hybrid: bool = True  # Use both EyeCU + Hamza
    hamza_weight: float = 0.5
    eyecu_weight: float = 0.5
    confidence_threshold: float = 0.6
    
    # Color clustering params
    n_clusters: int = 3  # Teams + referee
    color_space: str = "HSV"  # or "RGB", "LAB"
    roi_ratio: float = 0.6  # Crop ratio for jersey area

@dataclass
class OCRConfig:
    """Jersey OCR configuration"""
    ocr_engine: str = "paddleocr"  # or "tesseract"
    ocr_language: str = "en"
    confidence_threshold: float = 0.5
    number_range: Tuple[int, int] = (1, 99)
    
    # Tesseract config
    tesseract_config: str = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
    
    # PaddleOCR config
    paddle_use_gpu: bool = CUDA_AVAILABLE  # Use GPU only if available
    paddle_lang: str = "en"

@dataclass
class EventDetectionConfig:
    """Event detection configuration"""
    goal_zone_width: float = 7.32  # meters
    goal_zone_depth: float = 5.0   # meters
    ball_proximity_threshold: float = 2.0  # meters
    
    # Event types
    detect_goals: bool = True
    detect_passes: bool = True
    detect_tackles: bool = False
    detect_shots: bool = True
    
    # Speed thresholds
    sprint_speed: float = 5.5  # m/s
    walk_speed: float = 2.0    # m/s

@dataclass
class HighlightConfig:
    """Highlight generation configuration"""
    pre_event_buffer: float = 3.0   # seconds before event
    post_event_buffer: float = 5.0  # seconds after event
    min_highlight_duration: float = 5.0
    max_highlight_duration: float = 15.0
    
    # Priority levels
    goal_priority: int = 10
    shot_priority: int = 7
    pass_priority: int = 5

@dataclass
class VideoConfig:
    """Video processing configuration"""
    skip_frames: int = 2
    max_frames: Optional[int] = None
    output_fps: int = 15
    output_resolution: Optional[Tuple[int, int]] = None
    display_realtime: bool = False

@dataclass
class VisualizationConfig:
    """Visualization settings"""
    show_speed: bool = True
    show_distance: bool = True
    show_player_ids: bool = True
    show_team_colors: bool = True
    show_events: bool = True
    
    # Colors (BGR format)
    team_a_color: Tuple[int, int, int] = (255, 0, 0)    # Blue
    team_b_color: Tuple[int, int, int] = (0, 0, 255)    # Red
    ball_color: Tuple[int, int, int] = (0, 255, 255)    # Yellow
    referee_color: Tuple[int, int, int] = (0, 255, 0)   # Green
    
    # Drawing params
    bbox_thickness: int = 2
    text_scale: float = 0.6
    text_thickness: int = 2

@dataclass
class OutputConfig:
    """Output configuration"""
    output_dir: str = "output"
    tracked_video_dir: str = "tracked_videos"
    highlights_dir: str = "highlights"
    reports_dir: str = "reports"
    cache_dir: str = "cache"
    
    # Export formats
    export_json: bool = True
    export_csv: bool = True
    save_visualizations: bool = True
    
    # Cache settings
    use_cache: bool = False
    cache_detections: bool = True
    cache_tracks: bool = True

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.detection = DetectionConfig()
        self.tracking = TrackingConfig()
        self.team_assignment = TeamAssignmentConfig()
        self.ocr = OCRConfig()
        self.event_detection = EventDetectionConfig()
        self.highlight = HighlightConfig()
        self.video = VideoConfig()
        self.visualization = VisualizationConfig()
        self.output = OutputConfig()
        
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary output directories"""
        base_dir = self.output.output_dir
        dirs = [
            base_dir,
            os.path.join(base_dir, self.output.tracked_video_dir),
            os.path.join(base_dir, self.output.highlights_dir),
            os.path.join(base_dir, self.output.reports_dir),
            os.path.join(base_dir, self.output.cache_dir),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def update_from_dict(self, config_dict: dict):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                section = getattr(self, key)
                if isinstance(value, dict):
                    for k, v in value.items():
                        if hasattr(section, k):
                            setattr(section, k, v)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            "detection": self.detection.__dict__,
            "tracking": self.tracking.__dict__,
            "team_assignment": self.team_assignment.__dict__,
            "ocr": self.ocr.__dict__,
            "event_detection": self.event_detection.__dict__,
            "highlight": self.highlight.__dict__,
            "video": self.video.__dict__,
            "visualization": self.visualization.__dict__,
            "output": self.output.__dict__,
        }

# Global config instance
config = Config()