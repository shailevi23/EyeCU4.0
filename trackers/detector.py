"""
Detector interface for EyeCU.

One interface, two backends:

    LocalDetector      Ultralytics YOLO running locally. The production path.
    RoboflowDetector   Hosted model. Optional, opt-in, used only as a
                       labelling/benchmark aid (TODO.md section 5).

Every backend emits the same four classes and never collapses one into
another -- in particular `goalkeeper` stays `goalkeeper`. Team identity is not
a detector class; it is decided later in trackers/team_assigner.py.

Detection format (one dict per object):

    {'bbox': [x1, y1, x2, y2], 'class': 'player', 'confidence': 0.87}
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

# Fixed project-wide. Must match data/dataset/football.yaml and tools/pseudo_label.py.
CLASSES = ['player', 'goalkeeper', 'referee', 'ball']
CLASS_IDS = {name: i for i, name in enumerate(CLASSES)}

# Roles that are people. Used by duplicate suppression and team assignment.
HUMAN_CLASSES = ('player', 'goalkeeper', 'referee')

# Whatever a source model happens to call things -> our four classes.
# `person` maps to `player` because a COCO model cannot tell roles apart; a
# football-specific model reports the real role and keeps it.
ALIASES = {
    'player': 'player', 'players': 'player', 'football player': 'player',
    'outfield player': 'player', 'person': 'player',
    'goalkeeper': 'goalkeeper', 'goal keeper': 'goalkeeper', 'gk': 'goalkeeper',
    'referee': 'referee', 'ref': 'referee', 'main referee': 'referee',
    'side referee': 'referee', 'assistant referee': 'referee',
    'ball': 'ball', 'sports ball': 'ball', 'football': 'ball', 'soccer ball': 'ball',
}


def normalize_class(name) -> Optional[str]:
    """Map a source model's label to one of CLASSES, or None to drop it."""
    return ALIASES.get(str(name).strip().lower())


class BaseDetector(ABC):
    """Detects players, goalkeepers, referees and the ball in a single frame."""

    def __init__(self, confidence: float = 0.25):
        self.confidence = confidence
        self.detections_cache: Dict[int, List[Dict]] = {}
        self.inference_count = 0
        self.total_inference_time = 0.0

    @abstractmethod
    def _predict(self, image: np.ndarray) -> List[Dict]:
        """Backend-specific inference. Returns detections in the standard format."""

    def detect(self, image: np.ndarray, frame_id: Optional[int] = None) -> List[Dict]:
        """Detect objects, reusing the cached result when frame_id repeats."""
        if frame_id is not None and frame_id in self.detections_cache:
            return self.detections_cache[frame_id]

        start = time.time()
        detections = self._predict(image)
        self.total_inference_time += time.time() - start
        self.inference_count += 1

        if frame_id is not None:
            self.detections_cache[frame_id] = detections
        return detections

    def clear_cache(self) -> None:
        self.detections_cache.clear()

    @property
    def avg_inference_time(self) -> float:
        if not self.inference_count:
            return 0.0
        return self.total_inference_time / self.inference_count

    def stats(self) -> Dict:
        return {
            'backend': type(self).__name__,
            'inferences': self.inference_count,
            'avg_inference_time': round(self.avg_inference_time, 4),
            'fps': round(1 / self.avg_inference_time, 2) if self.avg_inference_time else 0,
        }


class LocalDetector(BaseDetector):
    """Ultralytics YOLO running locally. No network access required."""

    def __init__(self, model_path: str = 'yolov8s.pt',
                 confidence: float = 0.25,
                 iou: float = 0.5,
                 imgsz: int = 960,
                 device: Optional[str] = None):
        super().__init__(confidence)
        from ultralytics import YOLO

        self.model_path = model_path
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.model = YOLO(model_path)

        # Resolve the model's own label space once.
        self._class_map = {i: normalize_class(n) for i, n in self.model.names.items()}
        self._class_map = {i: n for i, n in self._class_map.items() if n}
        if not self._class_map:
            raise ValueError(
                f'{model_path} has no classes that map to {CLASSES}. '
                f'Model classes: {list(self.model.names.values())[:10]}'
            )

        available = sorted(set(self._class_map.values()))
        print(f'LocalDetector: {model_path} @ {imgsz}px, classes {available}')
        missing = [c for c in CLASSES if c not in available]
        if missing:
            print(f'  note: this model cannot produce {missing} '
                  f'(a COCO model only knows person/sports ball). '
                  f'Fine-tune on the football dataset to get all four.')

    def _predict(self, image: np.ndarray) -> List[Dict]:
        kwargs = dict(conf=self.confidence, iou=self.iou,
                      imgsz=self.imgsz, verbose=False)
        if self.device:
            kwargs['device'] = self.device
        result = self.model.predict(image, **kwargs)[0]

        detections = []
        for box in result.boxes:
            name = self._class_map.get(int(box.cls))
            if not name:
                continue  # a COCO class we do not care about
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class': name,
                'confidence': float(box.conf),
            })
        return detections


class RoboflowDetector(BaseDetector):
    """
    Hosted Roboflow model. Optional labelling/benchmark aid only -- it must not
    be required for normal inference.

    Needs ROBOFLOW_API_KEY (or an explicit api_key). Falls back to `fallback`
    when a request fails, so a flaky network cannot abort a run.
    """

    URL = 'https://detect.roboflow.com'

    def __init__(self, model_id: str = 'football-players-detection-3zvbc/12',
                 api_key: Optional[str] = None,
                 confidence: float = 0.25,
                 iou: float = 0.5,
                 fallback: Optional[BaseDetector] = None,
                 timeout: int = 30):
        super().__init__(confidence)
        import requests

        self._requests = requests
        self.api_key = api_key or os.environ.get('ROBOFLOW_API_KEY')
        if not self.api_key:
            raise ValueError(
                'No Roboflow API key. Set ROBOFLOW_API_KEY or pass api_key. '
                'Roboflow is optional -- use LocalDetector for normal runs.'
            )
        self.model_id = model_id
        self.iou = iou
        self.timeout = timeout
        self.fallback = fallback
        self.failure_count = 0
        print(f'RoboflowDetector: {model_id} (labelling/benchmark use only)')

    def _predict(self, image: np.ndarray) -> List[Dict]:
        import base64

        import cv2

        ok, buf = cv2.imencode('.jpg', image)
        if not ok:
            raise RuntimeError('Could not JPEG-encode frame for Roboflow')

        try:
            response = self._requests.post(
                f'{self.URL}/{self.model_id}',
                params={'api_key': self.api_key,
                        'confidence': int(self.confidence * 100),
                        'overlap': int(self.iou * 100),
                        'format': 'json'},
                data=base64.b64encode(buf.tobytes()),
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=self.timeout,
            )
            response.raise_for_status()
            predictions = response.json().get('predictions', [])
        except Exception as e:
            self.failure_count += 1
            if self.fallback is None:
                raise
            print(f'  Roboflow request failed ({e}); using {type(self.fallback).__name__}')
            return self.fallback._predict(image)

        detections = []
        for pred in predictions:
            name = normalize_class(pred.get('class', ''))
            if not name:
                continue
            cx, cy = pred['x'], pred['y']
            w, h = pred['width'], pred['height']
            detections.append({
                'bbox': [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                'class': name,
                'confidence': float(pred.get('confidence', 0.0)),
            })
        return detections

    def stats(self) -> Dict:
        s = super().stats()
        s['failures'] = self.failure_count
        return s


def create_detector(model_path: str = 'yolov8s.pt',
                    use_roboflow: bool = False,
                    api_key: Optional[str] = None,
                    confidence: float = 0.25,
                    iou: float = 0.5,
                    imgsz: int = 960,
                    model_id: Optional[str] = None) -> BaseDetector:
    """
    Build the detector for a run.

    Local by default. `use_roboflow=True` opts in to the hosted model and keeps
    a LocalDetector as the fallback, so a failed request degrades instead of
    killing the run.
    """
    local = LocalDetector(model_path=model_path, confidence=confidence,
                          iou=iou, imgsz=imgsz)
    if not use_roboflow:
        return local

    try:
        return RoboflowDetector(
            model_id=model_id or 'football-players-detection-3zvbc/12',
            api_key=api_key, confidence=confidence, iou=iou, fallback=local)
    except Exception as e:
        print(f'Roboflow unavailable ({e}); continuing with the local detector.')
        return local
