"""
Module 4: Player Tracking Across Frames
Dependencies: pip install filterpy lap scipy numpy
For BoT-SORT: pip install boxmot (or implement simplified version)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple
import cv2

class KalmanBoxTracker:
    """
    Kalman Filter for tracking bounding boxes
    State: [x, y, w, h, vx, vy, vw, vh]
    """
    count = 0
    
    def __init__(self, bbox: List[float]):
        """
        Initialize tracker with bounding box
        Args:
            bbox: [x1, y1, x2, y2]
        """
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0]
        ])
        
        # Measurement noise
        self.kf.R *= 10
        
        # Process noise
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        # Initial state
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox: List[float]):
        """Update tracker with new detection"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
        
    def predict(self):
        """Advance state and return predicted bbox"""
        if self.kf.x[2] + self.kf.x[6] <= 0:
            self.kf.x[6] *= 0.0
        if self.kf.x[3] + self.kf.x[7] <= 0:
            self.kf.x[7] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self):
        """Get current bbox estimate"""
        return self._convert_x_to_bbox(self.kf.x)
    
    @staticmethod
    def _convert_bbox_to_z(bbox):
        """Convert [x1,y1,x2,y2] to [x,y,w,h]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2
        y = bbox[1] + h/2
        return np.array([x, y, w, h]).reshape((4, 1))
    
    @staticmethod
    def _convert_x_to_bbox(x):
        """Convert [x,y,w,h] to [x1,y1,x2,y2]"""
        w = x[2]
        h = x[3]
        return np.array([
            x[0] - w/2,
            x[1] - h/2,
            x[0] + w/2,
            x[1] + h/2
        ]).flatten()


class PlayerTracker:
    """
    Multi-object tracker for football players
    Simplified BoT-SORT implementation
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, 
                 iou_threshold: float = 0.3):
        """
        Initialize tracker
        Args:
            max_age: Maximum frames to keep alive without detection
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        
        # Track history for re-identification
        self.track_history = defaultdict(lambda: {
            'bboxes': deque(maxlen=100),
            'features': deque(maxlen=10),
            'last_seen': 0,
            'status': 'active'  # active, lost, terminated
        })
        
    @staticmethod
    def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _match_detections_to_trackers(self, detections: np.ndarray, 
                                     trackers: np.ndarray) -> Tuple:
        """
        Match detections to trackers using IoU
        Returns:
            (matches, unmatched_detections, unmatched_trackers)
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty(0, dtype=int)
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.iou(det, trk)
        
        # Hungarian algorithm for assignment
        if min(iou_matrix.shape) > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.column_stack((row_ind, col_ind))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        # Filter out matches with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                continue
            matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        # Unmatched detections and trackers
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matches[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matches[:, 1]:
                unmatched_trackers.append(t)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections
        Args:
            detections: List of detection dicts with 'bbox', 'confidence', etc.
        Returns:
            List of tracked objects with tracking IDs
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Remove invalid trackers
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.delete(trks, to_del, axis=0)
        
        # Extract detection bboxes
        dets = np.array([d['bbox'] for d in detections]) if detections else np.empty((0, 4))
        
        # Match detections to trackers
        matched, unmatched_dets, unmatched_trks = self._match_detections_to_trackers(dets, trks)
        
        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)
        
        # Prepare output
        tracked_objects = []
        for trk in self.trackers:
            d = trk.get_state()
            
            # Only return confirmed tracks
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                tracked_objects.append({
                    'tracking_id': trk.id,
                    'bbox': d.tolist(),
                    'hits': trk.hits,
                    'age': trk.age,
                    'time_since_update': trk.time_since_update
                })
                
                # Update history
                self.track_history[trk.id]['bboxes'].append(d.tolist())
                self.track_history[trk.id]['last_seen'] = self.frame_count
                self.track_history[trk.id]['status'] = 'active'
        
        # Mark lost tracks
        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                self.track_history[trk.id]['status'] = 'lost'
        
        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]
        
        return tracked_objects
    
    def add_features_to_track(self, tracking_id: int, features: Dict):
        """
        Add visual/pose features to track history
        Args:
            tracking_id: Track ID
            features: Dictionary with 'pose_features', 'jersey_number', etc.
        """
        if tracking_id in self.track_history:
            self.track_history[tracking_id]['features'].append(features)
    
    def get_track_history(self, tracking_id: int) -> Optional[Dict]:
        """Get full history for a track"""
        if tracking_id in self.track_history:
            return dict(self.track_history[tracking_id])
        return None
    
    def visualize_tracks(self, frame: np.ndarray, 
                        tracked_objects: List[Dict],
                        show_ids: bool = True,
                        show_trails: bool = True) -> np.ndarray:
        """
        Visualize tracking results on frame
        Args:
            frame: Input frame
            tracked_objects: List of tracked objects from update()
            show_ids: Whether to show tracking IDs
            show_trails: Whether to show motion trails
        Returns:
            Annotated frame
        """
        vis = frame.copy()
        
        for obj in tracked_objects:
            bbox = obj['bbox']
            track_id = obj['tracking_id']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            color = self._get_color_for_id(track_id)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID
            if show_ids:
                label = f"ID: {track_id}"
                cv2.putText(vis, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trail
            if show_trails and track_id in self.track_history:
                history = self.track_history[track_id]['bboxes']
                if len(history) > 1:
                    points = []
                    for h_bbox in list(history)[-20:]:  # Last 20 positions
                        cx = int((h_bbox[0] + h_bbox[2]) / 2)
                        cy = int((h_bbox[1] + h_bbox[3]) / 2)
                        points.append((cx, cy))
                    
                    for i in range(1, len(points)):
                        cv2.line(vis, points[i-1], points[i], color, 2)
        
        return vis
    
    @staticmethod
    def _get_color_for_id(track_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for tracking ID"""
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color


# Example usage
if __name__ == "__main__":
    tracker = PlayerTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Process video
    cap = cv2.VideoCapture('soccer_match.mp4')
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get detections (from Module 1)
        detections = [
            {'bbox': [100, 100, 200, 300], 'confidence': 0.9},
            {'bbox': [300, 150, 400, 350], 'confidence': 0.85},
        ]
        
        # Update tracker
        tracked = tracker.update(detections)
        
        # Visualize
        vis_frame = tracker.visualize_tracks(frame, tracked)
        cv2.imshow('Tracking', vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_id += 1
    
    cap.release()
    cv2.destroyAllWindows()