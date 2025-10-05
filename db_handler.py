"""
Module 7: Database/Record Keeping System
Store and manage all player data across the match
"""

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import cv2

class MatchDatabase:
    """SQLite database for storing match and player data"""
    
    def __init__(self, db_path: str = "match_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        
    def create_tables(self):
        """Create database schema"""
        cursor = self.conn.cursor()
        
        # Players table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id INTEGER PRIMARY KEY,
                jersey_number TEXT,
                jersey_confidence REAL,
                first_seen_frame INTEGER,
                last_seen_frame INTEGER,
                total_appearances INTEGER,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Detections table (one row per detection)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                tracking_id INTEGER,
                frame_id INTEGER,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players (player_id)
            )
        """)
        
        # Features table (stores serialized features)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                frame_id INTEGER,
                feature_type TEXT,
                feature_data BLOB,
                FOREIGN KEY (player_id) REFERENCES players (player_id)
            )
        """)
        
        # Crops table (file paths to saved images)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crops (
                crop_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                tracking_id INTEGER,
                frame_id INTEGER,
                crop_type TEXT,
                file_path TEXT,
                FOREIGN KEY (player_id) REFERENCES players (player_id)
            )
        """)
        
        # Tracking history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracking_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                tracking_id INTEGER,
                start_frame INTEGER,
                end_frame INTEGER,
                FOREIGN KEY (player_id) REFERENCES players (player_id)
            )
        """)
        
        # Re-identification events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reid_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                old_tracking_id INTEGER,
                new_tracking_id INTEGER,
                frame_id INTEGER,
                similarity_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players (player_id)
            )
        """)
        
        # Match metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_metadata (
                match_id INTEGER PRIMARY KEY,
                video_path TEXT,
                fps REAL,
                total_frames INTEGER,
                start_time TIMESTAMP,
                end_time TIMESTAMP
            )
        """)
        
        self.conn.commit()
        
    def insert_player(self, player_id: int, jersey_number: str,
                     jersey_conf: float, first_frame: int) -> int:
        """Insert new player"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO players (player_id, jersey_number, jersey_confidence,
                               first_seen_frame, last_seen_frame, 
                               total_appearances, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (player_id, jersey_number, jersey_conf, first_frame, 
              first_frame, 1, 'active'))
        self.conn.commit()
        return player_id
    
    def update_player(self, player_id: int, last_frame: int, 
                     status: str = 'active'):
        """Update player's last seen frame"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE players 
            SET last_seen_frame = ?,
                total_appearances = total_appearances + 1,
                status = ?
            WHERE player_id = ?
        """, (last_frame, status, player_id))
        self.conn.commit()
    
    def insert_detection(self, player_id: int, tracking_id: int,
                        frame_id: int, bbox: List[float], 
                        confidence: float):
        """Insert detection record"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO detections (player_id, tracking_id, frame_id,
                                  bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (player_id, tracking_id, frame_id, 
              bbox[0], bbox[1], bbox[2], bbox[3], confidence))
        self.conn.commit()
    
    def insert_feature(self, player_id: int, frame_id: int,
                      feature_type: str, feature_data: np.ndarray):
        """Insert feature vector"""
        cursor = self.conn.cursor()
        # Serialize numpy array
        feature_blob = pickle.dumps(feature_data)
        cursor.execute("""
            INSERT INTO features (player_id, frame_id, feature_type, feature_data)
            VALUES (?, ?, ?, ?)
        """, (player_id, frame_id, feature_type, feature_blob))
        self.conn.commit()
    
    def insert_crop(self, player_id: int, tracking_id: int, 
                   frame_id: int, crop_type: str, file_path: str):
        """Insert crop file reference"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO crops (player_id, tracking_id, frame_id, crop_type, file_path)
            VALUES (?, ?, ?, ?, ?)
        """, (player_id, tracking_id, frame_id, crop_type, file_path))
        self.conn.commit()
    
    def insert_reid_event(self, player_id: int, old_tracking_id: int,
                         new_tracking_id: int, frame_id: int, 
                         similarity: float):
        """Log re-identification event"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO reid_events (player_id, old_tracking_id, new_tracking_id,
                                   frame_id, similarity_score)
            VALUES (?, ?, ?, ?, ?)
        """, (player_id, old_tracking_id, new_tracking_id, frame_id, similarity))
        self.conn.commit()
    
    def get_player_history(self, player_id: int) -> Dict:
        """Get complete history for a player"""
        cursor = self.conn.cursor()
        
        # Player info
        cursor.execute("SELECT * FROM players WHERE player_id = ?", (player_id,))
        player_row = cursor.fetchone()
        
        if not player_row:
            return {}
        
        # Detections
        cursor.execute("""
            SELECT frame_id, tracking_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence
            FROM detections WHERE player_id = ?
            ORDER BY frame_id
        """, (player_id,))
        detections = cursor.fetchall()
        
        # Crops
        cursor.execute("""
            SELECT frame_id, crop_type, file_path
            FROM crops WHERE player_id = ?
            ORDER BY frame_id
        """, (player_id,))
        crops = cursor.fetchall()
        
        # Re-ID events
        cursor.execute("""
            SELECT frame_id, old_tracking_id, new_tracking_id, similarity_score
            FROM reid_events WHERE player_id = ?
            ORDER BY frame_id
        """, (player_id,))
        reid_events = cursor.fetchall()
        
        return {
            'player_info': player_row,
            'detections': detections,
            'crops': crops,
            'reid_events': reid_events
        }
    
    def get_frame_data(self, frame_id: int) -> List[Dict]:
        """Get all detections for a specific frame"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT d.player_id, d.tracking_id, d.bbox_x1, d.bbox_y1, 
                   d.bbox_x2, d.bbox_y2, d.confidence, p.jersey_number
            FROM detections d
            JOIN players p ON d.player_id = p.player_id
            WHERE d.frame_id = ?
        """, (frame_id,))
        
        rows = cursor.fetchall()
        return [
            {
                'player_id': r[0],
                'tracking_id': r[1],
                'bbox': [r[2], r[3], r[4], r[5]],
                'confidence': r[6],
                'jersey_number': r[7]
            }
            for r in rows
        ]
    
    def export_to_json(self, output_path: str):
        """Export entire database to JSON"""
        cursor = self.conn.cursor()
        
        # Get all players
        cursor.execute("SELECT * FROM players")
        players = cursor.fetchall()
        
        export_data = {
            'players': [],
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_players': len(players)
            }
        }
        
        for player in players:
            player_id = player[0]
            history = self.get_player_history(player_id)
            export_data['players'].append({
                'player_id': player_id,
                'jersey_number': player[1],
                'total_appearances': player[5],
                'history': {
                    'detections': len(history['detections']),
                    'crops': len(history['crops']),
                    'reid_events': len(history['reid_events'])
                }
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported database to {output_path}")
    
    def close(self):
        """Close database connection"""
        self.conn.close()


class FileSystemManager:
    """Manage file system organization for match data"""
    
    def __init__(self, base_dir: str = "match_output"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        
    def setup_directories(self):
        """Create directory structure"""
        dirs = [
            'faces',
            'bodies',
            'visualizations',
            'meshes',
            'tracking_videos',
            'reports'
        ]
        
        for d in dirs:
            (self.base_dir / d).mkdir(parents=True, exist_ok=True)
    
    def get_face_path(self, player_id: int, frame_id: int) -> str:
        """Get path for face crop"""
        return str(self.base_dir / 'faces' / f"player_{player_id}_frame_{frame_id}.jpg")
    
    def get_body_path(self, player_id: int, frame_id: int) -> str:
        """Get path for body crop"""
        return str(self.base_dir / 'bodies' / f"player_{player_id}_frame_{frame_id}.jpg")
    
    def get_mesh_path(self, player_id: int, frame_id: int) -> str:
        """Get path for mesh data"""
        return str(self.base_dir / 'meshes' / f"player_{player_id}_frame_{frame_id}.pkl")
    
    def save_tracking_video(self, frames: List[np.ndarray], 
                           output_name: str = "tracking_output.mp4",
                           fps: int = 30):
        """Save annotated tracking video"""
        output_path = self.base_dir / 'tracking_videos' / output_name
        
        if not frames:
            return
        
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"Saved tracking video to {output_path}")


class MatchRecorder:
    """High-level interface for recording match data"""
    
    def __init__(self, match_id: int, output_dir: str = "match_output"):
        self.match_id = match_id
        self.db = MatchDatabase(f"{output_dir}/match_{match_id}.db")
        self.fs = FileSystemManager(output_dir)
        
    def record_detection(self, player_id: int, tracking_id: int,
                        frame_id: int, detection_data: Dict):
        """Record a complete detection with all data"""
        # Insert detection
        self.db.insert_detection(
            player_id, tracking_id, frame_id,
            detection_data['bbox'],
            detection_data.get('confidence', 0.0)
        )
        
        # Save and record mesh features
        if 'mesh_feature' in detection_data:
            self.db.insert_feature(
                player_id, frame_id, 'mesh',
                detection_data['mesh_feature']
            )
        
        # Save crops
        if 'face_crop' in detection_data and detection_data['face_crop'] is not None:
            face_path = self.fs.get_face_path(player_id, frame_id)
            cv2.imwrite(face_path, detection_data['face_crop'])
            self.db.insert_crop(player_id, tracking_id, frame_id, 'face', face_path)
        
        if 'body_crop' in detection_data and detection_data['body_crop'] is not None:
            body_path = self.fs.get_body_path(player_id, frame_id)
            cv2.imwrite(body_path, detection_data['body_crop'])
            self.db.insert_crop(player_id, tracking_id, frame_id, 'body', body_path)
    
    def generate_report(self) -> Dict:
        """Generate match statistics report"""
        cursor = self.db.conn.cursor()
        
        # Total players
        cursor.execute("SELECT COUNT(*) FROM players")
        total_players = cursor.fetchone()[0]
        
        # Total detections
        cursor.execute("SELECT COUNT(*) FROM detections")
        total_detections = cursor.fetchone()[0]
        
        # Re-ID events
        cursor.execute("SELECT COUNT(*) FROM reid_events")
        total_reids = cursor.fetchone()[0]
        
        # Player details
        cursor.execute("""
            SELECT player_id, jersey_number, total_appearances
            FROM players
            ORDER BY total_appearances DESC
        """)
        players = cursor.fetchall()
        
        report = {
            'match_id': self.match_id,
            'total_players': total_players,
            'total_detections': total_detections,
            'total_reid_events': total_reids,
            'players': [
                {
                    'player_id': p[0],
                    'jersey': p[1],
                    'appearances': p[2]
                }
                for p in players
            ]
        }
        
        # Save report
        report_path = self.fs.base_dir / 'reports' / f'match_{self.match_id}_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def close(self):
        """Close all resources"""
        self.db.close()


# Example usage
if __name__ == "__main__":
    # Initialize recorder
    recorder = MatchRecorder(match_id=1, output_dir="match_output")
    
    # Record some detections
    for frame_id in range(10):
        detection_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.9,
            'mesh_feature': np.random.randn(128),
            'body_crop': np.zeros((200, 100, 3), dtype=np.uint8)
        }
        
        recorder.record_detection(
            player_id=1,
            tracking_id=1,
            frame_id=frame_id,
            detection_data=detection_data
        )
    
    # Generate report
    report = recorder.generate_report()
    print(json.dumps(report, indent=2))
    
    # Close
    recorder.close()