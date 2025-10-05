"""
Module 5 & 6: Identity Loss Handling and Re-Assignment
Combined mesh similarity and jersey number matching for re-identification
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import cv2
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PlayerIdentity:
    """Represents a player's identity with all associated data"""
    player_id: int
    jersey_number: str
    jersey_confidence: float
    mesh_features: List[np.ndarray]
    body_proportions: List[Dict]
    face_embeddings: List[np.ndarray]
    first_seen: int
    last_seen: int
    tracking_ids: List[int]  # All tracking IDs assigned to this player
    total_appearances: int
    status: str  # 'active', 'lost', 'inactive'


class PlayerDatabase:
    """Database for storing and managing player identities"""
    
    def __init__(self):
        self.players: Dict[int, PlayerIdentity] = {}
        self.next_player_id = 1
        self.tracking_to_player = {}  # Maps tracking_id -> player_id
        
    def create_player(self, tracking_id: int, jersey_number: str,
                     jersey_conf: float, mesh_feature: np.ndarray,
                     body_props: Dict, frame_id: int) -> int:
        """Create new player identity"""
        player_id = self.next_player_id
        self.next_player_id += 1
        
        self.players[player_id] = PlayerIdentity(
            player_id=player_id,
            jersey_number=jersey_number,
            jersey_confidence=jersey_conf,
            mesh_features=[mesh_feature],
            body_proportions=[body_props],
            face_embeddings=[],
            first_seen=frame_id,
            last_seen=frame_id,
            tracking_ids=[tracking_id],
            total_appearances=1,
            status='active'
        )
        
        self.tracking_to_player[tracking_id] = player_id
        return player_id
    
    def update_player(self, player_id: int, tracking_id: int,
                     mesh_feature: np.ndarray, body_props: Dict,
                     frame_id: int):
        """Update existing player with new observation"""
        if player_id not in self.players:
            return
        
        player = self.players[player_id]
        player.mesh_features.append(mesh_feature)
        player.body_proportions.append(body_props)
        player.last_seen = frame_id
        player.total_appearances += 1
        player.status = 'active'
        
        if tracking_id not in player.tracking_ids:
            player.tracking_ids.append(tracking_id)
            self.tracking_to_player[tracking_id] = player_id
    
    def get_player_by_tracking_id(self, tracking_id: int) -> Optional[PlayerIdentity]:
        """Get player identity from tracking ID"""
        player_id = self.tracking_to_player.get(tracking_id)
        if player_id:
            return self.players.get(player_id)
        return None
    
    def get_active_players(self, current_frame: int, 
                          max_absence: int = 100) -> List[PlayerIdentity]:
        """Get players seen recently"""
        active = []
        for player in self.players.values():
            if current_frame - player.last_seen <= max_absence:
                active.append(player)
        return active
    
    def mark_player_lost(self, player_id: int):
        """Mark player as lost (tracking interrupted)"""
        if player_id in self.players:
            self.players[player_id].status = 'lost'


class ReIdentificationSystem:
    """
    Re-identification system combining multiple cues:
    - 3D mesh/pose similarity
    - Jersey number matching
    - Body proportions
    - Face embeddings (optional)
    """
    
    def __init__(self, 
                 mesh_weight: float = 0.4,
                 jersey_weight: float = 0.4,
                 proportion_weight: float = 0.2,
                 reid_threshold: float = 0.7):
        """
        Initialize re-ID system
        Args:
            mesh_weight: Weight for mesh similarity
            jersey_weight: Weight for jersey number matching
            proportion_weight: Weight for body proportions
            reid_threshold: Minimum similarity for re-identification
        """
        self.mesh_weight = mesh_weight
        self.jersey_weight = jersey_weight
        self.proportion_weight = proportion_weight
        self.reid_threshold = reid_threshold
        
        self.database = PlayerDatabase()
        
    def compute_mesh_similarity(self, feat1: np.ndarray, 
                                feat2_list: List[np.ndarray]) -> float:
        """
        Compare mesh feature against history
        Args:
            feat1: Query feature
            feat2_list: List of reference features
        Returns:
            Maximum similarity score
        """
        if not feat2_list:
            return 0.0
        
        similarities = []
        for feat2 in feat2_list[-5:]:  # Use last 5 observations
            # Cosine similarity
            min_len = min(len(feat1), len(feat2))
            f1 = feat1[:min_len]
            f2 = feat2[:min_len]
            
            dot = np.dot(f1, f2)
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)
            
            if norm1 > 0 and norm2 > 0:
                sim = dot / (norm1 * norm2)
                sim = (sim + 1) / 2  # Normalize to [0, 1]
                similarities.append(sim)
        
        return max(similarities) if similarities else 0.0
    
    def compute_jersey_similarity(self, jersey1: str, jersey2: str,
                                 conf1: float, conf2: float) -> float:
        """
        Compare jersey numbers
        Args:
            jersey1, jersey2: Jersey numbers as strings
            conf1, conf2: OCR confidence scores
        Returns:
            Similarity score [0, 1]
        """
        if not jersey1 or not jersey2:
            return 0.5  # Unknown
        
        # Exact match
        if jersey1 == jersey2:
            # Weight by confidence
            avg_conf = (conf1 + conf2) / 2
            return avg_conf
        
        # Partial match (e.g., "10" vs "1" could be OCR error)
        if jersey1 in jersey2 or jersey2 in jersey1:
            return 0.3
        
        return 0.0
    
    def compute_proportion_similarity(self, props1: Dict,
                                     props2_list: List[Dict]) -> float:
        """
        Compare body proportions against history
        Args:
            props1: Query proportions
            props2_list: List of reference proportions
        Returns:
            Average similarity score
        """
        if not props2_list:
            return 0.5
        
        keys = ['shoulder_to_torso_ratio', 'hip_to_torso_ratio']
        similarities = []
        
        for props2 in props2_list[-5:]:
            diffs = []
            for key in keys:
                if key in props1 and key in props2:
                    diff = abs(props1[key] - props2[key])
                    diffs.append(diff)
            
            if diffs:
                # Convert difference to similarity
                avg_diff = np.mean(diffs)
                sim = np.exp(-avg_diff * 5)
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def compute_combined_similarity(self, query: Dict,
                                   reference: PlayerIdentity) -> float:
        """
        Compute weighted combined similarity
        Args:
            query: Dict with 'mesh_feature', 'jersey_number', 'body_proportions'
            reference: PlayerIdentity to compare against
        Returns:
            Combined similarity score [0, 1]
        """
        # Mesh similarity
        mesh_sim = self.compute_mesh_similarity(
            query['mesh_feature'],
            reference.mesh_features
        )
        
        # Jersey similarity
        jersey_sim = self.compute_jersey_similarity(
            query['jersey_number'],
            reference.jersey_number,
            query['jersey_confidence'],
            reference.jersey_confidence
        )
        
        # Proportion similarity
        prop_sim = self.compute_proportion_similarity(
            query['body_proportions'],
            reference.body_proportions
        )
        
        # Weighted combination
        combined = (
            self.mesh_weight * mesh_sim +
            self.jersey_weight * jersey_sim +
            self.proportion_weight * prop_sim
        )
        
        return combined
    
    def find_best_match(self, query: Dict, 
                       candidates: List[PlayerIdentity]) -> Tuple[Optional[int], float]:
        """
        Find best matching player from candidates
        Args:
            query: Query features
            candidates: List of candidate players
        Returns:
            (player_id, similarity_score) or (None, 0.0)
        """
        if not candidates:
            return None, 0.0
        
        best_id = None
        best_score = 0.0
        
        for candidate in candidates:
            score = self.compute_combined_similarity(query, candidate)
            
            if score > best_score:
                best_score = score
                best_id = candidate.player_id
        
        return best_id, best_score
    
    def process_detection(self, tracking_id: int, detection_data: Dict,
                         frame_id: int) -> Tuple[int, bool]:
        """
        Process a detection and assign/reassign player ID
        Args:
            tracking_id: Current tracking ID from tracker
            detection_data: Dict with detection features
            frame_id: Current frame number
        Returns:
            (player_id, is_new_assignment)
        """
        # Check if tracking ID already has player ID
        existing_player = self.database.get_player_by_tracking_id(tracking_id)
        
        if existing_player and existing_player.status == 'active':
            # Update existing player
            self.database.update_player(
                existing_player.player_id,
                tracking_id,
                detection_data['mesh_feature'],
                detection_data['body_proportions'],
                frame_id
            )
            return existing_player.player_id, False
        
        # New or lost tracking ID - attempt re-identification
        query = {
            'mesh_feature': detection_data['mesh_feature'],
            'jersey_number': detection_data['jersey_number'],
            'jersey_confidence': detection_data['jersey_confidence'],
            'body_proportions': detection_data['body_proportions']
        }
        
        # Get candidate players (recently seen)
        candidates = self.database.get_active_players(frame_id, max_absence=100)
        
        # Find best match
        best_player_id, similarity = self.find_best_match(query, candidates)
        
        if best_player_id and similarity >= self.reid_threshold:
            # Re-assign to existing player
            self.database.update_player(
                best_player_id,
                tracking_id,
                detection_data['mesh_feature'],
                detection_data['body_proportions'],
                frame_id
            )
            print(f"Re-identified player {best_player_id} (similarity: {similarity:.3f})")
            return best_player_id, True
        
        # Create new player
        player_id = self.database.create_player(
            tracking_id,
            detection_data['jersey_number'],
            detection_data['jersey_confidence'],
            detection_data['mesh_feature'],
            detection_data['body_proportions'],
            frame_id
        )
        print(f"Created new player {player_id}")
        return player_id, True
    
    def handle_tracking_interruption(self, lost_tracking_id: int):
        """
        Handle when a tracking ID is lost
        Args:
            lost_tracking_id: Tracking ID that was lost
        """
        player = self.database.get_player_by_tracking_id(lost_tracking_id)
        if player:
            self.database.mark_player_lost(player.player_id)
            print(f"Player {player.player_id} lost (tracking_id: {lost_tracking_id})")
    
    def get_player_summary(self, player_id: int) -> Optional[Dict]:
        """Get summary statistics for a player"""
        if player_id not in self.database.players:
            return None
        
        player = self.database.players[player_id]
        
        return {
            'player_id': player.player_id,
            'jersey_number': player.jersey_number,
            'first_seen': player.first_seen,
            'last_seen': player.last_seen,
            'total_appearances': player.total_appearances,
            'tracking_ids_used': player.tracking_ids,
            'status': player.status,
            'avg_body_proportions': self._average_proportions(player.body_proportions)
        }
    
    @staticmethod
    def _average_proportions(props_list: List[Dict]) -> Dict:
        """Compute average body proportions"""
        if not props_list:
            return {}
        
        keys = props_list[0].keys()
        avg_props = {}
        
        for key in keys:
            values = [p[key] for p in props_list if key in p]
            if values:
                avg_props[key] = np.mean(values)
        
        return avg_props


class InterruptionDetector:
    """Detect and log tracking interruptions"""
    
    def __init__(self):
        self.previous_tracks = set()
        self.interruptions = []
        
    def detect_interruptions(self, current_tracks: List[int], 
                            frame_id: int) -> List[Tuple[str, int]]:
        """
        Detect new and lost tracks
        Args:
            current_tracks: List of current tracking IDs
            frame_id: Current frame number
        Returns:
            List of (event_type, tracking_id) tuples
        """
        current_set = set(current_tracks)
        events = []
        
        # Lost tracks
        lost = self.previous_tracks - current_set
        for track_id in lost:
            events.append(('lost', track_id))
            self.interruptions.append({
                'type': 'lost',
                'tracking_id': track_id,
                'frame_id': frame_id
            })
        
        # New tracks
        new = current_set - self.previous_tracks
        for track_id in new:
            events.append(('new', track_id))
            self.interruptions.append({
                'type': 'new',
                'tracking_id': track_id,
                'frame_id': frame_id
            })
        
        self.previous_tracks = current_set
        return events
    
    def get_interruption_stats(self) -> Dict:
        """Get statistics about interruptions"""
        return {
            'total_interruptions': len(self.interruptions),
            'lost_tracks': sum(1 for i in self.interruptions if i['type'] == 'lost'),
            'new_tracks': sum(1 for i in self.interruptions if i['type'] == 'new'),
            'interruptions': self.interruptions
        }


# Example usage
if __name__ == "__main__":
    # Initialize re-ID system
    reid_system = ReIdentificationSystem(
        mesh_weight=0.4,
        jersey_weight=0.4,
        proportion_weight=0.2,
        reid_threshold=0.7
    )
    
    interruption_detector = InterruptionDetector()
    
    # Simulate processing frames
    for frame_id in range(100):
        # Get tracked objects (from Module 4)
        tracked_objects = [
            {
                'tracking_id': 1,
                'bbox': [100, 100, 200, 300],
                'mesh_feature': np.random.randn(128),
                'jersey_number': '10',
                'jersey_confidence': 0.9,
                'body_proportions': {
                    'shoulder_to_torso_ratio': 0.45,
                    'hip_to_torso_ratio': 0.38
                }
            },
            {
                'tracking_id': 2,
                'bbox': [300, 150, 400, 350],
                'mesh_feature': np.random.randn(128),
                'jersey_number': '7',
                'jersey_confidence': 0.85,
                'body_proportions': {
                    'shoulder_to_torso_ratio': 0.42,
                    'hip_to_torso_ratio': 0.35
                }
            }
        ]
        
        # Detect interruptions
        current_track_ids = [obj['tracking_id'] for obj in tracked_objects]
        events = interruption_detector.detect_interruptions(current_track_ids, frame_id)
        
        # Handle interruptions
        for event_type, track_id in events:
            if event_type == 'lost':
                reid_system.handle_tracking_interruption(track_id)
        
        # Process each detection
        for obj in tracked_objects:
            player_id, is_reassigned = reid_system.process_detection(
                obj['tracking_id'],
                obj,
                frame_id
            )
            obj['player_id'] = player_id
            
            if is_reassigned:
                print(f"Frame {frame_id}: Tracking ID {obj['tracking_id']} -> Player {player_id}")
    
    # Print statistics
    print("\n=== Interruption Statistics ===")
    stats = interruption_detector.get_interruption_stats()
    print(f"Total interruptions: {stats['total_interruptions']}")
    print(f"Lost tracks: {stats['lost_tracks']}")
    print(f"New tracks: {stats['new_tracks']}")
    
    print("\n=== Player Summaries ===")
    for player_id in reid_system.database.players:
        summary = reid_system.get_player_summary(player_id)
        print(f"\nPlayer {player_id}:")
        print(f"  Jersey: {summary['jersey_number']}")
        print(f"  Appearances: {summary['total_appearances']}")
        print(f"  Tracking IDs: {summary['tracking_ids_used']}")
        print(f"  Status: {summary['status']}")