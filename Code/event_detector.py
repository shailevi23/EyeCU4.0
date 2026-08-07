"""
events/event_detector.py
Detection of key football events (goals, shots, passes, etc.)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import cv2

@dataclass
class Event:
    """Represents a detected event"""
    event_type: str
    frame_num: int
    timestamp: float
    player_id: Optional[str]
    team_id: int
    position: Tuple[float, float]
    confidence: float
    metadata: dict

    def to_dict(self):
        return {
            'event_type': self.event_type,
            'frame_num': self.frame_num,
            'timestamp': self.timestamp,
            'player_id': self.player_id,
            'team_id': self.team_id,
            'position': self.position,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class EventDetector:
    """Detect key events in football matches"""
    
    def __init__(self, fps: int = 30, field_width: float = 105.0, field_height: float = 68.0):
        self.fps = fps
        self.field_width = field_width
        self.field_height = field_height
        
        # Goal zones (in meters from center)
        self.goal_zone_width = 7.32
        self.goal_zone_depth = 5.0
        
        # Thresholds
        self.ball_proximity_threshold = 2.0  # meters
        self.shot_speed_threshold = 15.0  # m/s
        self.pass_min_distance = 5.0  # meters
        self.sprint_speed = 5.5  # m/s
        
        # Event history
        self.events: List[Event] = []
        
        # Ball tracking history
        self.ball_history = deque(maxlen=60)  # 2 seconds at 30fps
        
        # Player tracking history
        self.player_history = {}
        
        # Last possession
        self.last_possession = {'player_id': None, 'frame': 0}
        
    def update_ball_position(self, ball_pos: Tuple[float, float], frame_num: int):
        """Update ball position history"""
        self.ball_history.append({
            'position': ball_pos,
            'frame': frame_num,
            'timestamp': frame_num / self.fps
        })
    
    def update_player_position(self, player_id: str, position: Tuple[float, float], 
                              team_id: int, frame_num: int):
        """Update player position history"""
        if player_id not in self.player_history:
            self.player_history[player_id] = deque(maxlen=30)
        
        self.player_history[player_id].append({
            'position': position,
            'team_id': team_id,
            'frame': frame_num,
            'timestamp': frame_num / self.fps
        })
    
    def calculate_distance(self, pos1: Tuple[float, float], 
                          pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def calculate_speed(self, positions: List[Dict]) -> float:
        """Calculate average speed from position history"""
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(positions)):
            d = self.calculate_distance(
                positions[i]['position'], 
                positions[i-1]['position']
            )
            total_distance += d
        
        time_elapsed = (positions[-1]['timestamp'] - positions[0]['timestamp'])
        if time_elapsed > 0:
            return total_distance / time_elapsed
        return 0.0
    
    def is_in_goal_zone(self, position: Tuple[float, float]) -> bool:
        """Check if position is in goal zone"""
        x, y = position
        
        # Check both goals (assuming goals at x extremes)
        left_goal = abs(x + self.field_width/2) < self.goal_zone_depth and \
                    abs(y) < self.goal_zone_width/2
        right_goal = abs(x - self.field_width/2) < self.goal_zone_depth and \
                     abs(y) < self.goal_zone_width/2
        
        return left_goal or right_goal
    
    def detect_goal(self, frame_num: int) -> Optional[Event]:
        """Detect goal events"""
        if len(self.ball_history) < 10:
            return None
        
        recent_balls = list(self.ball_history)[-10:]
        
        # Check if ball crossed goal line
        for i in range(1, len(recent_balls)):
            prev_pos = recent_balls[i-1]['position']
            curr_pos = recent_balls[i]['position']
            
            # Check if ball entered goal zone
            if not self.is_in_goal_zone(prev_pos) and self.is_in_goal_zone(curr_pos):
                # Find closest player (likely scorer)
                scorer_id = self.find_closest_player(prev_pos, recent_balls[i-1]['frame'])
                
                if scorer_id:
                    player_info = self.player_history.get(scorer_id)
                    if player_info:
                        team_id = list(player_info)[-1]['team_id']
                        
                        event = Event(
                            event_type='goal',
                            frame_num=frame_num,
                            timestamp=frame_num / self.fps,
                            player_id=scorer_id,
                            team_id=team_id,
                            position=curr_pos,
                            confidence=0.9,
                            metadata={'ball_speed': self.calculate_ball_speed()}
                        )
                        self.events.append(event)
                        return event
        
        return None
    
    def detect_shot(self, frame_num: int) -> Optional[Event]:
        """Detect shot attempts"""
        if len(self.ball_history) < 5:
            return None
        
        ball_speed = self.calculate_ball_speed()
        
        # Shot is high-speed ball movement toward goal
        if ball_speed > self.shot_speed_threshold:
            recent_ball = list(self.ball_history)[-1]
            ball_pos = recent_ball['position']
            
            # Check if moving toward goal
            if self.is_moving_toward_goal(list(self.ball_history)[-5:]):
                shooter_id = self.find_closest_player(ball_pos, frame_num)
                
                if shooter_id:
                    player_info = self.player_history.get(shooter_id)
                    if player_info:
                        team_id = list(player_info)[-1]['team_id']
                        
                        event = Event(
                            event_type='shot',
                            frame_num=frame_num,
                            timestamp=frame_num / self.fps,
                            player_id=shooter_id,
                            team_id=team_id,
                            position=ball_pos,
                            confidence=0.75,
                            metadata={'ball_speed': ball_speed}
                        )
                        self.events.append(event)
                        return event
        
        return None
    
    def detect_pass(self, frame_num: int) -> Optional[Event]:
        """Detect pass events"""
        if len(self.ball_history) < 20:
            return None
        
        # Detect change in ball direction and possession
        current_closest = self.find_closest_player(
            list(self.ball_history)[-1]['position'], 
            frame_num
        )
        
        if current_closest and current_closest != self.last_possession['player_id']:
            # Calculate pass distance
            if self.last_possession['player_id']:
                last_pos = self.get_player_position(
                    self.last_possession['player_id'],
                    self.last_possession['frame']
                )
                curr_pos = self.get_player_position(current_closest, frame_num)
                
                if last_pos and curr_pos:
                    pass_distance = self.calculate_distance(last_pos, curr_pos)
                    
                    # Check if same team (pass) or different team (interception)
                    last_team = self.get_player_team(self.last_possession['player_id'])
                    curr_team = self.get_player_team(current_closest)
                    
                    if pass_distance > self.pass_min_distance:
                        if last_team == curr_team:
                            event_type = 'pass'
                            passer_id = self.last_possession['player_id']
                        else:
                            event_type = 'interception'
                            passer_id = current_closest
                        
                        event = Event(
                            event_type=event_type,
                            frame_num=frame_num,
                            timestamp=frame_num / self.fps,
                            player_id=passer_id,
                            team_id=last_team,
                            position=list(self.ball_history)[-1]['position'],
                            confidence=0.65,
                            metadata={
                                'receiver_id': current_closest if event_type == 'pass' else None,
                                'distance': pass_distance
                            }
                        )
                        self.events.append(event)
                        
                        # Update possession
                        self.last_possession = {'player_id': current_closest, 'frame': frame_num}
                        return event
            
            # Update possession
            self.last_possession = {'player_id': current_closest, 'frame': frame_num}
        
        return None
    
    def detect_sprint(self, player_id: str, frame_num: int) -> Optional[Event]:
        """Detect sprint events"""
        if player_id not in self.player_history:
            return None
        
        positions = list(self.player_history[player_id])
        if len(positions) < 10:
            return None
        
        speed = self.calculate_speed(positions[-10:])
        
        if speed > self.sprint_speed:
            team_id = positions[-1]['team_id']
            
            event = Event(
                event_type='sprint',
                frame_num=frame_num,
                timestamp=frame_num / self.fps,
                player_id=player_id,
                team_id=team_id,
                position=positions[-1]['position'],
                confidence=0.8,
                metadata={'speed': speed}
            )
            # Don't add to events (too many sprints)
            return event
        
        return None
    
    def find_closest_player(self, ball_pos: Tuple[float, float], 
                           frame_num: int) -> Optional[str]:
        """Find player closest to ball"""
        min_dist = float('inf')
        closest_player = None
        
        for player_id, history in self.player_history.items():
            if not history:
                continue
            
            player_pos = list(history)[-1]['position']
            dist = self.calculate_distance(ball_pos, player_pos)
            
            if dist < min_dist:
                min_dist = dist
                closest_player = player_id
        
        return closest_player if min_dist < self.ball_proximity_threshold else None
    
    def calculate_ball_speed(self) -> float:
        """Calculate current ball speed"""
        if len(self.ball_history) < 2:
            return 0.0
        
        recent = list(self.ball_history)[-5:]
        return self.calculate_speed(recent)
    
    def is_moving_toward_goal(self, ball_positions: List[Dict]) -> bool:
        """Check if ball is moving toward either goal"""
        if len(ball_positions) < 2:
            return False
        
        # Get direction vector
        start_pos = ball_positions[0]['position']
        end_pos = ball_positions[-1]['position']
        
        dx = end_pos[0] - start_pos[0]
        
        # Moving toward left or right goal
        return abs(dx) > abs(end_pos[1] - start_pos[1])
    
    def get_player_position(self, player_id: str, frame_num: int) -> Optional[Tuple[float, float]]:
        """Get player position at specific frame"""
        if player_id not in self.player_history:
            return None
        
        history = list(self.player_history[player_id])
        if not history:
            return None
        
        # Find closest frame
        closest = min(history, key=lambda x: abs(x['frame'] - frame_num))
        return closest['position']
    
    def get_player_team(self, player_id: str) -> Optional[int]:
        """Get player's team"""
        if player_id not in self.player_history:
            return None
        
        history = list(self.player_history[player_id])
        if not history:
            return None
        
        return history[-1]['team_id']
    
    def process_frame(self, frame_num: int, ball_pos: Optional[Tuple[float, float]],
                     player_detections: List[Dict]) -> List[Event]:
        """
        Process a frame and detect all events
        
        Args:
            frame_num: Current frame number
            ball_pos: Ball position (x, y) in meters
            player_detections: List of player detections with keys:
                              'player_id', 'position', 'team_id'
        
        Returns:
            List of detected events
        """
        frame_events = []
        
        # Update ball position
        if ball_pos:
            self.update_ball_position(ball_pos, frame_num)
        
        # Update player positions
        for detection in player_detections:
            self.update_player_position(
                detection['player_id'],
                detection['position'],
                detection['team_id'],
                frame_num
            )
        
        # Detect events
        goal = self.detect_goal(frame_num)
        if goal:
            frame_events.append(goal)
        
        shot = self.detect_shot(frame_num)
        if shot:
            frame_events.append(shot)
        
        pass_event = self.detect_pass(frame_num)
        if pass_event:
            frame_events.append(pass_event)
        
        return frame_events
    
    def get_all_events(self, event_type: Optional[str] = None) -> List[Event]:
        """Get all detected events, optionally filtered by type"""
        if event_type:
            return [e for e in self.events if e.event_type == event_type]
        return self.events
    
    def get_event_summary(self) -> Dict:
        """Get summary of detected events"""
        event_types = {}
        for event in self.events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        return {
            'total_events': len(self.events),
            'event_types': event_types,
            'goals': event_types.get('goal', 0),
            'shots': event_types.get('shot', 0),
            'passes': event_types.get('pass', 0),
            'interceptions': event_types.get('interception', 0)
        }