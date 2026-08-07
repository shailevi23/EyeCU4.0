"""
events/highlight_generator.py
Generate highlight clips from detected events
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

# Change from relative import to absolute import
from event_detector import Event

class HighlightGenerator:
    """Generate highlight video clips from events"""
    
    def __init__(self, pre_buffer: float = 3.0, post_buffer: float = 5.0,
                 min_duration: float = 5.0, max_duration: float = 15.0):
        """
        Args:
            pre_buffer: Seconds before event to include
            post_buffer: Seconds after event to include
            min_duration: Minimum highlight duration
            max_duration: Maximum highlight duration
        """
        self.pre_buffer = pre_buffer
        self.post_buffer = post_buffer
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        # Priority for different event types
        self.event_priorities = {
            'goal': 10,
            'shot': 7,
            'pass': 5,
            'interception': 6,
            'sprint': 3
        }
    
    def calculate_clip_bounds(self, event: Event, fps: int, 
                             total_frames: int) -> Tuple[int, int]:
        """Calculate start and end frames for a highlight clip"""
        # Calculate frame offsets
        pre_frames = int(self.pre_buffer * fps)
        post_frames = int(self.post_buffer * fps)
        
        start_frame = max(0, event.frame_num - pre_frames)
        end_frame = min(total_frames, event.frame_num + post_frames)
        
        # Ensure minimum duration
        min_frames = int(self.min_duration * fps)
        if end_frame - start_frame < min_frames:
            # Extend in both directions
            needed = min_frames - (end_frame - start_frame)
            start_frame = max(0, start_frame - needed // 2)
            end_frame = min(total_frames, end_frame + needed // 2)
        
        # Ensure maximum duration
        max_frames = int(self.max_duration * fps)
        if end_frame - start_frame > max_frames:
            # Keep event centered
            middle = event.frame_num
            start_frame = max(0, middle - max_frames // 2)
            end_frame = min(total_frames, start_frame + max_frames)
        
        return start_frame, end_frame
    
    def merge_overlapping_clips(self, clips: List[Dict]) -> List[Dict]:
        """Merge overlapping highlight clips"""
        if not clips:
            return []
        
        # Sort by start frame
        sorted_clips = sorted(clips, key=lambda x: x['start_frame'])
        
        merged = [sorted_clips[0]]
        
        for current in sorted_clips[1:]:
            last = merged[-1]
            
            # Check if overlapping
            if current['start_frame'] <= last['end_frame']:
                # Merge clips
                last['end_frame'] = max(last['end_frame'], current['end_frame'])
                last['events'].extend(current['events'])
                # Use highest priority
                last['priority'] = max(last['priority'], current['priority'])
            else:
                merged.append(current)
        
        return merged
    
    def create_highlight_plan(self, events: List[Event], fps: int,
                            total_frames: int) -> List[Dict]:
        """Create a plan for all highlight clips"""
        clips = []
        
        for event in events:
            start_frame, end_frame = self.calculate_clip_bounds(event, fps, total_frames)
            
            clip = {
                'start_frame': start_frame,
                'end_frame': end_frame,
                'events': [event],
                'priority': self.event_priorities.get(event.event_type, 5),
                'duration': (end_frame - start_frame) / fps
            }
            clips.append(clip)
        
        # Merge overlapping clips
        merged_clips = self.merge_overlapping_clips(clips)
        
        # Sort by priority
        merged_clips.sort(key=lambda x: x['priority'], reverse=True)
        
        return merged_clips
    
    def extract_clip(self, video_path: str, start_frame: int, end_frame: int,
                    output_path: str, add_overlay: bool = True,
                    events: List[Event] = None) -> bool:
        """
        Extract a single highlight clip from video
        
        Args:
            video_path: Path to source video
            start_frame: Start frame number
            end_frame: End frame number
            output_path: Path to save clip
            add_overlay: Whether to add event overlay
            events: Events occurring in this clip
        
        Returns:
            True if successful
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_num = start_frame
        while frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            if add_overlay and events:
                frame = self.add_event_overlay(frame, frame_num, events, fps)
            
            out.write(frame)
            frame_num += 1
        
        cap.release()
        out.release()
        
        return True
    
    def add_event_overlay(self, frame: np.ndarray, frame_num: int,
                         events: List[Event], fps: int) -> np.ndarray:
        """Add event information overlay to frame"""
        overlay = frame.copy()
        
        # Check if any events occur at this frame
        for event in events:
            # Show event label for a few seconds around the event
            time_diff = abs(frame_num - event.frame_num) / fps
            
            if time_diff < 3.0:  # Show for 3 seconds
                # Draw event label
                label = event.event_type.upper()
                if event.player_id:
                    label += f" - {event.player_id}"
                
                # Position at top center
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_BOLD, 1.5, 3)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = 80
                
                # Draw background
                padding = 20
                cv2.rectangle(overlay,
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(overlay, label,
                          (text_x, text_y),
                          cv2.FONT_HERSHEY_BOLD, 1.5,
                          (0, 255, 255), 3)
                
                # Add timestamp
                timestamp = f"{event.timestamp:.1f}s"
                cv2.putText(overlay, timestamp,
                          (text_x, text_y + 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                          (255, 255, 255), 2)
        
        # Blend overlay with frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def generate_highlights(self, video_path: str, events: List[Event],
                          output_dir: str, fps: int, total_frames: int) -> List[str]:
        """
        Generate all highlight clips
        
        Args:
            video_path: Path to source video
            events: List of detected events
            output_dir: Directory to save highlights
            fps: Video FPS
            total_frames: Total number of frames
        
        Returns:
            List of paths to generated highlight clips
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create highlight plan
        clips = self.create_highlight_plan(events, fps, total_frames)
        
        generated_clips = []
        
        for idx, clip in enumerate(clips):
            # Create filename based on primary event
            primary_event = max(clip['events'], key=lambda e: self.event_priorities.get(e.event_type, 0))
            filename = f"highlight_{idx:03d}_{primary_event.event_type}_frame{clip['start_frame']}.mp4"
            output_file = str(output_path / filename)
            
            # Extract clip
            success = self.extract_clip(
                video_path,
                clip['start_frame'],
                clip['end_frame'],
                output_file,
                add_overlay=True,
                events=clip['events']
            )
            
            if success:
                generated_clips.append(output_file)
                
                # Save metadata
                metadata_file = output_file.replace('.mp4', '_metadata.json')
                metadata = {
                    'start_frame': clip['start_frame'],
                    'end_frame': clip['end_frame'],
                    'duration': clip['duration'],
                    'priority': clip['priority'],
                    'events': [e.to_dict() for e in clip['events']],
                    'output_file': output_file
                }
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        return generated_clips
    
    def create_compilation(self, highlight_clips: List[str], output_path: str,
                          max_clips: int = 10) -> bool:
        """
        Create a single compilation video from multiple highlights
        
        Args:
            highlight_clips: List of highlight clip paths
            output_path: Path to save compilation
            max_clips: Maximum number of clips to include
        
        Returns:
            True if successful
        """
        if not highlight_clips:
            return False
        
        # Limit number of clips
        clips_to_use = highlight_clips[:max_clips]
        
        # Get properties from first clip
        cap = cv2.VideoCapture(clips_to_use[0])
        if not cap.isOpened():
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create output writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Add transition frames
        transition_frames = int(fps * 0.5)  # 0.5 second transition
        
        for idx, clip_path in enumerate(clips_to_use):
            cap = cv2.VideoCapture(clip_path)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            cap.release()
            
            # Add transition (black frames) between clips
            if idx < len(clips_to_use) - 1:
                black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                for _ in range(transition_frames):
                    out.write(black_frame)
        
        out.release()
        return True
    
    def export_highlights_report(self, clips: List[Dict], output_path: str):
        """Export highlights report as JSON"""
        report = {
            'total_highlights': len(clips),
            'total_duration': sum(c['duration'] for c in clips),
            'highlights': []
        }
        
        for idx, clip in enumerate(clips):
            highlight_info = {
                'id': idx,
                'start_frame': clip['start_frame'],
                'end_frame': clip['end_frame'],
                'duration': clip['duration'],
                'priority': clip['priority'],
                'events': [e.to_dict() for e in clip['events']],
                'event_types': list(set(e.event_type for e in clip['events']))
            }
            report['highlights'].append(highlight_info)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report