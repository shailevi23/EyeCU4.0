"""
Utility functions for video processing
"""

import cv2
import numpy as np
import os
from pathlib import Path

def read_video(video_path, max_frames=None, start_frame=0):
    """
    Read video frames into memory
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to read (None for all)
        start_frame: Starting frame index
    
    Returns:
        List of video frames
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Reading video: {video_path}")
    print(f"  - Dimensions: {width}x{height}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Starting from frame: {start_frame}")
    
    if max_frames:
        print(f"  - Reading max {max_frames} frames")
    
    # Skip to start_frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read frames
    frames = []
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        count += 1
        
        if count % 50 == 0:
            print(f"  - Read {count} frames...")
        
        if max_frames and count >= max_frames:
            break
    
    cap.release()
    print(f"Finished reading {len(frames)} frames")
    
    return frames

def save_video(frames, output_path, fps=30, codec='avc1'):
    """
    Save frames as video
    
    Args:
        frames: List of frames
        output_path: Output video path
        fps: Frames per second
        codec: Four character code for codec
    """
    if not frames:
        print("No frames to save")
        return False
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get frame dimensions
    h, w = frames[0].shape[:2]
    
    # Initialize video writer with better quality
    try:
        # First try with better codec
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # If failed, fall back to mp4v
        if not out.isOpened():
            print(f"Warning: Could not use {codec} codec, falling back to mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    except:
        # Fallback
        print("Warning: Error with codec, falling back to default...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file: {output_path}")
        return False
    
    # Write frames
    for i, frame in enumerate(frames):
        out.write(frame)
        
        if i % 50 == 0:
            print(f"  - Written {i} frames...")
    
    out.release()
    print(f"Saved {len(frames)} frames to {output_path}")
    
    return True

def get_video_info(video_path):
    """
    Get information about a video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        'path': video_path,
        'filename': os.path.basename(video_path),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration_sec': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    }
    
    cap.release()
    return info