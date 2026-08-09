#!/usr/bin/env python
"""
Football Analysis Pipeline Runner Script
This script provides a command-line interface to run the full Football Analysis Pipeline
"""

import argparse
import os
import sys

# Windows consoles default to cp1252, which cannot encode the emoji below.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from full_pipeline import FootballAnalysisPipeline

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Football Analysis Pipeline")
    
    # Required arguments
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default="tracked_output.mp4", help="Output video filename")
    
    # Optional arguments
    parser.add_argument("--output-dir", default="match_analysis_output",
                        help="Output directory for all results")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    parser.add_argument("--skip-frames", type=int, default=2, 
                        help="Process every N frames (higher values = faster processing)")
    parser.add_argument("--display", action="store_true", 
                        help="Display real-time visualization")
    parser.add_argument("--use-cache", action="store_true", 
                        help="Use cached detections/tracks if available")
    
    parser.add_argument("--use-roboflow", action="store_true",
                        help="Opt in to the Roboflow cloud detector. Off by default: "
                             "the local YOLO model is the production path and Roboflow "
                             "is kept only as a labelling/benchmark tool. Requires "
                             "ROBOFLOW_API_KEY.")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"),
                        help="Roboflow API key (defaults to the ROBOFLOW_API_KEY "
                             "environment variable; without it Roboflow is disabled "
                             "and the local YOLO model is used)")
    parser.add_argument("--yolo-model", default="yolov8x.pt",
                        help="YOLO model path (yolov8n.pt, yolov8s.pt, yolov8m.pt, "
                             "yolov8l.pt, yolov8x.pt, or a fine-tuned football model)")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="Detector inference image size (default: 960)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--max-ball-gap", type=int, default=15,
                        help="Frames the last known ball box may be held while the "
                             "ball is undetected, before it is reported unknown "
                             "(default: 15)")
    parser.add_argument("--show-speed", action="store_true", 
                        help="Show player speed in visualization")
    parser.add_argument("--show-distance", action="store_true", 
                        help="Show distance traveled in visualization")
    parser.add_argument("--fps", type=int, default=15, 
                        help="Output video FPS")

    args = parser.parse_args()
    
    # Check if input video exists
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        sys.exit(1)

    # Create configuration from arguments
    CONFIG = {
        'video_path': args.input,
        'yolo_model': args.yolo_model,
        'output_dir': args.output_dir,
        'match_id': 1,  # Default match ID
        'skip_frames': args.skip_frames,
        'max_frames': args.max_frames,
        'display': args.display,
        # Opt-in only, and only usable with a key; local YOLO otherwise.
        'use_roboflow': args.use_roboflow and bool(args.api_key),
        'use_cache': args.use_cache,
        'roboflow_api_key': args.api_key,
        'show_speed': args.show_speed,
        'show_distance': args.show_distance,
        'imgsz': args.imgsz,
        'confidence': args.conf,
        'max_ball_gap': args.max_ball_gap,
    }
    
    print("\n🏆 EyeCU Football Analysis Pipeline 🏆")
    print("=" * 50)
    print(f"Input video: {args.input}")
    print(f"Output directory: {args.output_dir}")
    if args.use_roboflow and not args.api_key:
        print("Warning: --use-roboflow given but ROBOFLOW_API_KEY is not set; "
              "falling back to the local YOLO model.")
    print(f"Detector: {'Roboflow cloud' if CONFIG['use_roboflow'] else 'local YOLO'}")
    print(f"YOLO model: {args.yolo_model}")
    print(f"Skip frames: {args.skip_frames}")
    if args.max_frames:
        print(f"Max frames: {args.max_frames}")
    print(f"Show visualization: {'Yes' if args.display else 'No'}")
    print("=" * 50 + "\n")
    
    # Ensure output directory exists
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Create subdirectories for outputs. bodies/faces/meshes/evaluation/
    # tracking_videos were only ever written by the removed legacy path.
    dirs = ['cache', 'reports', 'visualizations']

    for subdir in dirs:
        os.makedirs(os.path.join(CONFIG['output_dir'], subdir), exist_ok=True)
    
    # Initialize pipeline
    pipeline = FootballAnalysisPipeline(
        yolo_model=CONFIG['yolo_model'],
        output_dir=CONFIG['output_dir'],
        match_id=CONFIG['match_id'],
        use_roboflow=CONFIG['use_roboflow'],
        api_key=CONFIG['roboflow_api_key'],
        use_cache=CONFIG['use_cache'],
        show_speed=CONFIG['show_speed'],
        show_distance=CONFIG['show_distance'],
        imgsz=CONFIG['imgsz'],
        confidence=CONFIG['confidence'],
        max_ball_gap=CONFIG['max_ball_gap'],
    )
    
    # Process video
    stats = pipeline.process_video(
        video_path=CONFIG['video_path'],
        skip_frames=CONFIG['skip_frames'],
        max_frames=CONFIG['max_frames'],
        display_results=CONFIG['display']
    )
    
    # Save output video
    pipeline.save_output_video(args.output, fps=args.fps)
    
    # Generate final report
    report = pipeline.generate_final_report()
    
    # Cleanup
    pipeline.cleanup()
    
    print("\n✓ Pipeline execution complete!")
    print(f"✓ Output video saved to: {os.path.join(CONFIG['output_dir'], args.output)}")
    print(f"✓ All analysis results saved to: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()
