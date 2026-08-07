"""
main.py
Streamlit web interface for Football Analysis MVP
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import json
import tempfile
import os
from datetime import datetime

from pipeline import FootballAnalysisPipeline
from config import Config


# Page configuration
st.set_page_config(
    page_title="Football Analysis MVP",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .event-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .goal-badge { background-color: #4CAF50; color: white; }
    .shot-badge { background-color: #FF9800; color: white; }
    .pass-badge { background-color: #2196F3; color: white; }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'config' not in st.session_state:
        st.session_state.config = Config()


def render_sidebar():
    """Render sidebar with configuration options"""
    st.sidebar.title("⚙️ Configuration")
    
    # Video Processing
    st.sidebar.subheader("Video Processing")
    skip_frames = st.sidebar.slider("Skip Frames", 1, 10, 2,
                                    help="Process every N frames")
    max_frames = st.sidebar.number_input("Max Frames", 0, 10000, 0,
                                         help="0 = process all frames")
    output_fps = st.sidebar.slider("Output FPS", 10, 60, 15)
    
    # Team Assignment
    st.sidebar.subheader("Team Assignment")
    team_method = st.sidebar.selectbox(
        "Method",
        ["hybrid", "color", "eyecu"],
        help="Hybrid combines both methods"
    )
    
    if team_method == "hybrid":
        hamza_weight = st.sidebar.slider("Color Clustering Weight", 0.0, 1.0, 0.5)
        eyecu_weight = 1.0 - hamza_weight
    else:
        hamza_weight = 1.0 if team_method == "color" else 0.0
        eyecu_weight = 1.0 if team_method == "eyecu" else 0.0
    
    # OCR Settings
    st.sidebar.subheader("Player ID (OCR)")
    ocr_engine = st.sidebar.selectbox("OCR Engine", ["paddleocr", "tesseract"])
    ocr_confidence = st.sidebar.slider("OCR Confidence", 0.0, 1.0, 0.5)
    
    # Event Detection
    st.sidebar.subheader("Event Detection")
    detect_goals = st.sidebar.checkbox("Detect Goals", value=True)
    detect_shots = st.sidebar.checkbox("Detect Shots", value=True)
    detect_passes = st.sidebar.checkbox("Detect Passes", value=True)
    
    # Highlights
    st.sidebar.subheader("Highlights")
    generate_highlights = st.sidebar.checkbox("Generate Highlights", value=True)
    pre_buffer = st.sidebar.slider("Pre-Event Buffer (s)", 0.0, 10.0, 3.0)
    post_buffer = st.sidebar.slider("Post-Event Buffer (s)", 0.0, 15.0, 5.0)
    
    # Visualization
    st.sidebar.subheader("Visualization")
    show_player_ids = st.sidebar.checkbox("Show Player IDs", value=True)
    show_speed = st.sidebar.checkbox("Show Speed", value=False)
    show_distance = st.sidebar.checkbox("Show Distance", value=False)
    
    # Update config
    config = st.session_state.config
    config.video.skip_frames = skip_frames
    config.video.max_frames = max_frames if max_frames > 0 else None
    config.video.output_fps = output_fps
    
    config.team_assignment.hamza_weight = hamza_weight
    config.team_assignment.eyecu_weight = eyecu_weight
    
    config.ocr.ocr_engine = ocr_engine
    config.ocr.confidence_threshold = ocr_confidence
    
    config.event_detection.detect_goals = detect_goals
    config.event_detection.detect_shots = detect_shots
    config.event_detection.detect_passes = detect_passes
    
    config.highlight.pre_event_buffer = pre_buffer
    config.highlight.post_event_buffer = post_buffer
    
    config.visualization.show_player_ids = show_player_ids
    config.visualization.show_speed = show_speed
    config.visualization.show_distance = show_distance
    
    st.session_state.config = config
    
    return generate_highlights


def render_header():
    """Render main header"""
    st.markdown('<div class="main-header">⚽ Football Analysis MVP</div>', 
                unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; color: #666; font-size: 1.1rem;">
    AI-powered football match analysis with player tracking, team assignment, 
    jersey OCR, and event detection
    </p>
    """, unsafe_allow_html=True)


def render_upload_section():
    """Render video upload section"""
    st.markdown('<div class="section-header">📹 Upload Video</div>', 
                unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a football match video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a football match video for analysis"
    )
    
    return uploaded_file


def process_video(video_path: str, generate_highlights: bool):
    """Process video through pipeline"""
    try:
        # Initialize pipeline
        st.session_state.pipeline = FootballAnalysisPipeline(st.session_state.config)
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Processing video...")
        
        # Process video
        results = st.session_state.pipeline.process_video(
            video_path=video_path,
            output_path=None
        )
        
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        
        st.session_state.results = results
        st.session_state.processing = False
        
        return results
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        st.session_state.processing = False
        return None


def render_results(results: dict):
    """Render analysis results"""
    if results is None:
        return
    
    st.markdown('<div class="section-header">📊 Analysis Results</div>', 
                unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Frames Processed",
            f"{results['frames_processed']:,}",
            f"of {results['total_frames']:,}"
        )
    
    with col2:
        st.metric(
            "Events Detected",
            results['events_detected']
        )
    
    with col3:
        st.metric(
            "Highlights Generated",
            results['highlights_generated']
        )
    
    with col4:
        player_stats = results.get('player_stats', {})
        st.metric(
            "Players Identified",
            player_stats.get('total_players', 0)
        )
    
    # Detailed statistics in tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview",
        "👥 Teams",
        "🎽 Players",
        "🎯 Events"
    ])
    
    with tab1:
        render_overview_tab(results)
    
    with tab2:
        render_team_tab(results)
    
    with tab3:
        render_player_tab(results)
    
    with tab4:
        render_event_tab(results)


def render_overview_tab(results: dict):
    """Render overview tab"""
    st.subheader("Processing Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Video Information**")
        st.write(f"- Total Frames: {results['total_frames']:,}")
        st.write(f"- Frames Processed: {results['frames_processed']:,}")
        st.write(f"- Output Video: `{Path(results['output_video']).name}`")
    
    with col2:
        st.write("**Analysis Summary**")
        team_stats = results.get('team_stats', {})
        player_stats = results.get('player_stats', {})
        event_summary = results.get('event_summary', {})
        
        st.write(f"- Players Detected: {team_stats.get('total_players', 0)}")
        st.write(f"- Players Identified: {player_stats.get('total_players', 0)}")
        st.write(f"- Events Detected: {event_summary.get('total_events', 0)}")
    
    # Download buttons
    st.subheader("Downloads")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if Path(results['output_video']).exists():
            with open(results['output_video'], 'rb') as f:
                st.download_button(
                    "📥 Download Tracked Video",
                    f,
                    file_name=Path(results['output_video']).name,
                    mime="video/mp4"
                )
    
    with col2:
        report_dir = Path(st.session_state.config.output.output_dir) / "reports"
        summary_file = report_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                st.download_button(
                    "📥 Download JSON Report",
                    f,
                    file_name="analysis_report.json",
                    mime="application/json"
                )
    
    with col3:
        events_file = report_dir / "events.json"
        if events_file.exists():
            with open(events_file, 'r') as f:
                st.download_button(
                    "📥 Download Events",
                    f,
                    file_name="events.json",
                    mime="application/json"
                )


def render_team_tab(results: dict):
    """Render team statistics tab"""
    st.subheader("Team Assignment Statistics")
    
    team_stats = results.get('team_stats', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Players Tracked", team_stats.get('total_players', 0))
        st.metric("Players Assigned", team_stats.get('assigned_players', 0))
    
    with col2:
        avg_conf = team_stats.get('avg_confidence', 0)
        st.metric("Average Confidence", f"{avg_conf:.2%}")
    
    # Team distribution
    if 'team_distribution' in team_stats:
        st.subheader("Team Distribution")
        distribution = team_stats['team_distribution']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Team A", distribution.get('0', 0) + distribution.get(0, 0))
        
        with col2:
            st.metric("Team B", distribution.get('1', 0) + distribution.get(1, 0))


def render_player_tab(results: dict):
    """Render player statistics tab"""
    st.subheader("Player Identification Statistics")
    
    player_stats = results.get('player_stats', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tracks", player_stats.get('total_tracks', 0))
    
    with col2:
        st.metric("Unique Players", player_stats.get('total_players', 0))
    
    with col3:
        st.metric("OCR Detections", player_stats.get('ocr_detections', 0))
    
    avg_conf = player_stats.get('avg_confidence', 0)
    st.metric("Average OCR Confidence", f"{avg_conf:.2%}")
    
    # Player list
    if 'players' in player_stats and player_stats['players']:
        st.subheader("Identified Players")
        
        players_data = []
        for player_id, info in player_stats['players'].items():
            if isinstance(info, dict):
                players_data.append({
                    'Player ID': player_id,
                    'Jersey Number': info.get('jersey_number', 'N/A'),
                    'Team': info.get('team', 'N/A'),
                    'Confidence': f"{info.get('confidence', 0):.2%}"
                })
        
        if players_data:
            st.dataframe(players_data, use_container_width=True)


def render_event_tab(results: dict):
    """Render events tab"""
    st.subheader("Detected Events")
    
    event_summary = results.get('event_summary', {})
    
    # Event type badges
    st.write("**Event Types:**")
    event_types = event_summary.get('event_types', {})
    
    badges_html = ""
    for event_type, count in event_types.items():
        badge_class = f"{event_type}-badge"
        badges_html += f'<span class="event-badge {badge_class}">{event_type.upper()}: {count}</span>'
    
    st.markdown(badges_html, unsafe_allow_html=True)
    
    # Load and display events
    report_dir = Path(st.session_state.config.output.output_dir) / "reports"
    events_file = report_dir / "events.json"
    
    if events_file.exists():
        with open(events_file, 'r') as f:
            events_data = json.load(f)
        
        if 'events' in events_data and events_data['events']:
            st.subheader("Event Timeline")
            
            events_list = []
            for event in events_data['events']:
                events_list.append({
                    'Time': f"{event['timestamp']:.1f}s",
                    'Type': event['event_type'].upper(),
                    'Player': event.get('player_id', 'N/A'),
                    'Team': event.get('team_id', 'N/A'),
                    'Confidence': f"{event.get('confidence', 0):.2%}"
                })
            
            st.dataframe(events_list, use_container_width=True)
    
    # Highlight clips
    if results['highlights_generated'] > 0:
        st.subheader("Highlight Clips")
        
        highlight_dir = Path(st.session_state.config.output.output_dir) / "highlights"
        highlight_files = sorted(highlight_dir.glob("highlight_*.mp4"))
        
        if highlight_files:
            selected_highlight = st.selectbox(
                "Select highlight to view",
                [f.name for f in highlight_files]
            )
            
            if selected_highlight:
                highlight_path = highlight_dir / selected_highlight
                if highlight_path.exists():
                    st.video(str(highlight_path))


def main():
    """Main application"""
    initialize_session_state()
    
    render_header()
    
    # Sidebar configuration
    generate_highlights = render_sidebar()
    
    # Main content
    uploaded_file = render_upload_section()
    
    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Process button
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            st.session_state.processing = True
            
            with st.spinner("Processing video... This may take a while."):
                results = process_video(video_path, generate_highlights)
            
            # Clean up temp file
            try:
                os.unlink(video_path)
            except:
                pass
    
    # Display results if available
    if st.session_state.results is not None:
        render_results(st.session_state.results)


if __name__ == '__main__':
    main()