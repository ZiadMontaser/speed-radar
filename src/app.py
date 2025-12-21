import streamlit as st
import cv2
import numpy as np
import yaml
import json
import os
import tempfile
import zipfile
from pathlib import Path
from collections import deque
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Import pipeline modules (assumes they're in same directory)
try:
    from motion_detection import MotionDetector, MotionDetectionConfig
    from src.segmentation.segmentation import segment_foreground
    from src.tracking.tracking import Tracker
    from speed_capture import capture_violation_paper_method, compute_speed_paper_method, calculate_trajectory_metrics
    from src.data_structures import Frame, Calibration, TrackedObject
except ImportError:
    st.error("Pipeline modules not found. Ensure motion_detection.py, segmentation.py, tracking.py, speed_capture.py, and data_structures.py are in the same directory.")
    st.stop()


class FrameBuffer:
    def __init__(self, capacity=30):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, frame):
        self.buffer.append(frame)
    
    def get(self, index):
        for f in self.buffer:
            if f.index == index:
                return f
        return None
    
    def last_n(self, n):
        return list(self.buffer)[-n:]
    
    def latest(self):
        return self.buffer[-1] if self.buffer else None


def load_config():
    """Load configuration from config.yaml"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error("config.yaml not found. Please ensure it's in the working directory.")
        st.stop()


def extract_video_metadata(video_path):
    """Extract video metadata using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames,
        'duration': duration_sec
    }


def process_video(video_path, config, max_frames, progress_callback, frame_display_callback=None):
    """Main video processing pipeline"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, "Failed to open video file"
    
    # Initialize modules
    md_config = MotionDetectionConfig.from_dict(config)
    motion_detector = MotionDetector(config=md_config)
    tracker = Tracker(config)
    
    calibration = Calibration(
        scale_m_per_pixel=config.get('scale_m_per_pixel'),
        homography=None
    )
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer setup
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        output_path = tmp.name
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Processing loop
    frame_buffer = FrameBuffer(capacity=30)
    frame_idx = 0
    prev_frame = None
    prev2_frame = None
    
    violation_records = []
    
    frames_to_process = min(max_frames, total_frames) if max_frames else total_frames
    
    while cap.isOpened():
        if max_frames and frame_idx >= max_frames:
            break
        
        ret, frame_img = cap.read()
        if not ret:
            break
        
        timestamp = frame_idx / fps
        current_frame_obj = Frame(index=frame_idx, timestamp=timestamp, image=frame_img)
        frame_buffer.push(current_frame_obj)
        
        # Motion detection
        mask = motion_detector.compute_foreground_mask(frame_img, prev_frame, prev2_frame)
        
        if frame_idx < 2:
            prev2_frame = prev_frame
            prev_frame = frame_img.copy()
            frame_idx += 1
            out.write(frame_img)
            progress_callback(frame_idx, frames_to_process)
            continue
        
        # Segmentation and tracking
        regions = segment_foreground(mask, config)
        active_tracks = tracker.update(regions, frame_idx, timestamp)
        
        # Speed computation and violation capture
        for obj in active_tracks:
            if len(obj.trajectory) >= config['tracking']['min_trajectory_length']:
                speed = compute_speed_paper_method(obj, calibration, fps, config)
                obj.speed_m_s = speed
                
                speed_kmph = speed * 3.6
                if speed_kmph > config['speed_limit_kmph'] and not obj.captured_img_ref:
                    try:
                        json_path = capture_violation_paper_method(obj, list(frame_buffer.buffer), config)
                        obj.captured_img_ref = json_path
                        
                        # Record violation
                        with open(json_path, 'r') as f:
                            violation_data = json.load(f)
                        violation_records.append(violation_data)
                    except Exception as e:
                        print(f"Error capturing violation: {e}")
        
        # Visualization
        vis_frame = frame_img.copy()
        
        for obj in active_tracks:
            x, y, w, h = obj.bbox
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            id_text = f"ID: {obj.id}"
            cv2.putText(vis_frame, id_text, (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if obj.speed_m_s is not None:
                speed_kmph = obj.speed_m_s * 3.6
                speed_text = f"{speed_kmph:.1f} km/h"
                color = (0, 255, 0)
                if speed_kmph > config['speed_limit_kmph']:
                    color = (0, 0, 255)
                    cv2.putText(vis_frame, "VIOLATION", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                cv2.putText(vis_frame, speed_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if len(obj.trajectory) > 1:
                pts = [np.array(pt[1], dtype=np.int32) for pt in obj.trajectory]
                cv2.polylines(vis_frame, [np.array(pts)], False, (255, 0, 0), 1)
        
        out.write(vis_frame)
        
        prev2_frame = prev_frame
        prev_frame = frame_img.copy()
        frame_idx += 1
        progress_callback(frame_idx, frames_to_process)
        
        # Display frame every 100 frames for smooth real-time preview
        if frame_display_callback and frame_idx % 1 == 0:
            frame_display_callback(vis_frame)
    
    cap.release()
    out.release()
    
    # Get tracker statistics
    stats = {
        'total_objects_tracked': tracker.total_objects_tracked,
        'total_violations': len(violation_records),
        'active_tracks': len(tracker.tracked_objects),
        'exited_tracks': len(tracker.exited_tracks),
        'frames_processed': frame_idx
    }
    
    return {
        'output_video': output_path,
        'violations': violation_records,
        'stats': stats,
        'tracker_stats': tracker.get_statistics()
    }, None


def create_violations_report(violations):
    """Create a DataFrame report from violations"""
    data = []
    for v in violations:
        data.append({
            'ID': v.get('id', 'N/A'),
            'Speed (km/h)': v.get('speed_kmph', 0),
            'Speed (m/s)': v.get('speed_m_s', 0),
            'Frames': f"{v.get('Fr0', 0)}-{v.get('FrN', 0)}",
            'Trajectory Length': v.get('trajectory_length', 0),
            'Timestamp': v.get('timestamp', 'N/A'),
            'Image Path': v.get('image_path', 'N/A')
        })
    return pd.DataFrame(data)


def create_statistics_charts(violations, stats):
    """Create visualization charts"""
    charts = {}
    
    # Speed distribution
    if violations:
        speeds = [v.get('speed_kmph', 0) for v in violations]
        fig_dist = go.Figure(data=[
            go.Histogram(x=speeds, nbinsx=20, name='Speed Distribution')
        ])
        fig_dist.update_layout(
            title='Speed Distribution of Violations',
            xaxis_title='Speed (km/h)',
            yaxis_title='Count',
            template='plotly_dark'
        )
        charts['speed_dist'] = fig_dist
    
    # Statistics summary
    stats_text = f"""
    **Processing Summary**
    - Total Objects Tracked: {stats.get('total_objects_tracked', 0)}
    - Total Violations: {stats.get('total_violations', 0)}
    - Frames Processed: {stats.get('frames_processed', 0)}
    - Exited Tracks: {stats.get('exited_tracks', 0)}
    """
    charts['stats_text'] = stats_text
    
    return charts


def download_violations_zip(violations_folder):
    """Create a ZIP file of all violation data"""
    if not os.path.exists(violations_folder):
        return None
    
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        zip_path = tmp.name
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(violations_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, violations_folder)
                zipf.write(file_path, arcname)
    
    return zip_path


# ============ STREAMLIT APP ============

st.set_page_config(page_title="Speed Detection Camera System", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 0;
    }
    .metric-card {
        background-color: #1f77b4;
        padding: 20px;
        border-radius: 8px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load config
config = load_config()

# ============ SECTION 1: LANDING PAGE ============
st.title("🚗 Speed Detection Camera System")
st.markdown("""
An AI-powered traffic monitoring and speed violation detection system using computer vision.
Detects vehicles, tracks their motion, computes speeds, and captures violations.
""")

# Key features
st.markdown("""
### Key Features
- **Real-time Motion Detection**: Adaptive background model with shadow removal
- **Vehicle Tracking**: Multi-object tracking with Kalman filtering
- **Accurate Speed Computation**: Paper-based methodology with homography support
- **Violation Capture**: Multi-frame enhanced image capture with metadata
- **Detailed Reporting**: Statistics, charts, and downloadable violation records
""")

# ============ SECTION 2: VIDEO UPLOAD ============
st.divider()
st.header("📹 Upload Your Traffic Video")

uploaded_file = st.file_uploader("Select a video file (.mp4, .avi, .mov)", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name
    
    # ============ SECTION 3: VIDEO METADATA ============
    st.header("📊 Video Information")
    
    metadata = extract_video_metadata(video_path)
    
    if metadata:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Resolution", f"{metadata['width']}x{metadata['height']}")
        with col2:
            st.metric("FPS", f"{metadata['fps']:.1f}")
        with col3:
            st.metric("Total Frames", metadata['total_frames'])
        with col4:
            st.metric("Duration", f"{metadata['duration']:.1f}s")
    else:
        st.error("Failed to extract video metadata")
        st.stop()
    
    # ============ SECTION 4: CONFIG CONTROLS ============
    with st.sidebar:
        st.header("⚙️ Settings")
        
        scale_m_per_pixel = st.slider(
            "Scale (m/pixel)",
            min_value=0.01,
            max_value=0.1,
            value=config.get('scale_m_per_pixel', 0.05),
            step=0.01
        )
        config['scale_m_per_pixel'] = scale_m_per_pixel
        
        speed_limit_kmph = st.number_input(
            "Speed Limit (km/h)",
            min_value=1,
            max_value=200,
            value=config.get('speed_limit_kmph', 60)
        )
        config['speed_limit_kmph'] = speed_limit_kmph
        
        min_trajectory_length = st.slider(
            "Min Trajectory Length",
            min_value=3,
            max_value=15,
            value=config.get('tracking', {}).get('min_trajectory_length', 5)
        )
        config['tracking']['min_trajectory_length'] = min_trajectory_length
        
        max_frames = st.number_input(
            "Max Frames to Process",
            min_value=100,
            max_value=metadata['total_frames'],
            value=min(2000, metadata['total_frames']),
            step=100
        )
        
        alpha_background = st.slider(
            "Background Learning Rate",
            min_value=0.001,
            max_value=0.1,
            value=config.get('alpha_background', 0.01),
            step=0.001
        )
        config['alpha_background'] = alpha_background
        
        motion_threshold = st.slider(
            "Motion Threshold",
            min_value=10,
            max_value=50,
            value=config.get('motion_threshold', 30)
        )
        config['motion_threshold'] = motion_threshold
    
    # ============ SECTION 5: PROCESSING ============
    st.divider()
    st.header("🔄 Processing")
    
    if st.button("🚀 Start Detection", key="process_btn", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"Processing: {current}/{total} frames ({progress*100:.1f}%)")
        
        # Create placeholder for real-time frame display
        frame_display_col1, frame_display_col2 = st.columns([3, 1])
        with frame_display_col1:
            frame_placeholder = st.empty()
        with frame_display_col2:
            frame_info = st.empty()
        
        current_frame_count = {"count": 0}
        
        def display_frame(frame):
            current_frame_count["count"] += 1
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_container_width=True)
            frame_info.metric("Current Frame", current_frame_count["count"])
        
        with st.spinner("Processing video..."):
            result, error = process_video(video_path, config, max_frames, update_progress, display_frame)
        
        if error:
            st.error(f"Processing failed: {error}")
        else:
            st.success("✅ Processing complete!")
            
            # Store results in session state
            st.session_state.processing_result = result
            st.session_state.video_processed = True
    
    # ============ SECTION 6: VIOLATION REPORT ============
    if st.session_state.get('video_processed', False):
        st.divider()
        st.header("📋 Violation Report")
        
        result = st.session_state.processing_result
        violations = result['violations']
        stats = result['stats']
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Violations", stats['total_violations'])
        with col2:
            st.metric("Objects Tracked", stats['total_objects_tracked'])
        with col3:
            st.metric("Avg Speed", f"{np.mean([v['speed_kmph'] for v in violations]):.1f} km/h" if violations else "N/A")
        with col4:
            st.metric("Speed Limit", f"{config['speed_limit_kmph']} km/h")
        
        # Violations table
        if violations:
            violations_df = create_violations_report(violations)
            st.subheader("Violations Table")
            st.dataframe(violations_df, use_container_width=True)
            
            # Charts
            st.subheader("Statistics & Visualization")
            charts = create_statistics_charts(violations, stats)
            
            if 'speed_dist' in charts:
                st.plotly_chart(charts['speed_dist'], use_container_width=True)
            
            st.markdown(charts['stats_text'])
        else:
            st.info("No violations detected in the processed video.")
        
        # ============ SECTION 7: EXPORT OPTIONS ============
        st.divider()
        st.header("📥 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⬇️ Download Video", use_container_width=True):
                with open(result['output_video'], 'rb') as f:
                    st.download_button(
                        label="Download Processed Video",
                        data=f.read(),
                        file_name=f"processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4"
                    )
        
        with col2:
            if st.button("📦 Download Violations (ZIP)", use_container_width=True):
                violations_folder = config.get('violation_save_folder', './violations')
                zip_path = download_violations_zip(violations_folder)
                if zip_path:
                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            label="Download Violations ZIP",
                            data=f.read(),
                            file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
                else:
                    st.warning("No violations folder found")
        
        with col3:
            if st.button("📊 Download Report (CSV)", use_container_width=True):
                if violations:
                    violations_df = create_violations_report(violations)
                    csv = violations_df.to_csv(index=False)
                    st.download_button(
                        label="Download Report CSV",
                        data=csv,
                        file_name=f"violations_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

else:
    st.info("👆 Upload a video file to get started")

# Initialize session state
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'processing_result' not in st.session_state:
    st.session_state.processing_result = None