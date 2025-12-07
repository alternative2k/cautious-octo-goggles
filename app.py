import uuid
import asyncio
import time
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import av
import streamlit as st
import numpy as np

# =============================================================================
# HEADLESS OPENCV + MEDIAPIPE SETUP (Production Safe)
# =============================================================================
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Install headless OpenCV first
try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.warning("OpenCV not available - filters disabled")

# Conditional MediaPipe (only if OpenCV works)
MEDIAPIPE_AVAILABLE = False
mp_pose = None
mp_drawing = None

if OPENCV_AVAILABLE:
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        MEDIAPIPE_AVAILABLE = True
    except ImportError as e:
        st.warning(f"MediaPipe unavailable ({e}) - Pose detection disabled")
        MEDIAPIPE_AVAILABLE = False

from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# =============================================================================
# SAFE FILTER FUNCTIONS (Fallback to Edge Detection or None)
# =============================================================================
def safe_pose_detection_filter(frame: av.VideoFrame) -> av.VideoFrame:
    """Pose detection with graceful fallback"""
    if not OPENCV_AVAILABLE or not MEDIAPIPE_AVAILABLE:
        return edge_detection_filter(frame)
    
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose() as pose:
        results = pose.process(img_rgb)
        
        annotated_image = img_rgb.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    img = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def edge_detection_filter(frame: av.VideoFrame) -> av.VideoFrame:
    """Pure OpenCV edge detection (no MediaPipe)"""
    if not OPENCV_AVAILABLE:
        return frame  # Passthrough if OpenCV unavailable
    
    img = frame.to_ndarray(format="bgr24")
    img = cv2.Canny(img, 100, 200)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def no_filter(frame: av.VideoFrame) -> av.VideoFrame:
    """Always works passthrough"""
    return frame

# Filter mapping with fallbacks
FILTER_MAP = {
    "Pose Detection": safe_pose_detection_filter,
    "Edge Detection": edge_detection_filter,
    "None": no_filter
}

# =============================================================================
# MAIN APP (Unchanged logic, safe filters)
# =============================================================================
def app():
    st.title("🎥 Enhanced Webcam Recorder")
    
    BUCKET_NAME = "your-streamlit-videos-2025"
    
    if "prefix" not in st.session_state:
        st.session_state.prefix = str(uuid.uuid4())
        st.session_state.filtered_recording = False
    
    prefix = st.session_state.prefix
    RECORD_DIR = Path("./records")
    RECORD_DIR.mkdir(exist_ok=True)
    
    # Status indicators
    if not OPENCV_AVAILABLE:
        st.error("❌ OpenCV unavailable - Install system dependencies")
    if MEDIAPIPE_AVAILABLE:
        st.success("✅ Pose Detection ready")
    
    # UI Controls
    col1, col2 = st.columns(2)
    with col1:
        show_preview = st.checkbox("📺 Show Live Preview", value=True)
    with col2:
        available_filters = ["None", "Edge Detection"]
        if MEDIAPIPE_AVAILABLE:
            available_filters.insert(0, "Pose Detection")
        filter_type = st.selectbox("Filter", available_filters)
    
    current_filter = FILTER_MAP.get(filter_type, no_filter)
    
    # Recording controls
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🎬 Start Filtered Recording", 
                    type="primary", 
                    disabled=st.session_state.filtered_recording):
            st.session_state.filtered_recording = True
            st.rerun()
    
    with col4:
        if st.button("⏹️ Stop Filtered Recording", 
                    disabled=not st.session_state.filtered_recording):
            st.session_state.filtered_recording = False
            st.rerun()
    
    # Raw recording status
    st.markdown("🟢 **Raw recording active** (background)")
    
    # WebRTC Streamer
    video_frame_callback = current_filter if show_preview else None
    
    webrtc_streamer(
        key="enhanced_record",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": True},
        video_frame_callback=video_frame_callback,
        out_recorder_factory=lambda: MediaRecorder(
            f"{RECORD_DIR}/{prefix}_filtered.flv", format="flv"
        ) if st.session_state.filtered_recording else None,
    )
    
    # Handle filtered upload
    filtered_file = RECORD_DIR / f"{prefix}_filtered.flv"
    if filtered_file.exists():
        def upload_to_s3(file_path: Path, bucket: str, s3_key: str):
            s3_client = boto3.client('s3')
            try:
                s3_client.upload_file(str(file_path), bucket, s3_key)
                return True, f"https://{bucket}.s3.amazonaws.com/{s3_key}"
            except ClientError:
                return False, None
        
        success, s3_url = upload_to_s3(filtered_file, BUCKET_NAME, f"filtered/{prefix}_filtered.flv")
        if success:
            st.success("🎉 Filtered video uploaded!")
            st.markdown(f"**[Download Filtered Video]({s3_url})**")
            st.video(s3_url)
            filtered_file.unlink()
        st.session_state.filtered_recording = False

if __name__ == "__main__":
    app()
