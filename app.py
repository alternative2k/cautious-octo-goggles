import uuid
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import av
import streamlit as st
import numpy as np

# Headless OpenCV + MediaPipe (Safe)
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    OPENCV_AVAILABLE = True
except:
    OPENCV_AVAILABLE = False

MEDIAPIPE_AVAILABLE = False
if OPENCV_AVAILABLE:
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        MEDIAPIPE_AVAILABLE = True
    except:
        MEDIAPIPE_AVAILABLE = False

from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# =============================================================================
# SAFE FILTERS (No session state dependencies)
# =============================================================================
def safe_pose_filter(frame: av.VideoFrame) -> av.VideoFrame:
    if not OPENCV_AVAILABLE or not MEDIAPIPE_AVAILABLE:
        return edge_filter(frame)
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose() as pose:
        results = pose.process(img_rgb)
        annotated = img_rgb.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    img = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def edge_filter(frame: av.VideoFrame) -> av.VideoFrame:
    if not OPENCV_AVAILABLE:
        return frame
    img = frame.to_ndarray(format="bgr24")
    img = cv2.Canny(img, 100, 200)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def no_filter(frame: av.VideoFrame) -> av.VideoFrame:
    return frame

FILTERS = {
    "Pose Detection": safe_pose_filter,
    "Edge Detection": edge_filter,
    "None": no_filter
}

# =============================================================================
# S3 UPLOAD (Safe)
# =============================================================================
def upload_to_s3(file_path: Path, bucket: str, s3_key: str):
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(str(file_path), bucket, s3_key)
        return True, f"https://{bucket}.s3.amazonaws.com/{s3_key}"
    except ClientError:
        return False, None

# =============================================================================
# MAIN APP - SESSION STATE INITIALIZED FIRST
# =============================================================================
def app():
    st.title("🎥 Enhanced Webcam Recorder")
    
    # CRITICAL: Initialize ALL session state FIRST
    for key in ["prefix", "filtered_recording", "filtered_done"]:
        if key not in st.session_state:
            st.session_state[key] = False if "recording" in key or "done" in key else str(uuid.uuid4())
    
    prefix = st.session_state.prefix
    RECORD_DIR = Path("./records")
    RECORD_DIR.mkdir(exist_ok=True)
    
    BUCKET_NAME = "your-streamlit-videos-2025"
    
    # Status
    col1, col2 = st.columns(2)
    with col1:
        if OPENCV_AVAILABLE:
            st.success("✅ OpenCV Ready")
        else:
            st.error("❌ OpenCV Missing")
    with col2:
        if MEDIAPIPE_AVAILABLE:
            st.success("✅ Pose Detection Ready")
    
    # Controls
    col1, col2 = st.columns(2)
    with col1:
        show_preview = st.checkbox("📺 Show Live Preview", value=True)
    with col2:
        avail_filters = ["None", "Edge Detection"] + (["Pose Detection"] if MEDIAPIPE_AVAILABLE else [])
        filter_name = st.selectbox("Filter", avail_filters)
    
    # Recording buttons (Safe state access)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬 Start Filtered", type="primary", disabled=st.session_state.filtered_recording):
            st.session_state.filtered_recording = True
            st.session_state.filtered_done = False
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Filtered", disabled=not st.session_state.filtered_recording):
            st.session_state.filtered_recording = False
            st.rerun()
    
    st.markdown("🟢 **Raw recording active** (background)")
    
    # ✅ SAFE RECORDER FACTORIES (No session state in lambdas)
    def make_filtered_recorder():
        if st.session_state.filtered_recording:
            return MediaRecorder(f"{RECORD_DIR}/{prefix}_filtered.flv", format="flv")
        return None
    
    # WebRTC - SAFE callbacks only
    video_cb = FILTERS.get(filter_name, no_filter) if show_preview else None
    
    webrtc_streamer(
        key="record_safe",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": True},
        video_frame_callback=video_cb,
        out_recorder_factory=make_filtered_recorder,  # SAFE function
    )
    
    # Handle upload (runs after recording stops)
    filtered_file = RECORD_DIR / f"{prefix}_filtered.flv"
    if filtered_file.exists() and not st.session_state.filtered_done:
        success, s3_url = upload_to_s3(filtered_file, BUCKET_NAME, f"filtered/{prefix}_filtered.flv")
        if success:
            st.success("🎉 Filtered video uploaded!")
            st.markdown(f"**[📥 Download]({s3_url})**")
            st.video(s3_url)
            filtered_file.unlink()
            st.session_state.filtered_done = True
        else:
            with filtered_file.open("rb") as f:
                st.download_button("💾 Download Local", f, f"{prefix}_filtered.flv")

if __name__ == "__main__":
    app()
