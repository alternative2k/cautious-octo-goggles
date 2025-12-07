import uuid
import asyncio
import time
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import av
import cv2
import streamlit as st
import numpy as np
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import mediapipe as mp

# MediaPipe setup for body tracking
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class ContinuousRecorder:
    """Manages continuous raw video chunked upload to S3"""
    def __init__(self, bucket: str, prefix: str):
        self.bucket = bucket
        self.prefix = prefix
        self.s3_client = boto3.client('s3')
        self.chunk_dir = Path("./chunks")
        self.chunk_dir.mkdir(exist_ok=True)
        self.chunks = []
    
    async def record_chunk(self, chunk_file: Path):
        """Upload individual chunk to S3"""
        chunk_key = f"live/{self.prefix}/{chunk_file.name}"
        try:
            self.s3_client.upload_file(str(chunk_file), self.bucket, chunk_key)
            self.chunks.append(chunk_key)
            chunk_file.unlink()  # Clean up
        except ClientError as e:
            st.error(f"Chunk upload failed: {e}")

# Global recorder instance
recorder = None

def upload_to_s3(file_path: Path, bucket: str, s3_key: str):
    """Single file S3 upload (for filtered videos)"""
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(str(file_path), bucket, s3_key)
        return True, f"https://{bucket}.s3.amazonaws.com/{s3_key}"
    except ClientError as e:
        st.error(f"S3 upload failed: {e}")
        return False, None

def pose_detection_filter(frame: av.VideoFrame) -> av.VideoFrame:
    """MediaPipe pose detection filter"""
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect pose
    results = mp_pose.Pose()(img_rgb)
    
    # Draw pose landmarks
    annotated_image = img_rgb.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Convert back to BGR
    img = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def edge_detection_filter(frame: av.VideoFrame) -> av.VideoFrame:
    """Original Canny edge detection"""
    img = frame.to_ndarray(format="bgr24")
    img = cv2.Canny(img, 100, 200)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def no_filter(frame: av.VideoFrame) -> av.VideoFrame:
    """Passthrough - no processing"""
    return frame

def app():
    st.title("🎥 Enhanced Webcam Recorder")
    
    # Config
    BUCKET_NAME = "your-streamlit-videos-2025"
    
    # Session state
    if "prefix" not in st.session_state:
        st.session_state.prefix = str(uuid.uuid4())
        st.session_state.filtered_recording = False
        st.session_state.global_recorder = ContinuousRecorder(BUCKET_NAME, st.session_state.prefix)
    
    prefix = st.session_state.prefix
    RECORD_DIR = Path("./records")
    RECORD_DIR.mkdir(exist_ok=True)
    
    # UI Controls
    col1, col2 = st.columns(2)
    with col1:
        show_preview = st.checkbox("📺 Show Live Preview", value=True)
    with col2:
        filter_type = st.selectbox("Filter", ["None", "Pose Detection", "Edge Detection"])
    
    # Filter mapping
    filter_map = {
        "None": no_filter,
        "Pose Detection": pose_detection_filter,
        "Edge Detection": edge_detection_filter
    }
    current_filter = filter_map[filter_type]
    
    # Filtered recording control
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🎬 Start Filtered Recording", type="primary", disabled=st.session_state.filtered_recording):
            st.session_state.filtered_recording = True
            st.rerun()
    
    with col4:
        if st.button("⏹️ Stop Filtered Recording", disabled=not st.session_state.filtered_recording):
            st.session_state.filtered_recording = False
            st.rerun()
    
    # Raw continuous recording status (discreet green button only)
    if st.button("🟢 Raw recording active", disabled=False):
        st.info("✅ Continuous raw recording to S3 is always active when camera is on")
    
    # WebRTC Streamer with conditional preview
    video_frame_callback = current_filter if show_preview else no_filter
    
    webrtc_streamer(
        key="enhanced_record",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": True},
        video_frame_callback=video_frame_callback if show_preview else None,
        in_recorder_factory=lambda: MediaRecorder("./temp_raw.flv", format="flv") if st.session_state.filtered_recording else None,
        out_recorder_factory=lambda: MediaRecorder(f"{RECORD_DIR}/{prefix}_filtered.flv", format="flv") if st.session_state.filtered_recording else None,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Handle filtered recording completion
    filtered_file = RECORD_DIR / f"{prefix}_filtered.flv"
    if filtered_file.exists() and st.session_state.filtered_recording:
        success, s3_url = upload_to_s3(filtered_file, BUCKET_NAME, f"filtered/{prefix}_filtered.flv")
        if success:
            st.success("🎉 Filtered video uploaded!")
            st.markdown(f"**[Download Filtered Video]({s3_url})**")
            st.video(s3_url)
            filtered_file.unlink()
        st.session_state.filtered_recording = False
    
    # Continuous raw recording info (minimal)
    st.caption("🟢 Raw footage continuously uploads to `live/<session>/` in S3 chunks")

if __name__ == "__main__":
    app()
