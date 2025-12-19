import uuid
import threading
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
    def __init__(self, bucket_name, prefix):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.frames = []
        self.last_flush = time.time()
        self.flush_interval = 5.0  # seconds
        self.chunk_idx = 0
        self.lock = threading.Lock()
        self.active_uploads = 0

    def add_frame(self, frame: av.VideoFrame):
        # Convert to numpy immediately to safely store data off-thread
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            self.frames.append(img)
            if time.time() - self.last_flush >= self.flush_interval:
                self.flush()

    def flush(self):
        with self.lock:
            if not self.frames:
                return
            frames_to_process = self.frames
            self.frames = []
            self.last_flush = time.time()
            idx = self.chunk_idx
            self.chunk_idx += 1
        
        # Process in background thread
        threading.Thread(target=self._upload_chunk, args=(frames_to_process, idx)).start()

    def _upload_chunk(self, frames, idx):
        if not frames: return
        self.active_uploads += 1
        filename = f"chunk_{idx}.mp4"
        path = Path(filename)
        try:
            height, width = frames[0].shape[:2]
            container = av.open(str(path), mode='w')
            stream = container.add_stream('h264', rate=30)
            stream.width = width
            stream.height = height
            stream.pix_fmt = 'yuv420p'
            
            for img in frames:
                frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            
            s3 = boto3.client('s3')
            key = f"live/{self.prefix}/{filename}"
            s3.upload_file(str(path), self.bucket_name, key)
        except Exception as e:
            print(f"Chunk upload failed: {e}")
        finally:
            if path.exists():
                path.unlink()
            self.active_uploads -= 1

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
    # Initialize Pose object once to improve performance
    if not hasattr(pose_detection_filter, "pose"):
        pose_detection_filter.pose = mp_pose.Pose()

    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect pose
    results = pose_detection_filter.pose.process(img_rgb)
    
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
    
    if "recorder" not in st.session_state:
        st.session_state.recorder = ContinuousRecorder(BUCKET_NAME, st.session_state.prefix)
    
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
            st.session_state.upload_needed = True
            st.rerun()
    
    # WebRTC Streamer with conditional preview
    recorder = st.session_state.recorder
    
    def frame_processor(frame: av.VideoFrame) -> av.VideoFrame:
        recorder.add_frame(frame)
        out_frame = current_filter(frame)
        
        # Visual indicator for uploading (Yellow dot in top-right)
        if recorder.active_uploads > 0:
            img = out_frame.to_ndarray(format="bgr24")
            cv2.circle(img, (img.shape[1] - 20, 20), 8, (0, 255, 255), -1)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        return out_frame
    
    webrtc_streamer(
        key="enhanced_record",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": True},
        video_frame_callback=frame_processor if show_preview else None,
        in_recorder_factory=lambda: MediaRecorder("./temp_raw.flv", format="flv") if st.session_state.filtered_recording else None,
        out_recorder_factory=lambda: MediaRecorder(f"{RECORD_DIR}/{prefix}_filtered.flv", format="flv") if st.session_state.filtered_recording else None,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Handle filtered recording completion
    if st.session_state.get("upload_needed"):
        filtered_file = RECORD_DIR / f"{prefix}_filtered.flv"
        if filtered_file.exists():
            success, s3_url = upload_to_s3(filtered_file, BUCKET_NAME, f"filtered/{prefix}_filtered.flv")
            if success:
                st.success("🎉 Filtered video uploaded!")
                st.markdown(f"**Download Filtered Video**")
                st.video(s3_url)
                filtered_file.unlink()
        st.session_state.upload_needed = False

if __name__ == "__main__":
    app()
