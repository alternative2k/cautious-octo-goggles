import uuid
import asyncio
import time
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import av
import streamlit as st
import numpy as np

# ✅ HEADLESS OPENCV SETUP
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264"

try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except:
    pass  # Graceful fallback

from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import mediapipe as mp

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ... rest of your code unchanged ...
