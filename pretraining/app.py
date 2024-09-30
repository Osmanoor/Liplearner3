import streamlit as st
import tempfile
import pipeline
from preprocess import save2vid, preprocess_video
import time
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import threading
import os
# from twilio.rest import Client
import uuid
from pathlib import Path
from aiortc.contrib.media import MediaRecorder
import ffmpeg


# account_sid = ''
# auth_token = ''
# client = Client(account_sid, auth_token)

# token = client.tokens.create()

peer_connection_config = {
  "iceServers": [
    {
      "url": "stun:global.stun.twilio.com:3478",
      "urls": "stun:global.stun.twilio.com:3478"
    },
    {
      "url": "turn:global.turn.twilio.com:3478?transport=udp",
      "username": "72ba795b16e54d1d280649db774a72cbd457f45a747f9be8aadd225dae16bc7a",
      "urls": "turn:global.turn.twilio.com:3478?transport=udp",
      "credential": "kmAVKEQnf5dNqUeTxqV3zAgLiTJZXrtateM/8cFwROU="
    },
    {
      "url": "turn:global.turn.twilio.com:3478?transport=tcp",
      "username": "72ba795b16e54d1d280649db774a72cbd457f45a747f9be8aadd225dae16bc7a",
      "urls": "turn:global.turn.twilio.com:3478?transport=tcp",
      "credential": "kmAVKEQnf5dNqUeTxqV3zAgLiTJZXrtateM/8cFwROU="
    },
    {
      "url": "turn:global.turn.twilio.com:443?transport=tcp",
      "username": "72ba795b16e54d1d280649db774a72cbd457f45a747f9be8aadd225dae16bc7a",
      "urls": "turn:global.turn.twilio.com:443?transport=tcp",
      "credential": "kmAVKEQnf5dNqUeTxqV3zAgLiTJZXrtateM/8cFwROU="
    },
    {
      "url": "stun:stun.relay.metered.ca:80",
      "urls": "stun:stun.relay.metered.ca:80",
    },
    {
      "url": "turn:global.relay.metered.ca:80",
      "urls": "turn:global.relay.metered.ca:80",
      "username": "3b248090b0c8bb1b18704ca9",
      "credential": "3YG/B1kIs8fOwRUf",
    },
    {
      "url": "turn:global.relay.metered.ca:80?transport=tcp",
      "urls": "turn:global.relay.metered.ca:80?transport=tcp",
      "username": "3b248090b0c8bb1b18704ca9",
      "credential": "3YG/B1kIs8fOwRUf",
    },
    {
      "url": "turn:global.relay.metered.ca:443",
      "urls": "turn:global.relay.metered.ca:443",
      "username": "3b248090b0c8bb1b18704ca9",
      "credential": "3YG/B1kIs8fOwRUf",
    },
    {
      "url": "turns:global.relay.metered.ca:443?transport=tcp",
      "urls": "turns:global.relay.metered.ca:443?transport=tcp",
      "username": "3b248090b0c8bb1b18704ca9",
      "credential": "3YG/B1kIs8fOwRUf",
    }
  ],
}

def convert_flv_to_mp4(input_file: str, output_file: str):
    try:
        stream = ffmpeg.input(input_file)
        stream = ffmpeg.output(stream, output_file, codec='copy')
        ffmpeg.run(stream, overwrite_output=True)
        print(f"Conversion successful: {output_file}")
    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode()}")
# Set the page config to use the full width of the screen
st.set_page_config(layout="wide")
# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #1a1a1a; /* Dark background */
        color: #f0f0f0; /* Light text */
    }
    .title {
        font-size: 2.5em;
        color: #4CAF50; /* Green title */
        text-align: center;
        margin-bottom: 0.5em;
    }
    .header {
        font-size: 1.75em;
        color: #2196F3; /* Blue header */
        text-align: center;
        margin-bottom: 1em;
        border-bottom: 2px solid #444; /* Dark gray border */
        padding-bottom: 0.5em;
    }
    .subheader {
        font-size: 1.5em;
        color: #FF5722; /* Orange subheader */
        text-align: center;
        margin-bottom: 0.5em;
        border-bottom: 1px solid #666; /* Dark gray border */
        padding-bottom: 0.3em;
    }
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    .recognized-text {
        font-size: 1.5em;
        color: #f0f0f0; /* Light gray recognized text */
        text-align: center;
        margin-top: 1em;
        padding: 10px;
        border-radius: 10px;
        background-color: #333; /* Dark gray background */
        box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.1); /* White shadow for contrast */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with logo and information
with st.sidebar:
    st.image("lips.png", width=100)
    st.markdown('<div class="title">قارئ الشفاه</div>', unsafe_allow_html=True)
    st.markdown(
        """
        هذا التطبيق يستخدم التعرف على الشفاه لتحويل مقاطع الفيديو إلى نص. 
        قم بتحميل ملف فيديو MP4، وسيقوم التطبيق بمعالجة الفيديو للتعرف على النص المنطوق وعرضه.
        """,
        unsafe_allow_html=True
    )

# Main content
st.markdown('<div class="header">قم بتحميل ملف الفيديو الخاص بك</div>', unsafe_allow_html=True)

# WebtRTC
RECORD_DIR = Path("./records")
RECORD_DIR.mkdir(exist_ok=True)

if "prefix" not in st.session_state:
        st.session_state["prefix"] = str(uuid.uuid4())
prefix = st.session_state["prefix"]
in_file = RECORD_DIR / f"{prefix}_input.flv"

def in_recorder_factory() -> MediaRecorder:
        return MediaRecorder(
            str(in_file), format="flv"
        )

webrtc_streamer(
        key="record",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=peer_connection_config,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        in_recorder_factory=in_recorder_factory,
    )

if in_file.exists():
    with in_file.open("rb") as f:
        st.download_button(
            "Download the recorded video", f, "input.flv"
        )
    convert_flv_to_mp4(in_file,"output.mp4")

# Create a file uploader
uploaded_file = st.file_uploader("", type=["mp4"])

if uploaded_file is not None or in_file.exists():
    col1, col2 = st.columns(2)

    with col1:
        # Display the uploaded video
        st.markdown('<div class="subheader">ملف الفيديو المحمل</div>', unsafe_allow_html=True)
        video_file = 'output.mp4'
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                video_file = temp_file.name
        st.video(video_file)

    with col2:
        # Display the processed video
        st.markdown('<div class="subheader">معالجة الفيديو</div>', unsafe_allow_html=True)
        
        # Show a progress bar and message
        progress_message = st.empty()
        progress_bar = st.empty()
        
        progress_message.markdown('<div class="subheader">جارٍ معالجة الفيديو...</div>', unsafe_allow_html=True)
        progress_bar = progress_bar.progress(0)
        
        for percent_complete in range(100):
            time.sleep(0.05)
            progress_bar.progress(percent_complete + 1)

        # Clear the progress message and bar
        progress_message.empty()
        progress_bar.empty()
        
        # Show the processed video
        dst_filename = preprocess_video(src_filename=video_file, dst_filename="roi.mp4")
        st.video("roi.mp4")

        # Show a spinner while processing
        with st.spinner('جارٍ تشغيل التعرف على الكلام...'):
            pipeline = pipeline.build_pipeline()
            result = pipeline("roi.mp4", 3)

        # Show the result
        st.success("اكتملت المعالجة!")
        st.markdown('<div class="subheader">النص المنطوق</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="recognized-text">لقد قلت:{result}</div>', unsafe_allow_html=True)

else:
    st.warning("يرجى تحميل ملف فيديو MP4.")
