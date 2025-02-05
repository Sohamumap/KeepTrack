import streamlit as st
import pandas as pd
import cv2
import os
import google.generativeai as genai
from PIL import Image
import time
import tempfile
import io
import pyttsx3
import mediapipe as mp
import numpy as np
import playsound
import threading

# Configure page
st.set_page_config(
    page_title="KeepTrack - Video Item Inventory & Fall Alert",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper to parse CSV
def parse_csv_with_fixed_columns(csv_string, expected_columns):
    import csv, io
    records = []
    f = io.StringIO(csv_string)
    reader = csv.reader(f)
    try:
        header = next(reader)
    except StopIteration:
        return []
    for row in reader:
        try:
            if len(row) < expected_columns:
                row += ["NA"] * (expected_columns - len(row))
            elif len(row) > expected_columns:
                row = row[:expected_columns]
            record = dict(zip(header, row))
            records.append(record)
        except Exception as e:
            print(f"Error parsing CSV: {e}")
            continue
    return records

# Initialize session state
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'inventory_df' not in st.session_state:
    st.session_state.inventory_df = None
if 'frames' not in st.session_state:
    st.session_state.frames = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Title and description
st.title("KeepTrack - Video Item Inventory & Fall Alert")
st.markdown("""
This app helps you create an inventory of items from a video walkthrough of your space and features Fall Detection for safety monitoring.
""")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Gemini API Key", type="password")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.success("API key configured successfully!")
        except Exception as e:
            st.error(f"Error configuring API key: {str(e)}")
    st.markdown("---")
    st.markdown("### Instructions")

# Model and TTS caching
@st.cache_resource
def load_gemini_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro')

@st.cache_resource
def load_tts_engine():
    return pyttsx3.init()

tts_engine = load_tts_engine()

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload & Process", "Inventory", "Frames", "Find Item", "Fall Alert"])

# Upload & Process Tab
with tab1:
    st.header("Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file:
        process_button = st.button("Process Video")
        if process_button:
            with st.spinner("Processing video..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name

                    video_file = genai.upload_file(path=temp_path)
                    st.session_state.processed_video = video_file

                    model = load_gemini_model(api_key)
                    prompt = """Your CSV generation prompt here..."""
                    
                    response = model.generate_content([video_file, prompt])
                    csv_data = response.text.split("<csv>")[1].split("</csv>")[0].strip() if "<csv>" in response.text else None
                    
                    if csv_data:
                        records = parse_csv_with_fixed_columns(csv_data, 7)
                        df = pd.DataFrame(records)
                        st.session_state.inventory_df = df

                        cap = cv2.VideoCapture(temp_path)
                        frames = []
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frames.append(frame)
                        cap.release()
                        st.session_state.frames = frames
                        st.session_state.processing_complete = True
                        st.success("Processing complete!")

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")

# Inventory Tab
with tab2:
    if st.session_state.inventory_df is not None:
        st.header("Inventory")
        df = st.session_state.inventory_df
        edited_df = st.data_editor(df)
        
        # Add download buttons and statistics here...

# Frames Tab
with tab3:
    if st.session_state.frames:
        st.header("Video Frames")
        frame_selection = st.slider("Select frame", 0, len(st.session_state.frames)-1, 0)
        st.image(cv2.cvtColor(st.session_state.frames[frame_selection], cv2.COLOR_BGR2RGB))

# Find Item Tab (Corrected Section)
with tab4:
    st.header("Find Item Location")
    item_to_find = st.text_input("Item Name:")
    find_button = st.button("Find Location")

    if find_button:
        if not item_to_find:
            st.warning("Please enter an item name")
        elif not st.session_state.processed_video:
            st.warning("Please process a video first")
        else:
            try:
                with st.spinner("Searching..."):
                    model = load_gemini_model(api_key)
                    prompt = f"Find {item_to_find} in the video..."
                    response = model.generate_content([st.session_state.processed_video, prompt])
                    result = response.text
                    st.success(result)
                    tts_engine.say(result)
                    tts_engine.runAndWait()
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Fall Alert Tab
with tab5:
    st.header("Fall Detection")
    start_fall = st.button("Start Webcam Monitoring")
    
    if start_fall:
        st.markdown("### Live Fall Detection")
        # Add fall detection implementation here...

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Google's Gemini AI")
