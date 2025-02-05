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
            records.append(dict(zip(header, row)))
        except Exception as e:
            print(f"Error parsing CSV: {e}")
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
This app helps you create an inventory of items from a video walkthrough and monitor falls.
""")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.success("API key configured!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Cached resources
@st.cache_resource
def load_gemini_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro')

@st.cache_resource
def load_tts_engine():
    return pyttsx3.init()

tts_engine = load_tts_engine()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload & Process", "Inventory", "Frames", "Find Item", "Fall Alert"])

# Upload & Process Tab
with tab1:
    st.header("Upload Video")
    uploaded_file = st.file_uploader("Choose video file", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file and st.button("Process Video"):
        with st.spinner("Processing..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name

                video_file = genai.upload_file(path=temp_path)
                st.session_state.processed_video = video_file
                
                model = load_gemini_model(api_key)
                prompt = """Your inventory analysis prompt here..."""
                
                response = model.generate_content([video_file, prompt])
                
                if "<csv>" in response.text:
                    csv_data = response.text.split("<csv>")[1].split("</csv>")[0].strip()
                    records = parse_csv_with_fixed_columns(csv_data, 7)
                    st.session_state.inventory_df = pd.DataFrame(records)
                    
                    cap = cv2.VideoCapture(temp_path)
                    st.session_state.frames = [frame for _ in 
                        iter(lambda: cap.read()[1], None) for frame in _]
                    cap.release()
                    
                    st.success("Processing complete!")
                os.remove(temp_path)
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Inventory Tab
with tab2:
    if st.session_state.inventory_df is not None:
        st.data_editor(st.session_state.inventory_df)
        # Add download and analysis features here

# Fall Alert Tab (Key Fix)
with tab5:
    st.header("Fall Detection")
    
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False

    start_col, stop_col = st.columns(2)
    with start_col:
        if st.button("Start Detection"):
            st.session_state.detection_active = True
    with stop_col:
        if st.button("Stop Detection"):
            st.session_state.detection_active = False

    video_placeholder = st.empty()

    if st.session_state.detection_active:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        cap = cv2.VideoCapture(0)

        while st.session_state.detection_active and cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Pose detection logic here
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            # Simplified fall detection display
            if results.pose_landmarks:
                cv2.putText(frame, "Monitoring...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            video_placeholder.image(frame, channels="BGR")

        cap.release()
        pose.close()

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")

# Error handling improvements
def main():
    try:
        # Your app code
        pass
    except Exception as e:
        st.error(f"Critical Error: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()
