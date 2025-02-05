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
# import playsound  # Removed playsound import for Streamlit Cloud - Use for local if needed
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
            record = dict(zip(header,row))
            records.append(record)
        except Exception as e:
             print(f"Error parsing CSV, row: {row}, skipping this line: {e}")
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
This app helps you create an inventory of items from a video walkthrough of your space **and now also features Fall Detection for safety monitoring.**

### Inventory Features:
- 📹 Process video content using AI for item inventory
- 📝 Generate detailed inventory with item descriptions
- 💰 Estimate item values
- 📊 View inventory statistics
- ⬇️ Export data to CSV
- 🗣️ Text-to-speech for item location
- 🔍 Find misplaced items in your video

### New Fall Alert Feature:
- 🚨 Real-time Fall Detection using webcam
- 🔔 Visual alerts upon fall detection (Sound alert removed for web deployment)
- 🛡️ Proactive safety monitoring for homes and care environments

**Note:** Sound alerts for Fall Detection have been removed for better compatibility with web deployment environments. Fall detection will now rely on visual cues on the video frame. If running locally, you can uncomment `import playsound` and related code in the Fall Alert tab to enable sound alerts.
""")

# Sidebar for API key and configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Gemini API Key", type="password", help="Get your API key from Google AI Studio (required for Inventory features)")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.success("API key configured successfully!")
        except Exception as e:
            st.error(f"Error configuring API key: {str(e)}")

    st.markdown("---")
    st.markdown("""
    ### Instructions
    **For Inventory Features:**
    1. Enter your Gemini API key
    2. Upload a video file
    3. Click "Process Video"
    4. View results in the Inventory and Frames tabs
    5. Use "Find Item" tab to locate items (with voice output)

    **For Fall Alert Feature:**
    1. Go to the "Fall Alert" tab.
    2. Ensure your webcam is accessible.
    3. Click "Start Fall Detection" to begin monitoring.
    """)

# Cached Gemini Model
@st.cache_resource
def load_gemini_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro')

# Cached TTS Engine
@st.cache_resource
def load_tts_engine():
    return pyttsx3.init()

# Initialize TTS engine
tts_engine = load_tts_engine()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload & Process", "Inventory", "Frames", "Find Item", "Fall Alert"])

# Tab 1: Upload & Process
with tab1:
    st.header("Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file (MP4, MOV, or AVI)",
        type=['mp4', 'mov', 'avi'],
        help="Maximum file size: 200MB (Required for Inventory Features)"
    )

    if uploaded_file:
        file_details = st.expander("Video Details", expanded=False)
        with file_details:
            file_info = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "File type": uploaded_file.type
            }
            st.json(file_info)

        process_button = st.button(
            "Process Video (For Inventory)",
            help="Click to start AI analysis of the video for item inventory",
            use_container_width=True
        )

        if process_button:
            try:
                with st.spinner("Processing video for inventory..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name

                    try:
                        video_file = genai.upload_file(path=temp_path)
                        st.session_state.processed_video = video_file
                        model = load_gemini_model(api_key)

                        videoStatus = genai.get_file(video_file.name)
                        while videoStatus.state.name == 'PROCESSING':
                           time.sleep(1);
                           videoStatus = genai.get_file(video_file.name);

                        prompt = """You are an expert home inventory logger and home insurance reviewer... (rest of your prompt)""" # Put your Gemini prompt here

                        progress_bar = st.progress(0)
                        progress_text = st.empty()

                        progress_text.text("Analyzing video for inventory...")
                        progress_bar.progress(25)
                        response = model.generate_content([video_file, prompt])

                        progress_text.text("Extracting inventory data...")
                        progress_bar.progress(50)
                        print(f"Raw Gemini Response (Inventory Tab - Upload & Process): {response.text}")

                        csv_data = None
                        try:
                            if "<csv>" in response.text and "</csv>" in response.text:
                                csv_data = response.text.split("<csv>")[1].split("</csv>")[0].strip()
                            else:
                                st.error("Error: Could not find CSV data in the AI response.")
                                csv_data = None
                        except IndexError:
                            st.error("Error processing AI response: CSV data extraction failed.")
                            csv_data = None

                        print(f"Cleaned CSV Data (Inventory Tab - Upload & Process): {csv_data}")

                        if csv_data:
                            progress_text.text("Processing frames for inventory...")
                            progress_bar.progress(75)

                            try:
                                records = parse_csv_with_fixed_columns(csv_data, 7)
                                df = pd.DataFrame(records)
                                st.session_state.inventory_df = df
                            except Exception as csv_parse_error:
                                st.error(f"Error parsing CSV data: {csv_parse_error}")
                                st.error("CSV data from AI might be malformed.")
                                st.stop()

                            cap = cv2.VideoCapture(temp_path)
                            frames = []
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                frames.append(frame)
                            cap.release()
                            st.session_state.frames = frames

                            progress_bar.progress(100)
                            progress_text.text("Inventory processing complete!")
                            st.session_state.processing_complete = True
                            st.success("Video processed successfully for inventory!")

                    except Exception as e:
                        st.error(f"Error processing video for inventory: {str(e)}")
                        st.error("Please make sure your API key is valid and try again.")

                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

            except Exception as e:
                st.error(f"Error handling file: {str(e)}")

# Tab 2: Inventory
with tab2:
    st.header("Inventory")
    if st.session_state.inventory_df is not None:
        search_term = st.text_input("Search inventory", help="Filter items")

        df = st.session_state.inventory_df
        if search_term:
            mask = df.apply(lambda x: x.astype(str).str.contains(search_term, case=False)).any(axis=1)
            df = df[mask]

        try:
            df['value'] = df['value'].astype(str).str.strip()
            df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)
        except Exception as conversion_error:
            st.warning(f"Error converting 'value' column: {conversion_error}")
            df['value'] = 0

        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "item_number": st.column_config.TextColumn("Item No."),
                "item_name": st.column_config.TextColumn("Item Name"),
                "category": st.column_config.TextColumn("Category"),
                "description": st.column_config.TextColumn("Description", width="large"),
                "value": st.column_config.NumberColumn(
                    "Value (₹)",
                    help="Estimated value in Indian Rupees",
                    min_value=0,
                    format="₹%d"
                ),
                "condition": st.column_config.SelectboxColumn("Condition", options=["new", "excellent", "good", "fair", "poor"]),
                "room": st.column_config.TextColumn("Room")
            }
        )

        col1, col2 = st.columns(2)
        with col1:
            csv = edited_df.to_csv(index=False)
            st.download_button("Download CSV", csv, "inventory.csv", "text/csv", key='download-csv')
        with col2:
            excel_buffer = pd.ExcelWriter('inventory.xlsx', engine='xlsxwriter')
            edited_df.to_excel(excel_buffer, index=False)
            excel_buffer.close()
            with open('inventory.xlsx', 'rb') as f:
                excel_data = f.read()
            st.download_button("Download Excel", excel_data, "inventory.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key='download-excel')

        st.subheader("Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Items", len(edited_df))
        with col2:
            st.metric("Total Value", f"₹{edited_df['value'].sum():,.2f}")
        with col3:
            st.metric("Unique Categories", len(edited_df['category'].unique()))
        with col4:
            st.metric("Rooms Covered", len(edited_df['room'].unique()))

        st.subheader("Analysis")
        tab_stats1, tab_stats2 = st.tabs(["Categories", "Values by Room"])
        with tab_stats1:
            st.subheader("Category Breakdown")
            cat_counts = edited_df['category'].value_counts()
            st.bar_chart(cat_counts)
        with tab_stats2:
            st.subheader("Total Value by Room")
            room_values = edited_df.groupby('room')['value'].sum().sort_values(ascending=True)
            st.bar_chart(room_values)

# Tab 3: Frames
with tab3:
    st.header("Video Frames")
    if st.session_state.frames:
        try:
            total_frames = len(st.session_state.frames)
            frame_selection = st.slider("Select frame", 0, total_frames-1, 0)
            frame = st.session_state.frames[frame_selection]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption=f"Frame {frame_selection + 1} of {total_frames}")

            st.subheader("Frame Grid")
            cols = st.columns(3)
            for idx, frame in enumerate(st.session_state.frames[:9]):
                with cols[idx % 3]:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"Frame {idx+1}")
        except Exception as e:
            st.error(f"Error processing frames: {e}")
    else:
      st.text("No frames were extracted from video.")

# Tab 4: Find Item
with tab4:
    st.header("Find Item Location")
    st.markdown("Enter the name of the item you are looking for...")

    item_to_find = st.text_input("Item Name (e.g., spectacles, keys):", "")
    find_button = st.button("Find Location", use_container_width=True)
    location_output = st.empty()

    engine = load_tts_engine()

    if find_button and item_to_find and st.session_state.processed_video:
        try:
            with st.spinner(f"Searching for '{item_to_find}'..."):
                video_file = st.session_state.processed_video

                location_prompt = f"""You are an expert in analyzing home videos to find misplaced items... (rest of your prompt)""" # Put your Gemini prompt here

                model = load_gemini_model(api_key)
                print(f"Sending Find Item request to Gemini API for item: {item_to_find}")
                location_response = model.generate_content([video_file, location_prompt])
                print(f"Raw Gemini Response (Find Item Tab): {location_response}")
                print(f"Raw Gemini Response Text (Find Item Tab): {location_response.text}")
                location_text = location_response.text

                location_output.success(f"Location of '{item_to_find}':")
                location_output.write(location_text)

                try:
                    engine.say(location_text)
                    engine.runAndWait()
                except Exception as tts_error:
                    st.error(f"Text-to-speech error: {tts_error}")

        except Exception as e:
            location_output.error(f"Error finding item location: {str(e)}")
            location_output.error(f"Please make sure your API key is valid and try again. Error details: {e}")

    elif find_button and not st.session_state.processed_video:
        location_output.warning("Please upload and process a video first...")

# Tab 5: Fall Alert
with tab5:
    st.header("Fall Alert System")
    st.markdown("Click 'Start Fall Detection' to activate real-time fall monitoring using your webcam.")

    fall_detection_start = st.button("Start Fall Detection", use_container_width=True)
    fall_detection_display = st.empty()

    if fall_detection_start:
        st.markdown("### Fall Detection Activated")
        st.markdown("Monitoring for falls... Press 'q' on the video frame to stop.")

        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose()
        cap = cv2.VideoCapture(0)

        # --- Fall Detection Parameters --- (same as before)
        GROUND_SMOOTHING = 0.9
        FALL_THRESHOLD = 0.35
        ANGLE_THRESHOLD = 35
        FALL_DURATION = 1.5
        STILL_FALL_DURATION = 5
        NO_MOVEMENT_DURATION = 3
        ALERT_SOUND = "alert.mp3" # Ensure 'alert.mp3' is in the same directory
        SITTING_ANGLE_THRESHOLD = 60

        # --- Tracking Variables --- (same as before)
        previous_ground_y = None
        fall_start_time = None
        still_fall_start_time = None
        no_movement_start_time = None
        fall_detected = False
        previous_torso_y = None

        # Removed play_alert_sound function for Streamlit Cloud - uncomment for local use with sound
        # def play_alert_sound():
        #     """Plays an alert sound in a separate thread."""
        #     try:
        #         playsound.playsound(ALERT_SOUND)
        #     except Exception as e:
        #         print("Sound error:", e)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Error reading video feed. Please check your webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)
            h, w, _ = frame.shape

            ground_candidates = []
            keypoints_on_ground = 0
            total_keypoints = 0
            torso_angle = None
            torso_speed = 0

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark

                for i in [mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value]:
                    ground_candidates.append(int(landmarks[i].y * h))

                if ground_candidates:
                    estimated_ground_y = int(np.median(ground_candidates))
                    if previous_ground_y is None:
                        previous_ground_y = estimated_ground_y
                    else:
                        previous_ground_y = int(GROUND_SMOOTHING * previous_ground_y + (1 - GROUND_SMOOTHING) * estimated_ground_y)

                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

                dx = (right_shoulder.x - left_shoulder.x) * w
                dy = (right_shoulder.y - left_shoulder.y) * h
                if dx != 0:
                    torso_angle = np.degrees(np.arctan(abs(dy / dx)))

                for i in range(len(landmarks)):
                    x, y = int(landmarks[i].x * w), int(landmarks[i].y * h)
                    total_keypoints += 1
                    if y >= previous_ground_y:
                        keypoints_on_ground += 1
                        cv2.circle(frame, (x, y), 7, (0, 0, 255), -1)
                    else:
                        cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)

                is_lying_down = torso_angle is not None and torso_angle < ANGLE_THRESHOLD
                is_sitting = torso_angle is not None and torso_angle > SITTING_ANGLE_THRESHOLD

                if keypoints_on_ground / total_keypoints > FALL_THRESHOLD and is_lying_down and not is_sitting:
                    if not fall_detected:
                        fall_start_time = time.time()
                        fall_detected = True
                    elif time.time() - fall_start_time >= FALL_DURATION:
                        cv2.putText(frame, "FALL DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        # threading.Thread(target=play_alert_sound, daemon=True).start() # Removed sound alert for Streamlit Cloud

                elif fall_detected and is_lying_down:
                    if still_fall_start_time is None:
                        still_fall_start_time = time.time()
                    elif time.time() - still_fall_start_time >= STILL_FALL_DURATION:
                        cv2.putText(frame, "PERSON STILL LYING DOWN!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        # threading.Thread(target=play_alert_sound, daemon=True).start() # Removed sound alert for Streamlit Cloud

                    if no_movement_start_time is None:
                        no_movement_start_time = time.time()
                    elif time.time() - no_movement_start_time >= NO_MOVEMENT_DURATION:
                        cv2.putText(frame, "NO MOVEMENT DETECTED!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        # threading.Thread(target=play_alert_sound, daemon=True).start() # Removed sound alert for Streamlit Cloud
                else:
                    fall_detected = False
                    fall_start_time = None
                    still_fall_start_time = None
                    no_movement_start_time = None

            if previous_ground_y is not None:
                cv2.line(frame, (0, previous_ground_y), (w, previous_ground_y), (255, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fall_detection_display.image(frame_rgb, channels="RGB")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        pose.close()
        st.success("Fall Detection Stopped.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Made with ❤️ using Streamlit and Google's Gemini AI</p>
</div>
""", unsafe_allow_html=True)
