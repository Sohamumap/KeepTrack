import streamlit as st
import pandas as pd
import cv2
import os
import google.generativeai as genai
from PIL import Image
import time
import tempfile
import io
import pyttsx3  # Import the text-to-speech library
import mediapipe as mp
import numpy as np
# import playsound # Removed playsound import
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
    """
    Parses a CSV string, handling irregularities, and returns a list of records.
    """
    import csv, io
    records = []

    f = io.StringIO(csv_string)
    reader = csv.reader(f)

    try:
        header = next(reader)
    except StopIteration:
        return []  # Handle empty CSV string

    for row in reader:
      # Handle incomplete rows (add NA values) or extra columns by trimming the data
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

# Initialize session state (same as before)
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'inventory_df' not in st.session_state:
    st.session_state.inventory_df = None
if 'frames' not in st.session_state:
    st.session_state.frames = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Title and description (updated to include Fall Alert)
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

**Note:** Sound alerts for Fall Detection have been removed for better compatibility with web deployment environments. Fall detection will now rely on visual cues on the video frame.
""")

# Sidebar for API key and configuration (same as before)
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

# Cached Gemini Model (using st.cache_resource) (same as before)
@st.cache_resource
def load_gemini_model(api_key):
    genai.configure(api_key=api_key) # Configure API here as well, just to be safe
    return genai.GenerativeModel('gemini-1.5-pro')

# Cached TTS Engine (using st.cache_resource) (same as before)
@st.cache_resource
def load_tts_engine():
    return pyttsx3.init()

# Initialize TTS engine globally using the cached function (same as before)
tts_engine = load_tts_engine()


# Main content tabs (Added "Fall Alert" tab)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload & Process", "Inventory", "Frames", "Find Item", "Fall Alert"]) # Added tab5 "Fall Alert"

# Inventory Tabs (tab1, tab2, tab3, tab4) - same as before -  no changes here, just moved to different tab index
with tab1: # Upload & Process Tab - Same as before
    st.header("Upload Video")

    # File uploader with clear instructions
    uploaded_file = st.file_uploader(
        "Choose a video file (MP4, MOV, or AVI)",
        type=['mp4', 'mov', 'avi'],
        help="Maximum file size: 200MB (Required for Inventory Features)"
    )

    if uploaded_file:
        # Show video details
        file_details = st.expander("Video Details", expanded=False)
        with file_details:
            file_info = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "File type": uploaded_file.type
            }
            st.json(file_info)

        # Create process button
        process_button = st.button(
            "Process Video (For Inventory)",
            help="Click to start AI analysis of the video for item inventory",
            use_container_width=True
        )

        if process_button:
            # ... (rest of the Process Video button logic - same as before)
            try:
                with st.spinner("Processing video for inventory..."): # Updated spinner message
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name

                    try:
                        # Upload to Gemini
                        video_file = genai.upload_file(path=temp_path)
                        st.session_state.processed_video = video_file # Save the video_file to session state

                        # Get cached Gemini model
                        model = load_gemini_model(api_key) # Pass api_key here

                         # Wait for video to upload successfully
                        videoStatus = genai.get_file(video_file.name)
                        while videoStatus.state.name == 'PROCESSING':
                           time.sleep(1);
                           videoStatus = genai.get_file(video_file.name);


                        # Enhanced prompt for better inventory details WITH VALUE ESTIMATION
                        prompt = """You are an expert home inventory logger and home insurance reviewer. Your task is to catalog household items from a smartphone recorded video for home and contents insurance purposes.
                                Your goal is to create a comprehensive and accurate inventory of all items visible or mentioned in the attached video. **Crucially, you must also estimate a reasonable market value in US dollars for each item.**

                                I want you to create a CSV file with the following columns/schema.
                                item_number,item_name,category,description,value,condition,room

                                Follow these instructions:
                                1. Create a single line in the CSV for every unique item in the video.
                                2. For multiple identical items (e.g. 3 of the same chair or 4 of the same speaker), update the number_of_items field instead of creating separate entries. Use integer values to indicate their counts. Count as best you can.
                                3. Use simple but understandable descriptions for each item.
                                4. If an item is similar to another item in the inventory, note this in the is_similar_to field.
                                5. Make sure there are no blank fields, if something needs to be blank, fill it with NA.
                                 6. Ensure every column has a value, do not miss a column.
                                **7. For the 'value' column, estimate a reasonable resale or market value in US dollars for each item. Provide a numerical value only, do not include currency symbols or commas.** If you cannot reasonably estimate a value, put 0.

                                CSV Formatting Instructions:
                                    - Use commas only to separate fields rather than inside fileds.
                                    - Do not use commas in the item_description field.
                                    - For fields that inherently include commas, enclose the entire value in double quotes (`\"`).
                                    - If double quotes appear within a field value, escape them by doubling them (`\"\"`). For example, `John \"JJ\" Smith` should be written as `\"John \"\"JJ\"\" Smith\"`.\
                                    - Do not use any unescaped double quotes or other special characters that may cause the CSV to be invalid.
                                    - Never use commas in prices. For example, $1500.0 = good, $1,500.0 = bad.
                                     - Return all columns in the order they are presented in.

                                Return the full and valid CSV within <csv> tags so it can be easily parsed out.
                                  Include a note to yourself in <other_details_to_note> if you think there are more items to continue doing in the video, this should be inline with the <item_logging_status> as well as the next steps to improve upon the record keeping.
                        """

                        # Generate response with progress bar
                        progress_bar = st.progress(0)
                        progress_text = st.empty()

                        progress_text.text("Analyzing video for inventory...") # Updated progress text
                        progress_bar.progress(25)

                        response = model.generate_content([video_file, prompt])

                        progress_text.text("Extracting inventory data...")
                        progress_bar.progress(50)

                        # Debugging: Print the full response text
                        print(f"Raw Gemini Response (Inventory Tab - Upload & Process): {response.text}")

                        # Extract CSV data
                        csv_data = None
                        try:
                            if "<csv>" in response.text and "</csv>" in response.text: # Changed delimiters here
                                csv_data = response.text.split("<csv>")[1].split("</csv>")[0].strip() # Changed delimiters here
                            else:
                                st.error("Error: Could not find CSV data in the AI response. Please check the API key and try again.")
                                csv_data = None # Ensure csv_data is None to prevent further processing
                        except IndexError:
                            st.error("Error processing AI response: CSV data extraction failed. The response format might be incorrect.")
                            csv_data = None

                        print(f"Cleaned CSV Data (Inventory Tab - Upload & Process): {csv_data}")

                        if csv_data: # Only proceed if csv_data is successfully extracted
                            progress_text.text("Processing frames for inventory...") # Updated progress text
                            progress_bar.progress(75)

                            # Convert to DataFrame
                            try:
                                records = parse_csv_with_fixed_columns(csv_data, 7)
                                df = pd.DataFrame(records)
                                st.session_state.inventory_df = df
                            except Exception as csv_parse_error:
                                st.error(f"Error parsing CSV data: {csv_parse_error}")
                                st.error("The CSV data from the AI might be malformed. Please check the API key and try again.")
                                st.stop() # Stop further processing if CSV parsing fails

                            # Extract frames
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
                            progress_text.text("Inventory processing complete!") # Updated progress text

                            st.session_state.processing_complete = True
                            st.success("Video processed successfully for inventory! Check the Inventory and Frames tabs for results.") # Updated success message


                    except Exception as e:
                        st.error(f"Error processing video for inventory: {str(e)}") # Updated error message
                        st.error("Please make sure your API key is valid and try again.")

                    finally:
                        # Cleanup
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

            except Exception as e:
                st.error(f"Error handling file: {str(e)}")

with tab2: # Inventory Tab - Same as before
    st.header("Inventory")
    if st.session_state.inventory_df is not None:
        # Add search/filter functionality
        search_term = st.text_input("Search inventory", help="Filter items by name, category, or room")

        df = st.session_state.inventory_df
        if search_term:
            mask = df.apply(lambda x: x.astype(str).str.contains(search_term, case=False)).any(axis=1)
            df = df[mask]

        # **Debug: Print DataFrame info before data_editor**
        print("DataFrame before data_editor in Inventory Tab:")
        print(df.info())
        print(df.head())


        # **Robustly Convert 'value' column to numeric**
        try:
            # 1. Ensure 'value' column is string type before using str accessor
            df['value'] = df['value'].astype(str)

            # 2. Strip whitespace from the 'value' column first
            df['value'] = df['value'].str.strip()

            # 3. Try converting to numeric, handling errors by coercing to NaN
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

            # 4. Fill NaN values with 0 (or you could handle them differently)
            df['value'] = df['value'].fillna(0)

        except Exception as conversion_error:
            st.warning(f"Error converting 'value' column to numbers: {conversion_error}. Ensure 'value' column contains numeric data. Error details: {conversion_error}") # More detailed warning
            df['value'] = 0  # If conversion completely fails, set to 0

        # Display inventory with editing capability
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "condition": st.column_config.SelectboxColumn( # Moved 'condition' column first for testing
                    "Condition",
                    options=["new", "excellent", "good", "fair", "poor"] # Simplified options for testing
                ),
                "value": st.column_config.NumberColumn(
                    "Value ($)",
                    help="Estimated value in USD",
                    min_value=0,
                    format="$%d"
                ),

            }
        )

        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv = edited_df.to_csv(index=False)
            st.download_button(
                "Download Inventory (CSV)",
                csv,
                "inventory.csv",
                "text/csv",
                key='download-csv',
                help="Download the inventory as a CSV file"
            )

        with col2:
            # Export to Excel
            excel_buffer = pd.ExcelWriter('inventory.xlsx', engine='xlsxwriter')
            edited_df.to_excel(excel_buffer, index=False)
            excel_buffer.close()

            with open('inventory.xlsx', 'rb') as f:
                excel_data = f.read()

            st.download_button(
                "Download Inventory (Excel)",
                excel_data,
                "inventory.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='download-excel',
                help="Download the inventory as an Excel file"
            )

        # Summary statistics
        st.subheader("Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Items", len(edited_df))
        with col2:
            # **Now it should sum as numbers**
            st.metric("Total Value", f"${edited_df['value'].sum():,.2f}")
        with col3:
            st.metric("Unique Categories", len(edited_df['category'].unique()))
        with col4:
            st.metric("Rooms Covered", len(edited_df['room'].unique()))

        # Visualizations
        st.subheader("Analysis")
        tab_stats1, tab_stats2 = st.tabs(["Categories", "Values by Room"])

        with tab_stats1:
            # Category breakdown
            st.subheader("Category Breakdown")
            cat_counts = edited_df['category'].value_counts()
            st.bar_chart(cat_counts)

        with tab_stats2:
            # Value by room
            st.subheader("Total Value by Room")
            room_values = edited_df.groupby('room')['value'].sum().sort_values(ascending=True)
            st.bar_chart(room_values)

with tab3: # Frames Tab - Same as before
    st.header("Video Frames")
    if st.session_state.frames:
        try:
            # Add frame selection slider
            total_frames = len(st.session_state.frames)
            frame_selection = st.slider("Select frame", 0, total_frames-1, 0)

            # Display selected frame
            frame = st.session_state.frames[frame_selection]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption=f"Frame {frame_selection + 1} of {total_frames}")

            # Display grid of frames
            st.subheader("Frame Grid")
            cols = st.columns(3)
            for idx, frame in enumerate(st.session_state.frames[:9]):  # Show first 9 frames
                with cols[idx % 3]:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"Frame {idx+1}")
        except Exception as e:
            st.error(f"Error processing frames: {e}")
    else:
      st.text("No frames were extracted from video.")

with tab4: # Find Item Tab - Same as before
    st.header("Find Item Location")
    st.markdown("Enter the name of the item you are looking for, and I will try to find it in the video.")

    item_to_find = st.text_input("Item Name (e.g., spectacles, keys):", "")
    find_button = st.button("Find Location", use_container_width=True)
    location_output = st.empty() # Create an empty container to display location text

    # Get the already initialized TTS engine from the cached function
    engine = load_tts_engine() # Get cached TTS engine!

    if find_button and item_to_find and st.session_state.processed_video:
        try:
            with st.spinner(f"Searching for '{item_to_find}'..."):
                video_file = st.session_state.processed_video # Assuming you stored the video_file in session state

                # --- NEW PROMPT FOR FINDING ITEM LOCATION ---
                location_prompt = f"""
                    You are an expert in analyzing home videos to find misplaced items.
                    I will provide you with a video of a house walkthrough.
                    The user is looking for their **{item_to_find}**.

                    Please analyze the video and describe the location of the **{item_to_find}** if you can find it.
                    Be specific about its location relative to other objects in the scene.
                    For example, "The spectacles are on the red table next to a newspaper."

                    If you cannot find the **{item_to_find}** in the video, please respond with "Item not found in the video."

                    Respond in a concise and user-friendly way.
                """

                model = load_gemini_model(api_key) # Get cached Gemini model
                print(f"Sending Find Item request to Gemini API for item: {item_to_find}") # Debug print
                location_response = model.generate_content([video_file, location_prompt])
                print(f"Raw Gemini Response (Find Item Tab): {location_response}") # Debug print
                print(f"Raw Gemini Response Text (Find Item Tab): {location_response.text}") # Debug print
                location_text = location_response.text

                location_output.success(f"Location of '{item_to_find}':") # Display success message in container
                location_output.write(location_text) # Display location text in container

                # --- Text-to-speech implementation ---
                try: # Wrap TTS in try-except to isolate TTS errors
                    engine.say(location_text) # Queue the text for speech
                    engine.runAndWait() # Play the queued speech
                except Exception as tts_error:
                    st.error(f"Text-to-speech error: {tts_error}") # Display TTS error, but don't break the app

        except Exception as e:
            location_output.error(f"Error finding item location: {str(e)}") # Display error in container
            location_output.error(f"Please make sure your API key is valid and try again. Error details: {e}") # Display error in container

    elif find_button and not st.session_state.processed_video:
        location_output.warning("Please upload and process a video first in the 'Upload & Process' tab before using 'Find Item'.") # Display warning in container

with tab5:  # Fall Alert Tab
    st.header("Fall Alert System")
    st.markdown("Click 'Start Fall Detection' to activate real-time fall monitoring using your webcam.")

    # Add a stop button
    stop_detection = st.button("Stop Fall Detection", key="stop_fall_detection")
    
    fall_detection_start = st.button("Start Fall Detection", use_container_width=True, key="start_fall_detection")
    fall_detection_display = st.empty()  # Placeholder to display video frames

    # Flag to control detection loop
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False

    if fall_detection_start:
        st.session_state.detection_active = True

    if stop_detection:
        st.session_state.detection_active = False

    if st.session_state.detection_active:
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        # Video capture
        cap = cv2.VideoCapture(0)

        # Parameters
        GROUND_SMOOTHING = 0.9
        FALL_THRESHOLD = 0.35
        ANGLE_THRESHOLD = 35
        FALL_DURATION = 1.5
        STILL_FALL_DURATION = 5
        NO_MOVEMENT_DURATION = 3
        SITTING_ANGLE_THRESHOLD = 60

        # Tracking variables
        previous_ground_y = None
        fall_start_time = None
        still_fall_start_time = None
        no_movement_start_time = None
        fall_detected = False

        # Detection loop
        while st.session_state.detection_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Error reading video feed. Please check your webcam.")
                break

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform pose detection
            result = pose.process(rgb_frame)
            h, w, _ = frame.shape

            ground_candidates = []
            keypoints_on_ground = 0
            total_keypoints = 0
            torso_angle = None

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark

                # Identify ground level using ankle points
                for i in [mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value]:
                    ground_candidates.append(int(landmarks[i].y * h))

                # Smooth ground estimation
                if ground_candidates:
                    estimated_ground_y = int(np.median(ground_candidates))
                    if previous_ground_y is None:
                        previous_ground_y = estimated_ground_y
                    else:
                        previous_ground_y = int(GROUND_SMOOTHING * previous_ground_y + (1 - GROUND_SMOOTHING) * estimated_ground_y)

                # Get torso keypoints
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                # Calculate torso angle
                dx = (right_shoulder.x - left_shoulder.x) * w
                dy = (right_shoulder.y - left_shoulder.y) * h
                if dx != 0:
                    torso_angle = np.degrees(np.arctan(abs(dy / dx)))

                # Count keypoints near the ground
                for i in range(len(landmarks)):
                    x, y = int(landmarks[i].x * w), int(landmarks[i].y * h)
                    total_keypoints += 1
                    if y >= previous_ground_y:
                        keypoints_on_ground += 1
                        cv2.circle(frame, (x, y), 6, (0, 0, 255), 3)  # Red for ground contact
                    else:
                        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)  # Green for normal keypoints

                # Check if the person is lying down (fall position)
                is_lying_down = torso_angle is not None and torso_angle < ANGLE_THRESHOLD
                is_sitting = torso_angle is not None and torso_angle > SITTING_ANGLE_THRESHOLD

                # Detect fall based on motion and position (excluding sitting)
                if keypoints_on_ground / total_keypoints > FALL_THRESHOLD and is_lying_down and not is_sitting:
                    if not fall_detected:
                        fall_start_time = time.time()
                        fall_detected = True
                    elif time.time() - fall_start_time >= FALL_DURATION:
                        cv2.putText(frame, "FALL DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        st.warning("FALL DETECTED!")  # Streamlit warning

                # If a person has already fallen and remains still
                elif fall_detected and is_lying_down:
                    if still_fall_start_time is None:
                        still_fall_start_time = time.time()
                    elif time.time() - still_fall_start_time >= STILL_FALL_DURATION:
                        cv2.putText(frame, "PERSON STILL LYING DOWN!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        st.warning("PERSON STILL LYING DOWN!")
                    
                    # Detect no movement after falling
                    if no_movement_start_time is None:
                        no_movement_start_time = time.time()
                    elif time.time() - no_movement_start_time >= NO_MOVEMENT_DURATION:
                        cv2.putText(frame, "NO MOVEMENT DETECTED!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        st.warning("NO MOVEMENT DETECTED!")
                else:
                    fall_detected = False
                    fall_start_time = None
                    still_fall_start_time = None
                    no_movement_start_time = None

                # Draw ground reference
                cv2.line(frame, (0, previous_ground_y), (w, previous_ground_y), (255, 255, 0), 2)

            # Display frame in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fall_detection_display.image(frame_rgb, channels="RGB")

            # Add a small delay to prevent overwhelming the CPU
            time.sleep(0.1)

        # Cleanup
        cap.release()
        pose.close()
        st.session_state.detection_active = False
        st.success("Fall Detection Stopped.")


        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center">
            <p>Made with ❤️ using Streamlit and Google's Gemini AI</p>
        </div>
        """, unsafe_allow_html=True)

        # --- INSTRUCTIONS COMMENTED OUT BELOW ---
        # **To use this code:**
        # 1. **Copy the entire code block above.**
        # 2. **Replace the content of your `app.py` file** in your Streamlit project with this copied code.
        # 3. **Ensure `playsound` is removed from your `requirements.txt` file** (as per the previous instructions).
        # 4. **Commit and push your changes** to your Git repository.
        # 5. **Streamlit Cloud will automatically redeploy your app.**

        # **If it's still not working, please check the following and provide more details:**
        #     *   **Webcam Permissions:** Make sure your browser is allowing webcam access for your Streamlit app. You might need to check your browser settings.
        #     *   **Browser Console:** Open your browser's developer console (right-click on the page, "Inspect" or "Inspect Element", then go to "Console"). Are there any errors or warnings showing up there, especially when you start fall detection?
        #     *   **Debug Prints:** The code now includes `print()` statements in the Fall Alert tab. When you run the Fall Detection, check the Streamlit logs (or your local terminal if running locally) for the output of these `print()` statements. This will help understand if the fall detection logic is being triggered and what values are being calculated. Share these debug print outputs if you are still having issues.
        #     *   **Simplified Test (If still failing):** If even with the debug prints, you can't figure out the issue, try a very simplified test.  Replace the entire `while cap.isOpened():` loop in the Fall Alert tab with just these lines:
        #
        #         ```python
        #         while cap.isOpened():
        #             ret, frame = cap.read()
        #             if not ret:
        #                 st.warning("Error reading video feed. Please check your webcam.")
        #                 break
        #             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #             fall_detection_display.image(frame_rgb, channels="RGB")
        #             if cv2.waitKey(1) & 0xFF == ord('q'):
        #                 break
        #         ```
        #
        #         This simplified loop *only* displays the webcam feed without any pose estimation. If this works and you see your webcam video in the Streamlit app, then the issue is likely in the MediaPipe pose estimation or fall detection logic. If even this simplified test doesn't show the webcam feed, then the problem is with webcam access or OpenCV's `VideoCapture(0)`.
