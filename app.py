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

# Configure page
st.set_page_config(
    page_title="KeepTrack - Video Item Inventory",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
st.title("KeepTrack - Video Item Inventory")
st.markdown("""
This app helps you create an inventory of items from a video walkthrough of your space.
Upload a video and let AI help catalog your items for insurance or tracking purposes.

### Features:
- 📹 Process video content using AI
- 📝 Generate detailed inventory with item descriptions
- 💰 Estimate item values
- 📊 View inventory statistics
- ⬇️ Export data to CSV
- 🗣️ Text-to-speech for item location
- 🔍 Find misplaced items in your video
""")

# Sidebar for API key and configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Gemini API Key", type="password", help="Get your API key from Google AI Studio")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.success("API key configured successfully!")
        except Exception as e:
            st.error(f"Error configuring API key: {str(e)}")

    st.markdown("---")
    st.markdown("""
    ### Instructions
    1. Enter your Gemini API key
    2. Upload a video file
    3. Click "Process Video"
    4. View results in the tabs
    5. Use "Find Item" tab to locate items (with voice output)
    """)

# Cached Gemini Model (using st.cache_resource)
@st.cache_resource
def load_gemini_model(api_key):
    genai.configure(api_key=api_key) # Configure API here as well, just to be safe
    return genai.GenerativeModel('gemini-1.5-pro')

# Cached TTS Engine (using st.cache_resource)
@st.cache_resource
def load_tts_engine():
    return pyttsx3.init()

# Initialize TTS engine globally using the cached function
tts_engine = load_tts_engine()


# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Upload & Process", "Inventory", "Frames", "Find Item"]) # Added tab4 "Find Item"

with tab1:
    st.header("Upload Video")

    # File uploader with clear instructions
    uploaded_file = st.file_uploader(
        "Choose a video file (MP4, MOV, or AVI)",
        type=['mp4', 'mov', 'avi'],
        help="Maximum file size: 200MB"
    )

    if uploaded_file:
        # Show video details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
            "File type": uploaded_file.type
        }
        st.json(file_details)

        # Create process button
        process_button = st.button(
            "Process Video",
            help="Click to start AI analysis of the video",
            use_container_width=True
        )

        if process_button:
            try:
                with st.spinner("Processing video..."):
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
                                    - If double quotes appear within a field value, escape them by doubling them (`\"\"`). For example, `John \"JJ\" Smith` should be written as `\"John \"\"JJ\"\" Smith\"`.
                                    - Do not use any unescaped double quotes or other special characters that may cause the CSV to be invalid.
                                    - Never use commas in prices. For example, $1500.0 = good, $1,500.0 = bad.
                                     - Return all columns in the order they are presented in.

                                Return the full and valid CSV within <csv> tags so it can be easily parsed out.
                                  Include a note to yourself in <other_details_to_note> if you think there are more items to continue doing in the video, this should be inline with the <item_logging_status> as well as the next steps to improve upon the record keeping.
                        """

                        # Generate response with progress bar
                        progress_bar = st.progress(0)
                        progress_text = st.empty()

                        progress_text.text("Analyzing video...")
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
                            progress_text.text("Processing frames...")
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
                            progress_text.text("Processing complete!")

                            st.session_state.processing_complete = True
                            st.success("Video processed successfully! Check the Inventory and Frames tabs for results.")


                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
                        st.error("Please make sure your API key is valid and try again.")

                    finally:
                        # Cleanup
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

            except Exception as e:
                st.error(f"Error handling file: {str(e)}")

        with tab2:
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

        with tab3:
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

        with tab4: # New "Find Item" Tab
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


        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center">
            <p>Made with ❤️ using Streamlit and Google's Gemini AI</p>
        </div>
        """, unsafe_allow_html=True)