# ==========================================================
# Computer Vision Object Detector & Segmenter Pipeline
# Professional Streamlit Application
# ==========================================================

# Import Streamlit
import streamlit as st

# Import OpenCV
import cv2

# Import NumPy
import numpy as np

# Temporary file handling
import tempfile

# Time module for FPS calculation
import time

# YOLO Model
from ultralytics import YOLO


# ==========================================================
# Load YOLO Model
# ==========================================================

@st.cache_resource
def load_model():

    return YOLO("yolov8n.pt")

model = load_model()

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("Settings")

mode = st.sidebar.selectbox(

    "Select Input Type",

    [

        "Image",

        "Video",

        "Webcam"

    ]

)
st.sidebar.markdown("---")

st.sidebar.write("### Project Features")

st.sidebar.success("YOLOv8 Detection")

st.sidebar.success("Gaussian Blur")

st.sidebar.success("Adaptive Threshold")

st.sidebar.success("Morphological Operations")

st.sidebar.success("Contour Detection")

st.sidebar.success("Object Counting")

st.sidebar.success("FPS Calculation")

st.sidebar.success("Confidence Scores")

# ==========================================================
# Image Processing Function
# ==========================================================

def process_image(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Adaptive Threshold
    threshold = cv2.adaptiveThreshold(

        blur,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY_INV,

        11,

        2

    )

    # Morphological Closing
    kernel = np.ones((3,3), np.uint8)

    morph = cv2.morphologyEx(

        threshold,

        cv2.MORPH_CLOSE,

        kernel

    )

    # Contours
    contours, _ = cv2.findContours(

        morph,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )

    # YOLO Detection
    results = model(image)

    output = results[0].plot()

    object_count = len(results[0].boxes)

    return output, contours, object_count, results

# ==========================================================
# Detection Summary Function
# ==========================================================

def create_detection_table(results):

    detection_data = []

    names = results[0].names

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        detection_data.append({

            "Object": names[class_id],

            "Confidence": round(confidence, 3)

        })

    return detection_data
success, encoded_image = cv2.imencode(".jpg", output)

if success:

    st.download_button(

        "Download Processed Image",

        encoded_image.tobytes(),

        file_name="detected_image.jpg",

        mime="image/jpeg"

    )

# ==========================================================
# IMAGE MODE
# ==========================================================

if mode == "Image":

    uploaded = st.file_uploader(

        "Upload Image",

        type=["jpg","jpeg","png"]

    )

    if uploaded:

        file_bytes = np.asarray(

            bytearray(uploaded.read()),

            dtype=np.uint8

        )

        image = cv2.imdecode(

            file_bytes,

            cv2.IMREAD_COLOR

        )

        output, contours, count, results = process_image(image)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original")

            st.image(

                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),

                use_container_width=True

            )

        with col2:

            st.subheader("Detection")

            st.image(

                cv2.cvtColor(output, cv2.COLOR_BGR2RGB),

                use_container_width=True

            )

        st.success(f"Objects Detected: {count}")

        st.info(f"Contours Found: {len(contours)}")
        
        table = create_detection_table(results)

if table:

    st.subheader("Detection Summary")

    st.dataframe(table, use_container_width=True)
    
    

        # ==========================================================
# VIDEO MODE
# ==========================================================

elif mode == "Video":

    uploaded_video = st.file_uploader(

        "Upload Video",

        type=["mp4", "avi", "mov"]

    )

    if uploaded_video is not None:

        # Save uploaded video temporarily
        temp_video = tempfile.NamedTemporaryFile(delete=False)

        temp_video.write(uploaded_video.read())

        temp_video.close()

        # Open uploaded video
        cap = cv2.VideoCapture(temp_video.name)

        # Get video information
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = cap.get(cv2.CAP_PROP_FPS)

        # Output video path
        output_path = "output/processed_video.mp4"

        out = cv2.VideoWriter(

            output_path,

            cv2.VideoWriter_fourcc(*"mp4v"),

            fps_input,

            (frame_width, frame_height)

        )

        # Streamlit placeholders
        video_placeholder = st.empty()

        info_placeholder = st.empty()

        previous_time = time.time()

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            # ---------------------------------------
            # Image Processing
            # ---------------------------------------

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            threshold = cv2.adaptiveThreshold(

                blur,

                255,

                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

                cv2.THRESH_BINARY_INV,

                11,

                2

            )

            kernel = np.ones((3, 3), np.uint8)

            morph = cv2.morphologyEx(

                threshold,

                cv2.MORPH_CLOSE,

                kernel

            )

            contours, _ = cv2.findContours(

                morph,

                cv2.RETR_EXTERNAL,

                cv2.CHAIN_APPROX_SIMPLE

            )

            # ---------------------------------------
            # YOLO Detection
            # ---------------------------------------

            results = model(frame)

            annotated = results[0].plot()

            object_count = len(results[0].boxes)

            # ---------------------------------------
            # FPS
            # ---------------------------------------

            current_time = time.time()

            fps = 1 / (current_time - previous_time)

            previous_time = current_time

            # ---------------------------------------
            # Overlay Information
            # ---------------------------------------

            cv2.putText(

                annotated,

                f"Objects: {object_count}",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0, 255, 0),

                2

            )

            cv2.putText(

                annotated,

                f"Contours: {len(contours)}",

                (20, 80),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (255, 0, 0),

                2

            )

            cv2.putText(

                annotated,

                f"FPS: {fps:.2f}",

                (20, 120),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0, 0, 255),

                2

            )

            # Save processed frame
            out.write(annotated)

            # Display current frame
            video_placeholder.image(

                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),

                channels="RGB",

                use_container_width=True

            )

            # Live statistics
            info_placeholder.info(

                f"""
                **Objects Detected:** {object_count}

                **Contours:** {len(contours)}

                **FPS:** {fps:.2f}
                """
            )

        cap.release()

        out.release()

        st.success("Video processing completed successfully!")

        # Display processed video
        st.video(output_path)

        # Download button
        with open(output_path, "rb") as video_file:

            st.download_button(

                label="Download Processed Video",

                data=video_file,

                file_name="processed_video.mp4",

                mime="video/mp4"

            )
            
            # ==========================================================
# WEBCAM MODE
# ==========================================================

elif mode == "Webcam":

    st.warning("Press the Stop button in Streamlit to end webcam detection.")

    start_camera = st.button("Start Webcam")

    if start_camera:

        cap = cv2.VideoCapture(0)

        frame_placeholder = st.empty()

        info_placeholder = st.empty()

        previous_time = time.time()

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                st.error("Unable to access webcam.")
                break

            # Process frame using our helper function
            output, contours, object_count, results = process_image(frame)

            # FPS Calculation
            current_time = time.time()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time

            # Display statistics on frame
            cv2.putText(
                output,
                f"Objects: {object_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            cv2.putText(
                output,
                f"Contours: {len(contours)}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
                2
            )

            cv2.putText(
                output,
                f"FPS: {fps:.2f}",
                (20,120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2
            )

            # Display webcam frame
            frame_placeholder.image(
                cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )

            info_placeholder.success(
                f"""
Objects Detected: {object_count}

Contours: {len(contours)}

FPS: {fps:.2f}
"""
            )

        cap.release()
