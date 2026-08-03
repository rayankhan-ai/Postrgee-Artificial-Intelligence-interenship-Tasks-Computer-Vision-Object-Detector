# ==========================================================
# Computer Vision Object Detector & Segmenter Pipeline
# Professional Streamlit Application
# ==========================================================

# =========================
# Import Required Libraries
# =========================

import os
import time
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from ultralytics import YOLO

# ==========================================================
# Streamlit Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Computer Vision Object Detector",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Computer Vision Object Detector & Segmenter Pipeline")

st.markdown("""
This application performs:

- Real-Time YOLOv8 Object Detection
- Gaussian Blur
- Adaptive Thresholding
- Morphological Operations
- Contour Detection
- Image Detection
- Video Detection
- Webcam Detection
- FPS Monitoring
- Object Counting
""")

# ==========================================================
# Create Output Folder
# ==========================================================

os.makedirs("output", exist_ok=True)

# ==========================================================
# Load YOLO Model
# ==========================================================

MODEL_PATH = "yolov8n.pt"

@st.cache_resource
def load_model():

    try:

        model = YOLO(MODEL_PATH)

        return model

    except Exception as e:

        st.error(f"Unable to load YOLO model.\n\n{e}")

        st.stop()

model = load_model()

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("⚙ Settings")

mode = st.sidebar.selectbox(

    "Select Input",

    [

        "Image",

        "Video",

        "Webcam"

    ]

)

st.sidebar.markdown("---")

st.sidebar.subheader("Project Features")

st.sidebar.success("YOLOv8 Detection")
st.sidebar.success("Gaussian Blur")
st.sidebar.success("Adaptive Threshold")
st.sidebar.success("Morphological Closing")
st.sidebar.success("Contour Detection")
st.sidebar.success("Object Counter")
st.sidebar.success("FPS Counter")
st.sidebar.success("Download Results")

# ==========================================================
# Image Processing Function
# ==========================================================

def process_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(

        gray,

        (5,5),

        0

    )

    threshold = cv2.adaptiveThreshold(

        blur,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY_INV,

        11,

        2

    )

    kernel = np.ones(

        (3,3),

        np.uint8

    )

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

    if len(results[0].boxes) == 0:
        return pd.DataFrame(
            columns=["Object", "Confidence"]
        )

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        detection_data.append(

            {

                "Object": names[class_id],

                "Confidence": round(confidence, 3)

            }

        )

    return pd.DataFrame(detection_data)


# ==========================================================
# Draw Statistics on Image
# ==========================================================

    def draw_statistics(

        image,

        object_count,

        contour_count,

        fps=None

    ):

        cv2.putText(

            image,

            f"Objects : {object_count}",

            (20,40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            (0,255,0),

            2

        )

        cv2.putText(

            image,

            f"Contours : {contour_count}",

            (20,80),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            (255,0,0),

            2

        )

        if fps is not None:

            cv2.putText(

                image,

                f"FPS : {fps:.2f}",

                (20,120),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.9,

                (0,0,255),

                2

            )

        return image


    # ==========================================================
    # IMAGE MODE
    # ==========================================================

    if mode == "Image":

        st.header("🖼 Image Detection")

        uploaded = st.file_uploader(

            "Upload an Image",

            type=[

                "jpg",

                "jpeg",

                "png"

            ]

        )

        if uploaded is not None:

            file_bytes = np.asarray(

                bytearray(uploaded.read()),

                dtype=np.uint8

            )

            image = cv2.imdecode(

                file_bytes,

                cv2.IMREAD_COLOR

            )

            output, contours, count, results = process_image(image)

            output = draw_statistics(

                output,

                count,

                len(contours)

            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Original Image")

                st.image(

                    cv2.cvtColor(

                        image,

                        cv2.COLOR_BGR2RGB

                    ),

                    use_container_width=True

                )

            with col2:

                st.subheader("Detected Image")

                st.image(

                    cv2.cvtColor(

                        output,

                        cv2.COLOR_BGR2RGB

                    ),

                    use_container_width=True

                )

            st.success(

                f"Objects Detected : {count}"

            )

            st.info(

                f"Contours Found : {len(contours)}"

            )

            table = create_detection_table(results)

            st.subheader("Detection Summary")

            st.dataframe(

                table,

                use_container_width=True

            )

            success, encoded_image = cv2.imencode(

                ".jpg",

                output

            )

            if success:

                st.download_button(

                    label="📥 Download Processed Image",

                    data=encoded_image.tobytes(),

                    file_name="processed_image.jpg",

                    mime="image/jpeg"

                )


    # ==========================================================
    # IMAGE MODE
    # ==========================================================

    if mode == "Image":

        uploaded = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded is not None:

            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)

            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

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

            # Detection table
            table = create_detection_table(results)

            if table:
                st.subheader("Detection Summary")
                st.dataframe(table, use_container_width=True)

            # Download processed image
            success, encoded_image = cv2.imencode(".jpg", output)

            if success:
                st.download_button(
                    "Download Processed Image",
                    data=encoded_image.tobytes(),
                    file_name="detected_image.jpg",
                    mime="image/jpeg"
                )    
# ==========================================================
# VIDEO MODE
# ==========================================================

    elif mode == "Video":

        st.header("🎥 Video Detection")

        uploaded_video = st.file_uploader(

            "Upload a Video",

            type=["mp4", "avi", "mov"]

        )

        if uploaded_video is not None:

            # Save uploaded video temporarily
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

            temp_video.write(uploaded_video.read())

            temp_video.close()

            # Open video
            cap = cv2.VideoCapture(temp_video.name)

            if not cap.isOpened():

                st.error("Unable to open uploaded video.")

                st.stop()

            # Read video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fps_input = cap.get(cv2.CAP_PROP_FPS)

            if fps_input <= 0:

                fps_input = 30

            output_path = "output/processed_video.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out = cv2.VideoWriter(

                output_path,

                fourcc,

                fps_input,

                (frame_width, frame_height)

            )

            video_placeholder = st.empty()

            info_placeholder = st.empty()

            progress_bar = st.progress(0)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            processed_frames = 0

            previous_time = time.time()

            while True:

                success, frame = cap.read()

                if not success:

                    break

                # Process current frame
                output, contours, object_count, results = process_image(frame)

                current_time = time.time()

                elapsed = current_time - previous_time

                fps = 1 / elapsed if elapsed > 0 else 0

                previous_time = current_time

                output = draw_statistics(

                    output,

                    object_count,

                    len(contours),

                    fps

                )

                out.write(output)

                video_placeholder.image(

                    cv2.cvtColor(output, cv2.COLOR_BGR2RGB),

                    channels="RGB",

                    use_container_width=True

                )

                info_placeholder.info(

                    f"""
    ### Live Statistics

    **Objects Detected:** {object_count}

    **Contours Found:** {len(contours)}

    **FPS:** {fps:.2f}
    """
                )

                processed_frames += 1

                if total_frames > 0:

                    progress_bar.progress(

                        min(processed_frames / total_frames, 1.0)

                    )

            cap.release()

            out.release()

            progress_bar.empty()

            st.success("✅ Video Processing Completed Successfully!")

            st.video(output_path)

            with open(output_path, "rb") as video_file:

                st.download_button(

                    label="📥 Download Processed Video",

                    data=video_file,

                    file_name="processed_video.mp4",

                    mime="video/mp4"

                )
                
    # ==========================================================
    # WEBCAM MODE
    # ==========================================================

    elif mode == "Webcam":

        st.header("📷 Real-Time Webcam Detection")

        st.info(
            "Click **Start Webcam** to begin live object detection. "
            "Click **Stop** in Streamlit to terminate the session."
        )

        start_camera = st.button("▶ Start Webcam")

        if start_camera:

            cap = cv2.VideoCapture(0)

            if not cap.isOpened():

                st.error("❌ Unable to access your webcam.")

                st.stop()

            frame_placeholder = st.empty()

            info_placeholder = st.empty()

            previous_time = time.time()

            while True:

                success, frame = cap.read()

                if not success:

                    st.error("Unable to read webcam frame.")

                    break

                # Process current frame
                output, contours, object_count, results = process_image(frame)

                # Calculate FPS
                current_time = time.time()

                elapsed = current_time - previous_time

                fps = 1 / elapsed if elapsed > 0 else 0

                previous_time = current_time

                # Draw statistics
                output = draw_statistics(

                    output,

                    object_count,

                    len(contours),

                    fps

                )

                # Display webcam frame
                frame_placeholder.image(

                    cv2.cvtColor(

                        output,

                        cv2.COLOR_BGR2RGB

                    ),

                    channels="RGB",

                    use_container_width=True

                )

                # Display live statistics
                info_placeholder.success(

                    f"""
    ### Live Detection

    **Objects Detected:** {object_count}

    **Contours Found:** {len(contours)}

    **FPS:** {fps:.2f}
    """
                )

            cap.release()

            cv2.destroyAllWindows()

# ==========================================================
# APPLICATION FOOTER
# ==========================================================

st.markdown("---")

st.header("📊 Model Performance")

st.markdown("""
This application combines traditional Computer Vision techniques with
YOLOv8 deep learning for real-time object detection.

### Computer Vision Techniques Used
- Gaussian Blur
- Adaptive Thresholding
- Morphological Closing
- Contour Detection
- Bounding Box Visualization

### Deep Learning Model
- YOLOv8 Nano (Ultralytics)

### Frameworks
- Streamlit
- OpenCV
- NumPy
- Pandas
- Ultralytics YOLO

### Supported Inputs
- Images (.jpg, .jpeg, .png)
- Videos (.mp4, .avi, .mov)
- Live Webcam

### Performance Metrics
The application displays:

- Number of Objects Detected
- Number of Contours
- Confidence Scores
- FPS (Frames Per Second)

These metrics are updated dynamically during inference.
""")

st.markdown("---")

st.header("📄 Project Information")

st.info("""
Project Title:

Computer Vision Object Detector & Frame-by-Frame Segmenter Pipeline

Objective:

Build a real-time computer vision processing pipeline capable of
detecting objects using YOLOv8 while simultaneously extracting contours
through classical image processing techniques.

Features:

• Real-time Object Detection

• Gaussian Blur

• Adaptive Thresholding

• Morphological Operations

• Contour Detection

• Dynamic Object Counter

• FPS Counter

• Detection Summary

• Download Processed Image

• Download Processed Video

• Professional Streamlit Interface
""")

st.markdown("---")

st.success("✅ Application Loaded Successfully")

st.caption(
    "Computer Vision Object Detector & Segmenter Pipeline | "
    "Built with Streamlit, OpenCV and YOLOv8"
)