# 🎯 Computer Vision Object Detector & Frame-by-Frame Segmenter Pipeline

A real-time **Computer Vision Object Detection and Frame-by-Frame Segmentation Pipeline** built using **Python**, **OpenCV**, **YOLOv8**, and **Streamlit**. The application detects objects in images, videos, and live webcam streams while applying image processing techniques such as Gaussian filtering, adaptive thresholding, contour detection, and morphological operations.

---

## 📌 Project Overview

This project was developed as an internship mini project to demonstrate a complete computer vision pipeline capable of:

- Real-time object detection using YOLOv8
- Image preprocessing with OpenCV
- Contour extraction and segmentation
- Frame-by-frame video analysis
- Dynamic object counting
- FPS (Frames Per Second) monitoring
- Interactive Streamlit web application

---

## 🚀 Features

- 🖼️ Image Object Detection
- 🎥 Video Object Detection
- 📷 Live Webcam Detection
- 🤖 YOLOv8 Pre-trained Object Detector
- 🔍 Gaussian Blur
- ⚫ Adaptive Thresholding
- 🔷 Morphological Operations
- 📐 Contour Detection
- 🔢 Object Counting
- 📊 Detection Summary Table
- 📈 Confidence Score Visualization
- ⚡ FPS Counter
- 💾 Download Processed Images
- 💾 Download Processed Videos
- 🌐 Streamlit Web Interface

---

## 🛠️ Technologies Used

- Python 3.x
- OpenCV
- YOLOv8 (Ultralytics)
- NumPy
- Streamlit
- Pillow

---

## 📁 Project Structure

```
Computer-Vision-Object-Detector/

│
├── dataset/
│   ├── image1.jpg
│   └── traffic.mp4
│
├── models/
│   └── yolov8n.pt
│
├── output/
│   ├── detected_image.jpg
│   └── processed_video.mp4
│
├── app.py
├── requirements.txt
├── README.md
└── object_detector.ipynb
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/rayankhan-ai/Postrgee-Artificial-Intelligence-interenship-Tasks-Computer-Vision-Object-Detector

cd Computer-Vision-Object-Detector
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

After running the command, Streamlit will open the application in your default web browser.

---

## 🖼️ Supported Inputs

### Image

- JPG
- JPEG
- PNG

### Video

- MP4
- AVI
- MOV

### Webcam

- Live Camera Detection

---

## 🔍 Image Processing Pipeline

The application performs the following preprocessing steps before object detection:

1. Image Loading
2. Image Resizing
3. Grayscale Conversion
4. Gaussian Blur
5. Adaptive Thresholding
6. Morphological Closing
7. Contour Detection
8. YOLOv8 Object Detection
9. Object Counting
10. Frame Visualization

---

## 🤖 YOLOv8 Object Detection

The project uses the **YOLOv8 Nano** pre-trained model from **Ultralytics**.

Detected information includes:

- Object Class
- Bounding Box
- Confidence Score
- Total Object Count

---

## 📊 Output

The application displays:

- Original Image
- Processed Image
- Bounding Boxes
- Object Labels
- Confidence Scores
- Contour Count
- FPS Counter
- Detection Summary Table

Processed files are automatically saved in the **output/** folder.

---

## 📷 Screenshots

You can add screenshots here after deployment.

Example:

```
screenshots/

home_page.png

image_detection.png

video_detection.png

webcam_detection.png
```

---

## 📈 Future Improvements

- Object Tracking (DeepSORT)
- Instance Segmentation
- Custom YOLO Model Training
- Multi-Camera Support
- GPU Acceleration
- Detection History Logging
- Performance Dashboard

---

## 📚 Internship Requirements Covered

✅ Real-Time Computer Vision Pipeline

✅ OpenCV Image Processing

✅ Gaussian Filtering

✅ Adaptive Thresholding

✅ Contour Extraction

✅ YOLOv8 Object Detection

✅ Dynamic Object Count Overlay

✅ Frame-by-Frame Video Processing

✅ Streamlit Deployment

---

## 👨‍💻 Author

**Rayan Ahmad**

Software Engineering Student

Aspiring AI / Machine Learning Engineer

GitHub: https://github.com/rayankhan-ai

---

## 📄 License

This project is licensed under the MIT License.

Feel free to use, modify, and distribute this project for educational and research purposes.

---

## ⭐ Acknowledgements

- Ultralytics YOLOv8
- OpenCV
- Streamlit
- NumPy
- Python Community