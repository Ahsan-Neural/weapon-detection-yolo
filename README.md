# Weapon Detection System (YOLOv8)

A real-time computer vision application for detecting weapons and potential threat items in images, video recordings, and live webcam streams using a fine-tuned YOLOv8 model and Streamlit.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20App-FF4B4B?style=flat-square)](https://ahsan-weapon-detection.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Model-YOLOv8-00FFFF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

Live Application: [ahsan-weapon-detection.streamlit.app](https://ahsan-weapon-detection.streamlit.app/)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Detection Modes](#detection-modes)
- [Model Details](#model-details)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [Programmatic Usage (Python API)](#programmatic-usage-python-api)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Author](#author)

---

## Overview

This project provides an end-to-end threat detection pipeline built around Ultralytics YOLOv8 and an interactive Streamlit user interface. The system detects weapon categories (such as pistols and knives) while filtering and evaluating common objects often confused with weapons in surveillance and robbery scenarios.

The application supports three operational modes: static image upload, continuous live video streaming via WebRTC, and batch frame-by-frame analysis of uploaded video files with interactive timestamp indexing.

---

## Key Features

- Real-Time Inference: Fast object detection using a lightweight, fine-tuned YOLOv8 architecture suitable for CPU and GPU execution.
- WebRTC Live Streaming: Browser-based camera streaming with asynchronous frame processing and real-time bounding box overlays.
- Video Timeline Inspection: Analyzes recorded video footage, samples frames based on a user-defined stride, and generates an interactive timestamped catalog of flagged detections.
- Dynamic Thresholding: Adjustable confidence slider (0.10 to 0.95) allowing runtime tuning without reloading model weights.
- Structured Metrics: Displays bounding boxes, detection counts, confidence metrics, and category summaries for each evaluated source.
- Production Deployment Ready: Preconfigured with system package manifests (`packages.txt`) and Python runtime pins (`runtime.txt`) for cloud deployment.

---

## Detection Modes

### 1. Image Upload
- Supported formats: JPG, JPEG, PNG.
- Provides a side-by-side comparison of the original image and annotated detection results.
- Outputs summary metrics: total detection count, average confidence score, detected classes, and individual detection tags.

### 2. Live Webcam Feed
- Implemented via `streamlit-webrtc` using Google public STUN servers for peer-to-peer streaming.
- Asynchronous video processor converts frames to BGR format, evaluates detections using YOLOv8, and overlays bounding boxes and alert banners directly onto the live feed.

### 3. Video Analysis
- Supported formats: MP4, AVI, MOV, MKV.
- Configurable frame stride: inspect every Nth frame (default: 5) to balance processing throughput and detection sensitivity.
- Post-analysis summary reporting total frames processed, number of flagged frames, and total detections.
- Interactive frame selector displaying exact video timestamps (in seconds), annotated snapshots, and a timeline badge index.

---

## Model Details

- Architecture: YOLOv8 (Nano variant)
- Weights File: `best.pt` (~6.2 MB)
- Task: 2D Bounding Box Object Detection
- Model Classes:
  - Class 0: Pistol
  - Class 1: Smartphone
  - Class 2: Knife
  - Class 3: Purse
  - Class 4: Bill
  - Class 5: Card

The model includes everyday carry items (smartphones, wallets/purses, cards, banknotes) alongside primary threat objects (pistols, knives) to assist in reducing false-positive rates in surveillance and security screening contexts.

---

## Repository Structure

```text
weapon-detection-yolo/
|-- app.py              # Streamlit web application and WebRTC pipeline
|-- best.pt             # Pre-trained YOLOv8 model weights
|-- packages.txt        # Debian system dependencies (libgl1, libglib)
|-- requirements.txt    # Python library dependencies
|-- runtime.txt         # Pinned Python version (python-3.11)
|-- LICENSE             # MIT license file
`-- README.md           # Project documentation
```

---

## Prerequisites

- Operating System: Linux, macOS, or Windows
- Python: Version 3.10 or 3.11 (tested on Python 3.11)
- Pip: Standard Python package manager
- Webcam (optional): Required for the live WebRTC streaming mode

For Linux environments (including Ubuntu and Debian containers), install the following system graphics libraries required by OpenCV:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

---

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Ahsan-Neural/weapon-detection-yolo.git
   cd weapon-detection-yolo
   ```

2. Create and activate a virtual environment:

   On Linux/macOS:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   On Windows (Command Prompt):
   ```cmd
   python -m venv venv
   venv\Scripts\activate.bat
   ```

   On Windows (PowerShell):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install required Python packages:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Running the Application

Launch the Streamlit server from the root directory:

```bash
streamlit run app.py
```

Once started, the application will be accessible at:

```text
Local URL:  http://localhost:8501
```

Use the sidebar to choose the detection mode (Image, Webcam, Video) and adjust the confidence threshold slider as needed.

---

## Programmatic Usage (Python API)

You can also use the fine-tuned weights (`best.pt`) directly in your own scripts or pipelines with the Ultralytics Python package:

```python
from ultralytics import YOLO
import cv2

# Load the trained model weights
model = YOLO("best.pt")

# Perform inference on an image or video file
results = model.predict(source="sample.jpg", conf=0.5, verbose=False)

# Parse detections
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        print(f"Detected: {class_name} | Confidence: {confidence:.2f} | BBox: [{x1}, {y1}, {x2}, {y2}]")

    # Save annotated output image
    annotated_frame = result.plot()
    cv2.imwrite("output_detected.jpg", annotated_frame)
```

---

## Deployment

### Streamlit Community Cloud

This repository is structured for direct deployment on Streamlit Community Cloud:

1. Fork or push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Create a new application and point to:
   - Repository: `Ahsan-Neural/weapon-detection-yolo`
   - Branch: `main`
   - Main file path: `app.py`
4. The platform automatically detects `packages.txt` for system libraries and `requirements.txt` for Python packages.

---

## Troubleshooting

- WebRTC camera does not start:
  Ensure camera access permissions are granted in your web browser. If connecting from a remote network or behind symmetric NATs, configure additional STUN/TURN servers in the `RTCConfiguration` block inside `app.py`.
- OpenCV libGL error on Linux:
  If you encounter `ImportError: libGL.so.1: cannot open shared object file`, ensure `libgl1` is installed via `sudo apt-get install -y libgl1`.
- Slow video processing:
  Increase the frame stride value in the sidebar (for example, analyze every 10th or 15th frame) to speed up video scans on CPU-limited hosts.

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).

---

## Author

M. Ahsan
- GitHub: [@Ahsan-Neural](https://github.com/Ahsan-Neural)
- Kaggle: [@ahsanneural](https://www.kaggle.com/ahsanneural)
