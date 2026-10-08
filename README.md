<div align="center">
  <h1 align="center">SentryVision AI</h1>
  <h3>Quantum-Tier Real-Time PPE Compliance & Safety Monitoring OS</h3>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/Streamlit-UI%2FUX-FF4B4B.svg" alt="Streamlit">
</p>

## 📖 Overview
**SentryVision** is an advanced, real-time computer vision platform designed to automate safety audits on industrial and construction sites. By leveraging state-of-the-art deep learning algorithms, SentryVision continuously monitors video feeds to ensure that all active personnel are wearing their required Personal Protective Equipment (PPE) such as **Hardhats** and **Safety Vests**.

If a worker steps into the optical feed without the required safety gear, the system's spatial rule engine immediately flags the breach, registers it in the violation logs, and updates the real-time Safety Index telemetry.

## ✨ Key Features
- 🧠 **Deep Learning Object Detection:** Powered by a fine-tuned YOLOv8 (You Only Look Once) Convolutional Neural Network.
- 🎯 **Identity Tracking:** Utilizes **ByteTrack** and Kalman filters to maintain persistent ID tracking of workers as they move across the site.
- 📐 **Spatial Association Engine:** Custom geometric rule engine that uses Intersection-over-Union (IoU) to logically associate detached PPE gear (helmets/vests) with the correct human body bounding boxes.
- 💻 **Cyber-Glassmorphism Dashboard:** An ultra-modern, reactive Streamlit web interface featuring animated metrics, live video overlays, and glitch-art aesthetics.
- 📝 **Automated Evidentiary Logging:** Instantly writes compliance breaches to a timestamped CSV registry (`violation_log.csv`) for site managers to audit.

## 🛠️ Architecture Stack
1. **Vision Core:** `Ultralytics YOLOv8` (Detection) + `ByteTrack` (Tracking)
2. **Matrix Manipulation:** `OpenCV (cv2)` + `NumPy`
3. **Frontend Dashboard:** `Streamlit` + `Custom CSS`
4. **Data Handling:** `Pandas` (CSV logging & Dataframes)

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PAWAN150732/Sentry-Vision.git
   cd Sentry-Vision
   ```

2. **Create and activate a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

SentryVision provides two operating modes depending on your deployment needs:

### 1. Tactical Web Dashboard (Streamlit UI)
To launch the full graphical interface with live telemetry and CSV logs:
```bash
cd src
streamlit run streamlit_app.py
```
*(The dashboard will open automatically in your browser at `http://localhost:8501`)*

### 2. Lightweight CLI / OpenCV Window
To run the raw detection engine without the web server (great for low-resource edge devices):
```bash
cd src
python detect_track.py --source 0
```
*(Replace `0` with the path to an `.mp4` file or an RTSP IP Camera stream).*

## 🧠 Model Training
SentryVision is currently optimized to run the Ultralytics `construction-ppe` dataset. 
To train the model on your own hardware to improve accuracy:
```bash
cd src
python train.py --epochs 50 --imgsz 640
```
Once training completes, the system will save the new weights to `runs/models/ppe_yolo/weights/best.pt`. Update your `src/config.py` to point to this new model path!

---
*Built to ensure zero-harm work environments through artificial intelligence.*