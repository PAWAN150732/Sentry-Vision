"""
Streamlit dashboard for the PPE Compliance Monitoring system.

Run with:
    streamlit run streamlit_app.py
"""

import time
import tempfile

import cv2
import pandas as pd
import streamlit as st

import config
from ppe_engine import PPEComplianceEngine
from detect_track import draw_overlays

st.set_page_config(page_title="SentryVision OS | Quantum", layout="wide", initial_sidebar_state="collapsed")

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syncopate:wght@400;700&display=swap');
        
        /* 1. ANIMATED QUANTUM BACKGROUND */
        html, body, [class*="css"]  {
            font-family: 'Share Tech Mono', monospace !important;
            background: radial-gradient(circle at 50% 50%, #0d0628 0%, #03010b 100%) !important;
            color: #d4c5ff !important;
        }
        
        /* Animated background grid */
        body::before {
            content: "";
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background-image: 
                linear-gradient(rgba(141, 73, 255, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(141, 73, 255, 0.1) 1px, transparent 1px);
            background-size: 40px 40px;
            z-index: -1;
            animation: gridMove 20s linear infinite;
        }
        
        @keyframes gridMove {
            0% { transform: translateY(0); }
            100% { transform: translateY(40px); }
        }

        /* 2. HEADERS & TYPOGRAPHY */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Syncopate', sans-serif !important;
            text-transform: uppercase;
            letter-spacing: 4px;
            color: #fff !important;
            text-shadow: 0 0 10px #b17aff, 0 0 20px #8d49ff, 0 0 40px #8d49ff;
        }

        /* 3. FLOATING GLASSMORPHISM CONTAINERS */
        div[data-testid="stVerticalBlock"] > div {
            background: rgba(13, 6, 40, 0.4) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(141, 73, 255, 0.3) !important;
            border-radius: 15px !important;
            padding: 10px;
            box-shadow: 0 8px 32px 0 rgba(141, 73, 255, 0.15);
            transition: all 0.3s ease;
        }
        
        div[data-testid="stVerticalBlock"] > div:hover {
            box-shadow: 0 8px 32px 0 rgba(141, 73, 255, 0.4);
            border: 1px solid rgba(141, 73, 255, 0.8) !important;
        }

        /* 4. METRIC HUD CARDS */
        [data-testid="stMetricValue"] {
            font-family: 'Syncopate', sans-serif !important;
            font-size: 2.5rem !important;
            color: #00f0ff !important;
            text-shadow: 0 0 10px #00f0ff, 0 0 20px #00f0ff;
            animation: pulseText 2s infinite alternate;
        }
        @keyframes pulseText {
            0% { opacity: 0.8; text-shadow: 0 0 10px #00f0ff; }
            100% { opacity: 1; text-shadow: 0 0 25px #00f0ff, 0 0 50px #00f0ff; }
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
            color: #b17aff !important;
            letter-spacing: 2px;
        }
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, rgba(20,10,50,0.8) 0%, rgba(5,2,15,0.9) 100%);
            border: 2px solid #8d49ff;
            border-top: 4px solid #00f0ff;
            border-radius: 10px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }
        
        /* Scanning laser across metrics */
        div[data-testid="metric-container"]::after {
            content: '';
            position: absolute;
            top: 0; left: -100%;
            width: 50%; height: 100%;
            background: linear-gradient(to right, transparent, rgba(0, 240, 255, 0.2), transparent);
            transform: skewX(-20deg);
            animation: scan 3s infinite;
        }
        @keyframes scan {
            0% { left: -100%; }
            100% { left: 200%; }
        }

        /* 5. VIDEO TARGETING FRAME */
        [data-testid="stImage"] {
            position: relative;
            padding: 10px;
        }
        [data-testid="stImage"] img {
            border-radius: 8px;
            border: 1px solid #00f0ff;
            box-shadow: 0 0 30px rgba(0, 240, 255, 0.4);
            filter: contrast(1.1) brightness(1.1);
        }
        
        /* Crosshairs for video frame */
        [data-testid="stImage"]::before, [data-testid="stImage"]::after {
            content: ''; position: absolute; border: 2px solid #ff0055; width: 40px; height: 40px; z-index: 10;
        }
        [data-testid="stImage"]::before { top: 0; left: 0; border-right: none; border-bottom: none; }
        [data-testid="stImage"]::after { bottom: 0; right: 0; border-left: none; border-top: none; }

        /* 6. BUTTONS */
        .stButton>button {
            background: linear-gradient(45deg, #8d49ff, #00f0ff) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 30px !important;
            font-family: 'Syncopate', sans-serif !important;
            font-weight: 700;
            letter-spacing: 2px;
            padding: 10px 30px;
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.5);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            width: 100%;
        }
        .stButton>button:hover {
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 10px 25px rgba(0, 240, 255, 0.8), 0 0 15px rgba(141, 73, 255, 0.8);
        }
        
        /* Abort button specific styling */
        .stButton>button:nth-child(2) {
            background: linear-gradient(45deg, #ff0055, #ff6b00) !important;
            box-shadow: 0 0 15px rgba(255, 0, 85, 0.5);
        }
        .stButton>button:nth-child(2):hover {
            box-shadow: 0 10px 25px rgba(255, 0, 85, 0.8);
        }

        /* 7. DATAFRAMES / LOGS */
        [data-testid="stDataFrame"] {
            border: 1px solid #8d49ff;
            box-shadow: 0 0 20px rgba(141, 73, 255, 0.3);
            border-radius: 10px;
        }
        
        /* 8. GLITCH TITLE */
        .glitch-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            margin-bottom: 20px;
        }
        .glitch {
            position: relative;
            font-size: 3.5rem;
            font-family: 'Syncopate', sans-serif;
            font-weight: 700;
            color: #ffffff;
            text-shadow: 0 0 20px #00f0ff;
        }
        .glitch::before, .glitch::after {
            content: attr(data-text);
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
        }
        .glitch::before {
            left: 3px;
            text-shadow: -2px 0 #ff0055;
            animation: glitch-anim-1 2s infinite linear alternate-reverse;
        }
        .glitch::after {
            left: -3px;
            text-shadow: 2px 0 #00f0ff;
            animation: glitch-anim-2 3s infinite linear alternate-reverse;
        }
        @keyframes glitch-anim-1 {
            0% { clip-path: inset(20% 0 80% 0); }
            20% { clip-path: inset(60% 0 10% 0); }
            40% { clip-path: inset(40% 0 50% 0); }
            60% { clip-path: inset(80% 0 5% 0); }
            80% { clip-path: inset(10% 0 70% 0); }
            100% { clip-path: inset(30% 0 20% 0); }
        }
        @keyframes glitch-anim-2 {
            0% { clip-path: inset(10% 0 60% 0); }
            20% { clip-path: inset(30% 0 20% 0); }
            40% { clip-path: inset(70% 0 10% 0); }
            60% { clip-path: inset(20% 0 50% 0); }
            80% { clip-path: inset(50% 0 30% 0); }
            100% { clip-path: inset(5% 0 80% 0); }
        }
        </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_engine(model_path: str, conf: float):
    return PPEComplianceEngine(model_path=model_path, conf_threshold=conf)


def main():
    inject_custom_css()

    st.markdown('''
        <div class="glitch-wrapper">
            <div class="glitch" data-text="SENTRY_VISION_QUANTUM">SENTRY_VISION_QUANTUM</div>
        </div>
        <div style="text-align:center; font-family:'Share Tech Mono'; color:#00f0ff; letter-spacing:5px; margin-top:-20px; margin-bottom:30px;">
            > OPTICAL_CORTEX_ONLINE :: SYSTEM_V3.0_NEXUS
        </div>
    ''', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<h3 style='color:#00f0ff;'>COMMAND NODE</h3>", unsafe_allow_html=True)
        # Using a new key so Streamlit Cloud drops the old cached relative path
        model_path = st.text_input("NEURAL NET", value=config.MODEL_PATH, key="model_path_v2")
        conf = st.slider("CONFIDENCE GATE", 0.1, 0.9, config.CONF_THRESHOLD, 0.05)
        source_type = st.radio("FEED SOURCE", ["UPLOAD_VIDEO", "LIVE_WEBCAM"])
        video_file = None
        if source_type == "UPLOAD_VIDEO":
            video_file = st.file_uploader("ENGAGE VIDEO FILE", type=["mp4", "avi", "mov", "mkv"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("INITIALIZE NEXUS")
        stop_button = st.button("ABORT SEQUENCE")

    # Layout for top metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    m1 = metrics_col1.empty()
    m2 = metrics_col2.empty()
    m3 = metrics_col3.empty()
    m4 = metrics_col4.empty()
    
    # Layout for main video and logs
    st.markdown("<h3 style='margin-top:20px;'>LIVE OPTICAL STREAM</h3>", unsafe_allow_html=True)
    frame_placeholder = st.empty()
    
    st.markdown("<h3 style='margin-top:40px;'>VIOLATION REGISTRY MATRIX</h3>", unsafe_allow_html=True)
    log_placeholder = st.empty()

    if "running" not in st.session_state:
        st.session_state.running = False
    if run_button:
        st.session_state.running = True
    if stop_button:
        st.session_state.running = False

    if st.session_state.running:
        engine = load_engine(model_path, conf)

        if source_type == "LIVE_WEBCAM":
            cap = cv2.VideoCapture(0)
        else:
            if video_file is None:
                st.warning("SYSTEM ERROR: NO VIDEO FEED DETECTED.")
                return
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)

        frame_idx = 0
        while cap.isOpened() and st.session_state.running:
            ok, frame = cap.read()
            if not ok:
                break

            statuses = engine.process_frame(frame, frame_idx)
            
            # Apply intense deep purple/cyan quantum tint to the video
            overlay = frame.copy()
            # Draw a purple overlay
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (200, 0, 100), -1) 
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            frame = draw_overlays(frame, engine.last_raw_detections)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            compliant = sum(1 for s in statuses if s.compliant)
            total = len(statuses)
            violations = total - compliant
            rate = (compliant / total * 100) if total else 0.0

            m1.metric("ENTITIES", f"{total}")
            m2.metric("COMPLIANT", f"{compliant}")
            m3.metric("BREACHES", f"{violations}")
            m4.metric("SAFETY_IX", f"{rate:.0f}%")

            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            try:
                log_df = pd.read_csv(config.VIOLATION_LOG_CSV)
                log_placeholder.dataframe(log_df.iloc[::-1].head(50), use_container_width=True)
            except FileNotFoundError:
                pass

            frame_idx += 1
            time.sleep(0.01)

        cap.release()
    else:
        st.info("AWAITING INITIALIZATION... SELECT FEED AND ENGAGE.")
        try:
            log_df = pd.read_csv(config.VIOLATION_LOG_CSV)
            log_placeholder.dataframe(log_df.iloc[::-1].head(50), use_container_width=True)
        except FileNotFoundError:
            log_placeholder.write("NO VIOLATIONS DETECTED.")


if __name__ == "__main__":
    main()
