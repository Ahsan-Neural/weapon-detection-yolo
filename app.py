import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import time
import tempfile
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentinel — Weapon Detection System",
    page_icon="assets/favicon.png" if os.path.exists("assets/favicon.png") else "🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------
# FUTURISTIC THEME
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 10%, #0d1424 0%, #05060d 55%, #030308 100%);
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e18 0%, #05070d 100%);
    border-right: 1px solid rgba(0,245,255,0.15);
}

.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.4rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00f5ff, #8b5cf6 60%, #ff2e88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
    margin-bottom: 0;
    text-shadow: 0 0 30px rgba(0,245,255,0.25);
}
.hero-sub {
    font-size: 1rem;
    color: #7d8aa3;
    margin-top: 2px;
    margin-bottom: 22px;
    letter-spacing: 0.3px;
}

.status-row { display:flex; gap:10px; margin-bottom:22px; flex-wrap:wrap; }
.status-pill {
    font-family:'Rajdhani',sans-serif; font-size:0.78rem; font-weight:600;
    padding:5px 14px; border-radius:20px; letter-spacing:0.06em; text-transform:uppercase;
    border:1px solid rgba(0,245,255,0.3); color:#00f5ff; background:rgba(0,245,255,0.06);
}
.status-pill.violet { color:#c084fc; border-color:rgba(192,132,252,0.35); background:rgba(192,132,252,0.06); }
.status-pill.pink { color:#ff6ec7; border-color:rgba(255,110,199,0.35); background:rgba(255,110,199,0.06); }

.glass-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(0,245,255,0.14);
    border-radius: 14px;
    padding: 18px 22px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 25px rgba(0,245,255,0.04), inset 0 0 40px rgba(139,92,246,0.02);
    margin-bottom: 14px;
}

.metric-label { font-size:0.72rem; color:#7d8aa3; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; }
.metric-value { font-family:'Orbitron',sans-serif; font-size:1.7rem; font-weight:700; color:#e8f6ff; margin-top:4px; }
.metric-value.cyan { color:#00f5ff; text-shadow:0 0 12px rgba(0,245,255,0.4); }
.metric-value.violet { color:#c084fc; text-shadow:0 0 12px rgba(192,132,252,0.4); }
.metric-value.pink { color:#ff6ec7; text-shadow:0 0 12px rgba(255,110,199,0.4); }

.section-header {
    font-family:'Orbitron', sans-serif; font-size:0.95rem; font-weight:700; color:#e8f6ff;
    margin: 6px 0 14px 0; padding-bottom:8px; letter-spacing:0.04em;
    border-bottom: 1px solid rgba(0,245,255,0.15);
}

.detection-tag {
    display:inline-block; font-weight:600; font-size:0.85rem;
    padding:5px 13px; margin:3px 4px 3px 0; border-radius:20px;
    background: rgba(0,245,255,0.08); border:1px solid rgba(0,245,255,0.35); color:#00f5ff;
}
.detection-tag.knife { border-color: rgba(255,110,199,0.4); color:#ff6ec7; background:rgba(255,110,199,0.08); }
.detection-tag.pistol { border-color: rgba(0,245,255,0.4); color:#00f5ff; background:rgba(0,245,255,0.08); }
.detection-tag.rifle { border-color: rgba(52,211,153,0.4); color:#34d399; background:rgba(52,211,153,0.08); }

div[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid rgba(0,245,255,0.2);
    box-shadow: 0 0 30px rgba(0,245,255,0.06);
}

.empty-state {
    border: 1px dashed rgba(0,245,255,0.25);
    border-radius: 14px;
    padding: 70px 30px;
    text-align: center;
    color: #7d8aa3;
    background: rgba(0,245,255,0.02);
}

.live-dot {
    height:10px; width:10px; border-radius:50%; background:#ff2e4d; display:inline-block;
    margin-right:6px; box-shadow:0 0 10px #ff2e4d; animation: pulse 1.4s infinite;
}
@keyframes pulse { 0%{opacity:1;} 50%{opacity:0.35;} 100%{opacity:1;} }

.footer-link { color:#00f5ff; text-decoration:none; font-size:0.85rem; }

[data-baseweb="slider"] div[role="slider"] { background-color: #00f5ff !important; }
.stButton > button {
    background: linear-gradient(90deg, #00f5ff22, #8b5cf622);
    border: 1px solid rgba(0,245,255,0.4); color:#e8f6ff; border-radius:8px; font-weight:600;
}
.stButton > button:hover { border-color:#00f5ff; color:#00f5ff; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

CLASS_COLORS_BGR = {
    "knife":  (139, 110, 255),
    "pistol": (255, 245, 0),
    "rifle":  (153, 211, 52),
}
CLASS_CSS = {"knife": "knife", "pistol": "pistol", "rifle": "rifle"}

def draw_boxes(img_bgr, result, model_names):
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = model_names[cls_id].lower()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS_BGR.get(label, (200, 200, 200))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label.upper()} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img_bgr, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        detections.append({"class": label.capitalize(), "confidence": round(conf, 2)})
    return img_bgr, detections

def run_detection_on_pil(image_pil, conf_threshold):
    img_array = np.array(image_pil.convert("RGB"))
    results = model.predict(img_array, conf=conf_threshold, verbose=False)
    result = results[0]
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_bgr, detections = draw_boxes(img_bgr, result, model.names)
    img_result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_result), detections

def render_detection_tags(detections):
    if not detections:
        st.info("No weapons detected above the confidence threshold.")
        return
    tags_html = ""
    for d in detections:
        css_class = CLASS_CSS.get(d["class"].lower(), "")
        tags_html += f'<span class="detection-tag {css_class}">{d["class"]} — {d["confidence"]}</span>'
    st.markdown(tags_html, unsafe_allow_html=True)

def render_metrics(detections):
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="glass-card"><p class="metric-label">Total Detections</p>'
                    f'<p class="metric-value cyan">{len(detections)}</p></div>', unsafe_allow_html=True)
    with m2:
        avg_conf = round(sum(d["confidence"] for d in detections) / len(detections), 2) if detections else 0.0
        st.markdown(f'<div class="glass-card"><p class="metric-label">Average Confidence</p>'
                    f'<p class="metric-value violet">{avg_conf}</p></div>', unsafe_allow_html=True)
    with m3:
        classes_found = sorted(set(d["class"] for d in detections))
        classes_str = ", ".join(classes_found) if classes_found else "None"
        st.markdown(f'<div class="glass-card"><p class="metric-label">Classes Detected</p>'
                    f'<p class="metric-value pink" style="font-size:1.15rem;">{classes_str}</p></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="section-header">Configuration</p>', unsafe_allow_html=True)
    confidence = st.slider("Confidence Threshold", min_value=0.10, max_value=0.95, value=0.50, step=0.05)

    st.markdown('<p class="section-header" style="margin-top:20px;">Detection Mode</p>', unsafe_allow_html=True)
    input_mode = st.radio(
        "",
        ["Upload Image", "Live Webcam", "Upload Video"],
        label_visibility="collapsed"
    )

    frame_stride = 1
    if input_mode == "Upload Video":
        st.markdown('<p class="section-header" style="margin-top:20px;">Video Scan Settings</p>', unsafe_allow_html=True)
        frame_stride = st.slider("Analyze every Nth frame", min_value=1, max_value=30, value=5,
                                  help="Lower = more thorough but slower. Higher = faster but may skip brief detections.")

    st.markdown("---")
    st.markdown('<p class="metric-label">Model</p><p style="color:#e8f6ff;font-size:0.9rem;">YOLOv8n — Weapon Detection</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="metric-label" style="margin-top:12px;">Classes</p>'
        '<p style="color:#e8f6ff;font-size:0.9rem;">Knife &nbsp;|&nbsp; Pistol &nbsp;|&nbsp; Rifle</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
        '<p class="metric-label">Built by</p>'
        '<a class="footer-link" href="https://www.kaggle.com/ahsanneural" target="_blank">Ahsan Neural — Kaggle Profile</a>',
        unsafe_allow_html=True
    )

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown('<p class="hero-title">SENTINEL // WEAPON DETECTION</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Real-time computer-vision threat detection powered by YOLOv8 — image, live feed, and video analysis.</p>', unsafe_allow_html=True)
st.markdown(
    '<div class="status-row">'
    '<span class="status-pill">● Model Online</span>'
    '<span class="status-pill violet">3 Classes Tracked</span>'
    f'<span class="status-pill pink">Confidence ≥ {confidence:.2f}</span>'
    '</div>',
    unsafe_allow_html=True
)

# ----------------------------------------------------------------------------
# MODE 1 — IMAGE UPLOAD
# ----------------------------------------------------------------------------
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<p class="section-header">Input Image</p>', unsafe_allow_html=True)
            st.image(uploaded_image, use_container_width=True)

        result_image, detections = run_detection_on_pil(uploaded_image, confidence)

        with col2:
            st.markdown('<p class="section-header">Detection Output</p>', unsafe_allow_html=True)
            st.image(result_image, use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="section-header">Detection Summary</p>', unsafe_allow_html=True)
        render_metrics(detections)

        if detections:
            st.markdown('<p class="section-header" style="margin-top:10px;">Individual Detections</p>', unsafe_allow_html=True)
            render_detection_tags(detections)
    else:
        st.markdown('<div class="empty-state">Upload an image to begin detection.</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# MODE 2 — LIVE WEBCAM (real-time streaming detection)
# ----------------------------------------------------------------------------
elif input_mode == "Live Webcam":
    st.markdown(
        '<div class="status-row"><span class="status-pill pink"><span class="live-dot"></span>LIVE FEED</span></div>',
        unsafe_allow_html=True
    )
    st.markdown('<p class="section-header">Real-Time Detection Stream</p>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card" style="font-size:0.9rem;color:#7d8aa3;">'
                'Click <b>START</b> below to open your camera. Every frame is scanned live — '
                'bounding boxes are drawn instantly whenever a weapon enters the frame.</div>',
                unsafe_allow_html=True)

    class YOLOLiveProcessor(VideoProcessorBase):
        def __init__(self):
            self.confidence = confidence
            self.detection_count = 0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model.predict(img, conf=self.confidence, verbose=False)
            result = results[0]
            img, detections = draw_boxes(img, result, model.names)
            self.detection_count = len(detections)
            if detections:
                cv2.rectangle(img, (0, 0), (img.shape[1], 4), (0, 0, 255), -1)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    webrtc_ctx = webrtc_streamer(
        key="weapon-live-detect",
        video_processor_factory=YOLOLiveProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence = confidence

    st.markdown(
        '<div class="glass-card" style="margin-top:14px;font-size:0.85rem;color:#7d8aa3;">'
        'Note: live streaming requires camera permission in your browser. Processing speed depends on '
        'your device/CPU — lower the confidence panel width or run on a GPU-backed host for smoother FPS.'
        '</div>', unsafe_allow_html=True
    )

# ----------------------------------------------------------------------------
# MODE 3 — VIDEO UPLOAD (frame-by-frame scan)
# ----------------------------------------------------------------------------
elif input_mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1])
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.markdown('<p class="section-header">Scanning Video</p>', unsafe_allow_html=True)
        progress = st.progress(0, text="Analyzing frames...")

        flagged_frames = []      # frames that had at least one detection
        all_scanned = 0
        total_detections = 0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_stride == 0:
                results = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf=confidence, verbose=False)
                result = results[0]
                annotated, detections = draw_boxes(frame.copy(), result, model.names)
                all_scanned += 1
                if detections:
                    total_detections += len(detections)
                    timestamp_sec = frame_idx / fps
                    flagged_frames.append({
                        "frame_index": frame_idx,
                        "timestamp": timestamp_sec,
                        "image": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        "detections": detections,
                    })
                if total_frames:
                    progress.progress(min(frame_idx / total_frames, 1.0), text=f"Analyzing frame {frame_idx}/{total_frames}...")
            frame_idx += 1

        cap.release()
        os.unlink(video_path)
        progress.empty()

        st.markdown('<p class="section-header">Scan Summary</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="glass-card"><p class="metric-label">Frames Scanned</p>'
                        f'<p class="metric-value cyan">{all_scanned}</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="glass-card"><p class="metric-label">Frames With Detections</p>'
                        f'<p class="metric-value violet">{len(flagged_frames)}</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="glass-card"><p class="metric-label">Total Detections</p>'
                        f'<p class="metric-value pink">{total_detections}</p></div>', unsafe_allow_html=True)

        if flagged_frames:
            st.markdown('<p class="section-header" style="margin-top:16px;">Flagged Frames</p>', unsafe_allow_html=True)
            options = [f"Frame {f['frame_index']}  ·  {f['timestamp']:.2f}s  ·  {len(f['detections'])} detection(s)" for f in flagged_frames]
            choice = st.selectbox("Select a flagged frame to inspect", options)
            selected = flagged_frames[options.index(choice)]

            colA, colB = st.columns([2, 1], gap="large")
            with colA:
                st.image(selected["image"], use_container_width=True)
            with colB:
                st.markdown(f'<p class="section-header">Frame {selected["frame_index"]} Details</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="glass-card"><p class="metric-label">Timestamp</p>'
                            f'<p class="metric-value cyan" style="font-size:1.2rem;">{selected["timestamp"]:.2f}s</p></div>', unsafe_allow_html=True)
                render_detection_tags(selected["detections"])

            st.markdown('<p class="section-header" style="margin-top:20px;">Detection Timeline</p>', unsafe_allow_html=True)
            timeline_html = ""
            for f in flagged_frames:
                timeline_html += (f'<span class="detection-tag pistol" style="cursor:default;">'
                                   f'{f["timestamp"]:.1f}s ({len(f["detections"])})</span>')
            st.markdown(timeline_html, unsafe_allow_html=True)
        else:
            st.info("No weapons were detected in this video at the current confidence threshold.")
    else:
        st.markdown('<div class="empty-state">Upload a video file to scan it frame by frame for weapons.</div>', unsafe_allow_html=True)
