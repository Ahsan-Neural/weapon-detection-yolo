import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

st.set_page_config(
    page_title="Weapon Detection System",
    page_icon="assets/favicon.png" if os.path.exists("assets/favicon.png") else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .main-title { font-size: 2rem; font-weight: 700; color: #e6edf3; letter-spacing: -0.5px; margin-bottom: 0; }
    .sub-title { font-size: 0.95rem; color: #8b949e; margin-top: 4px; margin-bottom: 24px; }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #e6edf3; }
    .detection-tag {
        display: inline-block;
        background-color: #1f2d3d;
        border: 1px solid #1f6feb;
        color: #58a6ff;
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 0.85rem;
        margin: 3px 3px 3px 0;
    }
    .footer-link { color: #58a6ff; text-decoration: none; font-size: 0.85rem; }
    .section-header { font-size: 1rem; font-weight: 600; color: #e6edf3; margin-bottom: 12px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }
    div[data-testid="stImage"] img { border-radius: 6px; border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

with st.sidebar:
    st.markdown('<p class="section-header">Configuration</p>', unsafe_allow_html=True)
    confidence = st.slider("Confidence Threshold", min_value=0.10, max_value=0.95, value=0.50, step=0.05)
    
    st.markdown('<p class="section-header" style="margin-top:24px;">Input Source</p>', unsafe_allow_html=True)
    input_mode = st.radio("", ["Upload Image", "Webcam Snapshot"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        '<p class="metric-label">Model</p><p style="color:#e6edf3;font-size:0.9rem;">YOLOv8n — Weapon Detection</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="metric-label" style="margin-top:12px;">Classes</p>'
        '<p style="color:#e6edf3;font-size:0.9rem;">Knife &nbsp;|&nbsp; Pistol &nbsp;|&nbsp; Rifle</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
        '<p class="metric-label">Built by</p>'
        '<a class="footer-link" href="https://www.kaggle.com/ahsanneural" target="_blank">Ahsan Neural — Kaggle Profile</a>',
        unsafe_allow_html=True
    )

st.markdown('<p class="main-title">Weapon Detection System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Real-time weapon identification using YOLOv8 — Knife, Pistol, Rifle</p>',
    unsafe_allow_html=True
)

CLASS_COLORS = {
    "knife":  (255, 100, 100),
    "pistol": (100, 180, 255),
    "rifle":  (100, 220, 140),
}

def run_detection(image_pil, conf_threshold):
    img_array = np.array(image_pil.convert("RGB"))
    results = model.predict(img_array, conf=conf_threshold, verbose=False)
    result = results[0]
    detections = []
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(label, (200, 200, 200))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label.capitalize()} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img_bgr, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        detections.append({"class": label.capitalize(), "confidence": round(conf, 2), "bbox": (x1, y1, x2, y2)})
    img_result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_result), detections

uploaded_image = None

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)
else:
    cam_image = st.camera_input("Take a photo")
    if cam_image:
        uploaded_image = Image.open(cam_image)

if uploaded_image:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<p class="section-header">Input Image</p>', unsafe_allow_html=True)
        st.image(uploaded_image, use_container_width=True)

    result_image, detections = run_detection(uploaded_image, confidence)

    with col2:
        st.markdown('<p class="section-header">Detection Output</p>', unsafe_allow_html=True)
        st.image(result_image, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Detection Summary</p>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><p class="metric-label">Total Detections</p><p class="metric-value">{len(detections)}</p></div>', unsafe_allow_html=True)
    with m2:
        avg_conf = round(sum(d["confidence"] for d in detections) / len(detections), 2) if detections else 0.0
        st.markdown(f'<div class="metric-card"><p class="metric-label">Average Confidence</p><p class="metric-value">{avg_conf}</p></div>', unsafe_allow_html=True)
    with m3:
        classes_found = list(set(d["class"] for d in detections))
        classes_str = ", ".join(classes_found) if classes_found else "None"
        st.markdown(f'<div class="metric-card"><p class="metric-label">Classes Detected</p><p class="metric-value" style="font-size:1.1rem;">{classes_str}</p></div>', unsafe_allow_html=True)

    if detections:
        st.markdown('<p class="section-header" style="margin-top:16px;">Individual Detections</p>', unsafe_allow_html=True)
        tags_html = ""
        for d in detections:
            tags_html += f'<span class="detection-tag">{d["class"]} — {d["confidence"]}</span>'
        st.markdown(tags_html, unsafe_allow_html=True)
    else:
        st.info("No weapons detected above the confidence threshold.")
else:
    st.markdown(
        '<div style="border: 1px dashed #30363d; border-radius: 8px; padding: 60px; text-align:center; color:#8b949e;">'
        'Upload an image or take a webcam snapshot to begin detection.'
        '</div>',
        unsafe_allow_html=True
    )
