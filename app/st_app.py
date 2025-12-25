import os
import cv2
import time
import torch
import tempfile
import numpy as np
import streamlit as st
from pathlib import Path

# ---------------- PATH FIXES ----------------
import sys
import pathlib
pathlib.PosixPath = pathlib.WindowsPath  # Fix YOLO weights saved on Linux

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
YOLO_DIR = ROOT / "yolov5"
ADAS_DIR = ROOT / "adas_models"

sys.path.insert(0, str(YOLO_DIR))
sys.path.append(str(ROOT))

# YOLO imports
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# Lane math & TTC modules
from app.utils.lane_math import (
    get_lane_points, fit_lane_poly, lane_curvature,
    lateral_offset_pixels, drift_speed, time_to_lane_departure
)

from app.utils.ttc_math import (
    estimate_distance_px, relative_speed, time_to_collision
)

# Lane model
from adas_models.lane_model import LaneSegNet


# ---------------------------------------------------------------
# Streamlit UI CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="ADAS Enhanced", page_icon="🚗", layout="wide")
st.title("🚗 ADAS – Lane Segmentation + Objects + Offset + TLD + TTC")

st.markdown("""
<style>
.ok {color:#0a0;}
.warn {color:#e69500;}
.danger {color:#d00;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# MODEL LOADERS
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_lane_model(weights_path: str, device: str = "cuda"):
    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = LaneSegNet(pretrained=False).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str, device_choice: str = "cuda"):
    if device_choice == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    weights_path = weights_path.replace("\\", "/")
    model = DetectMultiBackend(weights_path, device=device)
    return model


# ---------------------------------------------------------------
# INFERENCE HELPERS
# ---------------------------------------------------------------
def run_yolo_inference(model, frame_bgr, imgsz, conf, iou, device):
    img = letterbox(frame_bgr, imgsz, stride=model.stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB → CHW
    img = np.ascontiguousarray(img)

    im = torch.from_numpy(img).to(device)
    im = im.float() / 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    with torch.no_grad():
        pred = model(im)
    pred = non_max_suppression(pred, conf_thres=conf, iou_thres=iou)[0]

    if len(pred):
        pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], frame_bgr.shape).round()

    return pred


def preprocess_lane(frame, size, device):
    h, w = size
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    x = x.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    return x


def post_mask(logits, orig_wh, thr):
    prob = torch.sigmoid(logits).detach().cpu().squeeze().numpy()
    mask = (prob > thr).astype(np.uint8) * 255
    W, H = orig_wh
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask


def overlay_mask(frame, mask):
    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 0)
    return cv2.addWeighted(frame, 1.0, overlay, 0.35, 0)


# ---------------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

weights_lane = st.sidebar.text_input("Lane Model (.pth)", "weights/lane_seg_best.pth")
weights_yolo = st.sidebar.text_input("YOLOv5 Model (.pt)", "weights/yolo_best.pt")

conf = st.sidebar.slider("YOLO Confidence", 0.05, 0.9, 0.35)
iou_thr = st.sidebar.slider("YOLO IoU Threshold", 0.1, 0.9, 0.45)
mask_thr = st.sidebar.slider("Lane Mask Threshold", 0.1, 0.9, 0.40)

lane_h, lane_w = st.sidebar.selectbox(
    "LaneNet Size", [(352, 640), (320, 576), (288, 512)], index=0
)

imgsz_yolo = st.sidebar.selectbox("YOLO Inference Size", [640, 960, 1280], index=0)

device_choice = st.sidebar.selectbox("Compute Device", ["cuda", "cpu"], index=0)
device = device_choice if (device_choice == "cuda" and torch.cuda.is_available()) else "cpu"
st.sidebar.write(f"Using device: **{device}**")

preview_n = st.sidebar.slider("Preview Frames", 0, 10, 4)
save_every_n = st.sidebar.slider("Draw Every Nth Frame", 1, 5, 1)


# ---------------------------------------------------------------
# MAIN UI LAYOUT
# ---------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("📼 Input Video")
    up = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    use_sample = st.checkbox("Use sample", value=not up)

with right:
    st.subheader("🧠 Model Paths")
    st.write(weights_lane)
    st.write(weights_yolo)

st.markdown("---")

run_btn = st.button("🚀 Run ADAS Inference", type="primary")

# ---------------------------------------------------------------
# RUN INFERENCE
# ---------------------------------------------------------------
if run_btn:

    # Load video file
    if up:
        tmp = tempfile.mkdtemp()
        video_path = os.path.join(tmp, up.name)
        with open(video_path, "wb") as f:
            f.write(up.read())
    else:
        video_path = "data/sample_videos/drive.mp4"
        if not os.path.exists(video_path):
            st.error("Sample video missing.")
            st.stop()

    # Load models
    with st.spinner("Loading models..."):
        lane_model = load_lane_model(weights_lane, device)
        yolo_model = load_yolo_model(weights_yolo, device)

    st.success("Models loaded successfully.")

    # Video setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Failed to open video.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_tmp = tempfile.mkdtemp()
    out_path = os.path.join(out_tmp, "adas_output.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    st.session_state.prev_offset = 0
    st.session_state.prev_heights = {}

    pbar = st.progress(0)
    previews = []
    idx = 0

    # -----------------------------------------------------------
    # ADAS CLASS FILTERING (FINAL FIXED VERSION)
    # -----------------------------------------------------------
    ADAS_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle',
        'bus', 'truck', 'traffic light', 'stop sign'
    ]

    if isinstance(yolo_model.names, dict):
        name_to_id = {v: k for k, v in yolo_model.names.items()}
    else:
        name_to_id = {name: i for i, name in enumerate(yolo_model.names)}

    adas_ids = [name_to_id[n] for n in ADAS_CLASSES if n in name_to_id]

    # -----------------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        # -------- LANE SEGMENTATION --------
        inp = preprocess_lane(frame, (lane_h, lane_w), device)
        with torch.no_grad():
            logits = lane_model(inp)

        mask = post_mask(logits, (W, H), mask_thr)
        vis = overlay_mask(frame, mask)

        # -------- LANE GEOMETRY --------
        xs, ys, w_img, h_img = get_lane_points(mask)
        fit, ploty, fitx = fit_lane_poly(xs, ys, h_img)

        if fit is not None:
            offset = lateral_offset_pixels(fit, w_img, h_img)
            drift = drift_speed(st.session_state.prev_offset, offset, fps)
            st.session_state.prev_offset = offset
            tld = time_to_lane_departure(offset, drift)

            cv2.putText(vis, f"Offset: {offset:.1f}px", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(vis, f"Drift: {drift:.1f}px/s", (25, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if tld is not None and tld < 3.0:
                cv2.putText(vis, f"TLD WARNING: {tld:.2f}s", (25, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        # -------- YOLO / TTC --------
        if idx % save_every_n == 0:
            preds = run_yolo_inference(yolo_model, frame, imgsz_yolo, conf, iou_thr, device)

            for det in preds:
                cls = int(det[5])
                if cls not in adas_ids:
                    continue

                x1, y1, x2, y2 = map(int, det[:4])
                h_curr = y2 - y1
                conf_val = float(det[4])
                label = f"{yolo_model.names[cls]} {conf_val:.2f}"

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if cls not in st.session_state.prev_heights:
                    st.session_state.prev_heights[cls] = h_curr
                else:
                    h_prev = st.session_state.prev_heights[cls]
                    ttc = time_to_collision(h_prev, h_curr, fps)
                    st.session_state.prev_heights[cls] = h_curr

                    if ttc is not None and ttc < 2.5:
                        cv2.putText(vis, f"TTC {ttc:.2f}s", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 3)

        # Write frame
        writer.write(vis)

        # Save preview frames
        if len(previews) < preview_n:
            previews.append(vis[..., ::-1])

        pbar.progress(min(idx / frames_total, 1.0))

    cap.release()
    writer.release()

    st.success("✔ Processing Completed")

    # Show previews
    if previews:
        st.subheader("🔎 Sample Previews")
        cols = st.columns(2)
        for i, img in enumerate(previews):
            cols[i % 2].image(img, use_column_width=True)

    with open(out_path, "rb") as f:
        st.download_button(
            "⬇ Download ADAS Output",
            f.read(),
            "adas_output.mp4",
            "video/mp4"
        )
