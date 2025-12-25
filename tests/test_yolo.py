# ---- FIX FOR LINUX-SAVED YOLO WEIGHTS ON WINDOWS ----
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

import cv2
import torch
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = ROOT / "yolov5"
sys.path.insert(0, str(YOLO_DIR))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox

WEIGHTS = ROOT / "weights/yolo_best.pt"
TEST_VIDEO = ROOT / "data/sample_videos/video1.mp4"


def test_yolo():
    print("Loading YOLO model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(str(WEIGHTS), device=device)

    # ---------------------
    # LOAD FIRST FRAME OF VIDEO
    # ---------------------
    cap = cv2.VideoCapture(str(TEST_VIDEO))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open test video: {TEST_VIDEO}")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise ValueError("Failed to read a frame from test video.")

    print("Video frame loaded for inference.")

    # ---------------------
    # RUN YOLO INFERENCE
    # ---------------------
    img = letterbox(frame, 640, stride=model.stride)[0]
    # BGR→RGB, HWC→CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    # ✅ make array contiguous to remove negative strides
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)

    det = non_max_suppression(pred)[0]
    print("YOLO inference OK. Det count:", len(det))


if __name__ == "__main__":
    test_yolo()
    print("TEST PASSED ✔ YOLO integrated correctly.")
