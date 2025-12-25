# tests/test_lane.py
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------- PATHS ----------------
ROOT = Path(__file__).resolve().parents[1]  # C:\Minor Project-ADAS

# Make project and app modules importable
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "adas_models"))

from adas_models.lane_model import LaneSegNet  # your U-Net

WEIGHTS = ROOT / "weights" / "lane_seg_best.pth"

# Use the same sample video you used in YOLO test
TEST_VIDEO = ROOT / "data" / "sample_videos" / "video1.mp4"
# If your file is named differently, e.g. drive.mp4, change accordingly:
# TEST_VIDEO = ROOT / "data" / "sample_videos" / "drive.mp4"


def test_lane():
    print("Loading LaneSeg model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LaneSegNet(pretrained=False).to(device)
    state = torch.load(WEIGHTS, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ---- read one frame from video instead of imread(video.mp4) ----
    cap = cv2.VideoCapture(str(TEST_VIDEO))
    assert cap.isOpened(), f"Could not open test video: {TEST_VIDEO}"

    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None, "Could not read a frame from the test video."

    # ---- preprocess exactly like in your app (resize to 640x352, normalize) ----
    frame_resized = cv2.resize(frame, (640, 352))  # (W,H)
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,352,640]

    with torch.no_grad():
        logits = model(x)

    print("Model output shape:", list(logits.shape))
    assert logits.ndim == 4, "Expected 4D output [B,C,H,W]"
    assert logits.shape[1] == 1, "Expected 1-channel mask output."
    assert logits.shape[2] == 352 and logits.shape[3] == 640, "Unexpected mask spatial size."

    print("LaneSeg test passed ✔")


if __name__ == "__main__":
    test_lane()
