# app/adas_infer.py
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from models.lane_model import LaneSegNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load models ---
# Lane CNN
lane_model = LaneSegNet(pretrained=False).to(DEVICE)
lane_model.load_state_dict(torch.load("weights/lane_seg_best.pth", map_location=DEVICE))
lane_model.eval()

# YOLO
yolo_model = YOLO("weights/yolo_best.pt")  # replace with yolov5s.pt if needed

# ADAS relevant classes
ADAS_IDX = [0, 1, 2, 3, 5, 7, 9, 11]  # person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign

# --- Helpers ---
def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 352)) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    return tensor

def postprocess_mask(logits, orig_shape):
    mask = torch.sigmoid(logits).detach().cpu().squeeze().numpy()
    mask = (mask > 0.4).astype(np.uint8) * 255  # threshold
    mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
    return mask

# --- Main inference loop ---
def run_inference(video_path, output_path="adas_output.mp4"):
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Lane prediction
        inp = preprocess_frame(frame)
        with torch.no_grad():
            logits = lane_model(inp)
        mask = postprocess_mask(logits, frame.shape)

        # Overlay lane mask
        overlay = frame.copy()
        overlay[mask > 0] = (0, 255, 0)  # green lanes
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # YOLO detection
        results = yolo_model.predict(frame, conf=0.35, imgsz=640, classes=ADAS_IDX, device=DEVICE, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{yolo_model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Output saved at {output_path}")

if __name__ == "__main__":
    run_inference("data/sample_videos/drive.mp4")
