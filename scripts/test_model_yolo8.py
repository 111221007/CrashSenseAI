import os
import cv2
import numpy as np
from pathlib import Path
import warnings
import torch
from ultralytics import YOLO

# ========= SUPPRESS WARNINGS =========
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ========= SETUP PATHS =========
model_path = Path(r"C:\Users\cmpor\PycharmProjects\EdgeAI_Benchmark_Project\accident_detection\model\best.pt")  # <== Change to yolov8s.pt
video_path = Path(r"C:\Users\cmpor\PycharmProjects\EdgeAI_Benchmark_Project\accident_detection\data\input\test_videos\test (2).mp4")
save_dir = Path(r"C:/Users/cmpor/PycharmProjects/EdgeAI_Benchmark_Project/accident_detection/data/output/accident_frames")

# Create save directory if it doesn't exist
save_dir.mkdir(parents=True, exist_ok=True)

# ========= SAFETY CHECKS =========
if not model_path.exists():
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")

if not video_path.exists():
    raise FileNotFoundError(f"❌ Video file not found: {video_path}")

# ========= LOAD YOLOV8 MODEL =========
model = YOLO(str(model_path))  # Load YOLOv8 model

# ========= OPEN VIDEO =========
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise IOError(f"❌ Cannot open video file: {video_path}")

# Get original video FPS
video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / video_fps)

print(f"✅ Starting accident detection with YOLOv8 at {video_fps:.2f} FPS... Press 'q' or close window to exit.")

# ========= CONFIGURABLE PARAMETERS =========
CONFIDENCE_THRESHOLD = 0.5  # For object detection confidence
CONSECUTIVE_ACCIDENT_FRAMES = 3

accident_streak = 0
frame_count = 0
accident_frame_count = 0

window_name = "Accident Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# ========= PROCESS VIDEO =========
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No more frames or video closed.")
        break

    # Check if window is still open BEFORE doing anything
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        print("❌ Window closed by user.")
        break

    frame_count += 1
    h, w, _ = frame.shape

    # Inference
    results = model.predict(frame, imgsz=640, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    detections = results.boxes

    detected_accident = False

    for box in detections:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label_name = model.names[cls_id]

        # If class is "Accident" or similar (depends on training labels)
        if "accident" in label_name.lower() and conf > CONFIDENCE_THRESHOLD:
            detected_accident = True
            break

    # Smoothing logic
    if detected_accident:
        accident_streak += 1
    else:
        accident_streak = 0

    if accident_streak >= CONSECUTIVE_ACCIDENT_FRAMES:
        label = "Accident Detected!"
        color = (0, 0, 255)  # Red

        if accident_frame_count < 5:  # ✅ Save only first 5 accident frames
            accident_frame_count += 1
            save_path = save_dir / f"accident_frame_{accident_frame_count:04d}.jpg"
            cv2.imwrite(str(save_path), frame)
            print(f"💾 Saved accident frame: {save_path}")
        else:
            print(f"⚠️ Already saved 5 accident frames. Skipping save.")

        accident_streak = 0
    else:
        label = "No Accident"
        color = (0, 255, 0)

    # Draw frame
    thickness = 8
    cv2.rectangle(frame, (0, 0), (w, h), color, thickness)
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Show frame
    cv2.imshow(window_name, frame)

    # Wait and check for 'q' press
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        print("❌ 'q' pressed. Exiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print(f"✅ Video inference complete. {accident_frame_count} accident frames saved.")
