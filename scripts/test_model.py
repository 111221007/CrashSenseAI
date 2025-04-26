import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import warnings

# ========= SUPPRESS WARNINGS =========
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ========= SETUP PATHS =========
model_path = Path(r"/accident_detection/model/crash_detect.tflite")
video_path = Path(r"C:/Users/cmpor/PycharmProjects/EdgeAI_Benchmark_Project/accident_detection/data/input/video0.mp4")
save_dir = Path(r"/accident_detection/data/output/accident_frames")

# Create save directory if it doesn't exist
save_dir.mkdir(parents=True, exist_ok=True)

# ========= SAFETY CHECKS =========
if not model_path.exists():
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")

if not video_path.exists():
    raise FileNotFoundError(f"❌ Video file not found: {video_path}")

# ========= LOAD TFLITE MODEL =========
interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, channels = input_details[0]['shape']

# ========= OPEN VIDEO =========
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise IOError(f"❌ Cannot open video file: {video_path}")

# Get original video FPS
video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / video_fps)

print(f"✅ Starting accident detection at {video_fps:.2f} FPS... Press 'q' or close window to exit.")

# ========= CONFIGURABLE PARAMETERS =========
CONFIDENCE_THRESHOLD = 0.8
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

    # Preprocess frame
    img = cv2.resize(frame, (width, height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0][0]

    # Smoothing logic
    if prediction > CONFIDENCE_THRESHOLD:
        accident_streak += 1
    else:
        accident_streak = 0

    if accident_streak >= CONSECUTIVE_ACCIDENT_FRAMES:
        label = "Accident :"
        color = (0, 0, 255)  # Red

        # Save accident frame
        accident_frame_count += 1
        save_path = save_dir / f"accident_frame_{accident_frame_count:04d}.jpg"
        cv2.imwrite(str(save_path), frame)
        print(f"💾 Saved accident frame: {save_path}")

        accident_streak = 0
    else:
        label = "No Accident :"
        color = (0, 255, 0)

    # Draw frame
    thickness = 8
    cv2.rectangle(frame, (0, 0), (w, h), color, thickness)
    cv2.putText(frame, f"{label} ({prediction:.2f})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

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
