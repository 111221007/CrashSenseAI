import os
import cv2
import numpy as np
import threading
from pathlib import Path
import warnings
import torch
import datetime
from tkinter import Tk, Label, Button, Listbox, SINGLE, END, messagebox, Canvas
from PIL import Image, ImageTk
from ultralytics import YOLO

# ========= CONFIGURATIONS =========
MODEL_PATH = Path(r"C:\Users\cmpor\PycharmProjects\EdgeAI_Benchmark_Project\accident_detection\model\best.pt")
VIDEOS_DIR = Path(r"C:\Users\cmpor\PycharmProjects\EdgeAI_Benchmark_Project\accident_detection\data\input\test_videos")
SAVE_DIR = Path(r"C:/Users/cmpor/PycharmProjects/EdgeAI_Benchmark_Project/accident_detection/data/output/accident_frames")

CONFIDENCE_THRESHOLD = 0.5
CONSECUTIVE_ACCIDENT_FRAMES = 3
MAX_ACCIDENT_FRAMES = 5

# ========= GUI VIDEO SETTINGS =========
VIDEO_CANVAS_WIDTH = 600
VIDEO_CANVAS_HEIGHT = 360

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ========= GLOBAL =========
stop_detection = False

# ========= DETECTION FUNCTION =========
def detect_accidents(video_path, status_label, start_button, accident_label, video_canvas):
    global stop_detection
    try:
        model = YOLO(str(MODEL_PATH))
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            messagebox.showerror("Error", "❌ Cannot open selected video.")
            reset_ui(status_label, start_button, accident_label)
            return

        video_filename = Path(video_path).stem
        video_save_dir = SAVE_DIR / video_filename
        video_save_dir.mkdir(parents=True, exist_ok=True)

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = int(1000 / video_fps) if video_fps > 0 else 30

        accident_streak = 0
        accident_frame_count = 0

        status_label.config(text="Status: Running 🚀")
        accident_label.config(text="No Accident Reported")

        while cap.isOpened():
            if stop_detection:
                break

            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, imgsz=640, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
            detections = results.boxes

            detected_accident = False

            for box in detections:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label_name = model.names[cls_id]

                if "accident" in label_name.lower() and conf > CONFIDENCE_THRESHOLD:
                    detected_accident = True
                    break

            if detected_accident:
                accident_streak += 1
            else:
                accident_streak = 0

            if accident_streak >= CONSECUTIVE_ACCIDENT_FRAMES:
                label = "Accident Detected!"
                color = (0, 0, 255)

                if accident_frame_count < MAX_ACCIDENT_FRAMES:
                    accident_frame_count += 1

                    # 🛠 Add accident frame number and timestamp
                    label_text = f"Accident Frame {accident_frame_count}"
                    timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Draw both labels
                    cv2.putText(frame, label_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    cv2.putText(frame, timestamp_now, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                    save_path = video_save_dir / f"accident_frame_{accident_frame_count:04d}.jpg"
                    cv2.imwrite(str(save_path), frame)

                    print(f"💾 Saved accident frame: {save_path}")
                    accident_label.config(text=f"Accidents Detected: {accident_frame_count} incident photos captured")
                accident_streak = 0
            else:
                label = "No Accident"
                color = (0, 255, 0)

            # Draw border
            h, w, _ = frame.shape
            thickness = 8
            cv2.rectangle(frame, (0, 0), (w, h), color, thickness)
            cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # Display inside GUI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_pil = img_pil.resize((VIDEO_CANVAS_WIDTH, VIDEO_CANVAS_HEIGHT))
            imgtk = ImageTk.PhotoImage(image=img_pil)

            video_canvas.create_image(0, 0, anchor='nw', image=imgtk)
            video_canvas.imgtk = imgtk

            video_canvas.update()
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

        cap.release()

    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        reset_ui(status_label, start_button, accident_label)

# ========= RESET FUNCTION =========
def reset_ui(status_label, start_button, accident_label):
    status_label.config(text="Status: Waiting...")
    start_button.config(text="Start Detection")
    accident_label.config(text="No Accident Reported")

# ========= GUI FUNCTIONS =========
def start_detection(selected_video, status_label, start_button, accident_label, video_canvas):
    global stop_detection
    stop_detection = False
    if selected_video is None:
        messagebox.showwarning("Warning", "Please select a video first.")
        return
    threading.Thread(target=detect_accidents, args=(selected_video, status_label, start_button, accident_label, video_canvas), daemon=True).start()

def stop_detection_now():
    global stop_detection
    stop_detection = True

def select_video_and_start(listbox, status_label, start_button, accident_label, video_canvas):
    selected = listbox.curselection()
    if not selected:
        messagebox.showwarning("Warning", "Please select a video first.")
        return
    video_filename = listbox.get(selected[0])
    video_path = VIDEOS_DIR / video_filename

    if start_button['text'] == "Start Detection":
        start_button.config(text="Stop Detection")
        status_label.config(text="Status: Starting 🚀")
        accident_label.config(text="No Accident Reported")
        start_detection(video_path, status_label, start_button, accident_label, video_canvas)
    else:
        stop_detection_now()

# ========= MAIN GUI =========
def main_gui():
    app = Tk()
    app.title("🚗 Accident Detection GUI")
    app.geometry("800x900")

    Label(app, text="Select a Test Video:", font=("Helvetica", 16)).pack(pady=10)

    listbox = Listbox(app, selectmode=SINGLE, font=("Helvetica", 14), width=70)
    listbox.pack(pady=10)

    for file in os.listdir(VIDEOS_DIR):
        if file.endswith((".mp4", ".avi", ".mov")):
            listbox.insert(END, file)

    start_button = Button(app, text="Start Detection", font=("Helvetica", 14))
    start_button.pack(pady=10)

    video_canvas = Canvas(app, width=VIDEO_CANVAS_WIDTH, height=VIDEO_CANVAS_HEIGHT, bg="black")
    video_canvas.pack(pady=10)

    status_label = Label(app, text="Status: Waiting...", font=("Helvetica", 14))
    status_label.pack(pady=10)

    accident_label = Label(app, text="No Accident Reported", font=("Helvetica", 14))
    accident_label.pack(pady=5)

    start_button.config(
        command=lambda: select_video_and_start(listbox, status_label, start_button, accident_label, video_canvas)
    )

    app.mainloop()

if __name__ == "__main__":
    main_gui()
