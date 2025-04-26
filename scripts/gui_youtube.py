import os
import cv2
import numpy as np
import threading
from pathlib import Path
import warnings
import torch
import datetime
import yt_dlp
from tkinter import Tk, Label, Button, Entry, messagebox, Canvas
from PIL import Image, ImageTk
from ultralytics import YOLO

# ========= CONFIGURATIONS =========
MODEL_PATH = Path(r"C:\Users\cmpor\PycharmProjects\EdgeAI_Benchmark_Project\accident_detection\model\best.pt")
SAVE_DIR = Path(r"C:/Users/cmpor/PycharmProjects/EdgeAI_Benchmark_Project/accident_detection/data/output/accident_frames_streaming")

CONFIDENCE_THRESHOLD = 0.5
CONSECUTIVE_ACCIDENT_FRAMES = 3
MAX_ACCIDENT_FRAMES = 5

VIDEO_CANVAS_WIDTH = 600
VIDEO_CANVAS_HEIGHT = 360

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ========= GLOBAL =========
stop_detection = False

# ========= YOUTUBE STREAMING =========
def get_stream_url(youtube_url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio/best',
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        title = info.get('title', 'youtube_video').replace(' ', '_').replace('/', '_')
        if 'url' in info:
            return info['url'], title
        for fmt in info.get('formats', []):
            if (fmt.get('url') and fmt.get('ext') == 'mp4'
                    and fmt.get('vcodec') != 'none'
                    and fmt.get('acodec') != 'none'):
                return fmt['url'], title
        raise RuntimeError("No suitable MP4 format found")

# ========= DETECTION FUNCTION =========
def detect_accidents_from_stream(youtube_url, status_label, start_button, accident_label, video_canvas):
    global stop_detection
    try:
        model = YOLO(str(MODEL_PATH))
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        stream_url, video_title = get_stream_url(youtube_url)
        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            messagebox.showerror("Error", "❌ Cannot open YouTube stream URL.")
            reset_ui(status_label, start_button, accident_label)
            return

        video_save_dir = SAVE_DIR / "YouTube"
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

                    # Add accident frame number and timestamp
                    label_text = f"Accident Frame {accident_frame_count}"
                    timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
def start_detection(url_entry, status_label, start_button, accident_label, video_canvas):
    global stop_detection
    stop_detection = False
    youtube_url = url_entry.get().strip()
    if not youtube_url:
        messagebox.showwarning("Warning", "Please enter a YouTube link.")
        return
    threading.Thread(target=detect_accidents_from_stream, args=(youtube_url, status_label, start_button, accident_label, video_canvas), daemon=True).start()

def stop_detection_now():
    global stop_detection
    stop_detection = True

# ========= MAIN GUI =========
def main_gui():
    app = Tk()
    app.title("🚗 Accident Detection from YouTube GUI")
    app.geometry("800x900")

    Label(app, text="Paste YouTube Link:", font=("Helvetica", 16)).pack(pady=10)

    url_entry = Entry(app, font=("Helvetica", 14), width=70)
    url_entry.pack(pady=10)

    start_button = Button(app, text="Start Detection", font=("Helvetica", 14))
    start_button.pack(pady=10)

    video_canvas = Canvas(app, width=VIDEO_CANVAS_WIDTH, height=VIDEO_CANVAS_HEIGHT, bg="black")
    video_canvas.pack(pady=10)

    status_label = Label(app, text="Status: Waiting...", font=("Helvetica", 14))
    status_label.pack(pady=10)

    accident_label = Label(app, text="No Accident Reported", font=("Helvetica", 14))
    accident_label.pack(pady=5)

    start_button.config(
        command=lambda: start_detection(url_entry, status_label, start_button, accident_label, video_canvas)
    )

    app.mainloop()

if __name__ == "__main__":
    main_gui()
# https://www.youtube.com/watch?v=JhRyLKfmACM&ab_channel=FriantRoulette
