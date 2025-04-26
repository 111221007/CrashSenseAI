import os
import cv2
import numpy as np
import threading
import time
from pathlib import Path
import warnings
import torch
import datetime
import pygame
from ttkbootstrap import Style, Label, Button
from ttkbootstrap.constants import *
from tkinter import Listbox, Canvas, SINGLE, END, messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
from ultralytics import YOLO

# ========= CONFIGURATIONS =========
MODEL_PATH = Path(r"C:\Users\cmpor\PycharmProjects\EdgeAI_Benchmark_Project\accident_detection\model\best.pt")
VIDEOS_DIR = Path(r"C:\Users\cmpor\PycharmProjects\EdgeAI_Benchmark_Project\accident_detection\data\input\test_videos")
SAVE_DIR = Path(r"C:/Users/cmpor/PycharmProjects/EdgeAI_Benchmark_Project/accident_detection/data/output/accident_frames")
SOUND_PATH = Path(r"C:/Users/cmpor/PycharmProjects\EdgeAI_Benchmark_Project\accident_detection/data/alarm.mp3")

CONFIDENCE_THRESHOLD = 0.5
CONSECUTIVE_ACCIDENT_FRAMES = 3
MAX_ACCIDENT_FRAMES = 5

VIDEO_CANVAS_WIDTH = 600
VIDEO_CANVAS_HEIGHT = 360

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ========= GLOBAL =========
stop_detection = False
is_muted = False
model = YOLO(str(MODEL_PATH)).to('cuda' if torch.cuda.is_available() else 'cpu')

# ========= DETECTION FUNCTION =========
def detect_accidents(video_path, status_label, start_button, accident_label, progress_bar, video_canvas):
    global stop_detection
    try:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            messagebox.showerror("Error", "❌ Cannot open selected video.")
            reset_ui(status_label, start_button, accident_label, progress_bar)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_filename = Path(video_path).stem
        video_save_dir = SAVE_DIR / "SelectedVideo" / video_filename
        video_save_dir.mkdir(parents=True, exist_ok=True)

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = int(1000 / video_fps) if video_fps > 0 else 30

        accident_streak = 0
        accident_frame_count = 0
        frame_count = 0

        status_label.config(text="Status: Running 🚀")
        accident_label.config(text="No Accident Reported")

        while cap.isOpened():
            if stop_detection:
                print("Detection stopped by user.")
                break

            frame_start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

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

                    label_text = f"Accident Frame {accident_frame_count}"
                    timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # ====== Draw Label with WHITE box ======
                    (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (20, 20), (20 + label_width + 10, 20 + label_height + 10), (255, 255, 255),
                                  -1)  # white box
                    cv2.putText(frame, label_text, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # black text

                    # ====== Draw Timestamp with WHITE box ======
                    (time_width, time_height), _ = cv2.getTextSize(timestamp_now, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (20, 60), (20 + time_width + 10, 60 + time_height + 10), (255, 255, 255),
                                  -1)  # white box
                    cv2.putText(frame, timestamp_now, (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),
                                2)  # black text

                    save_path = video_save_dir / f"accident_frame_{accident_frame_count:04d}.jpg"
                    cv2.imwrite(str(save_path), frame)

                    print(f"💾 Saved accident frame: {save_path}")
                    accident_label.config(text=f"Accidents Detected: {accident_frame_count} incident photos captured")

                    play_alarm_sound()

                accident_streak = 0
            else:
                label = "No Accident"
                color = (0, 255, 0)

            h, w, _ = frame.shape
            thickness = 8
            cv2.rectangle(frame, (0, 0), (w, h), color, thickness)
            cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_pil = img_pil.resize((VIDEO_CANVAS_WIDTH, VIDEO_CANVAS_HEIGHT))
            imgtk = ImageTk.PhotoImage(image=img_pil)

            video_canvas.create_image(0, 0, anchor='nw', image=imgtk)
            video_canvas.imgtk = imgtk

            progress = (frame_count / total_frames) * 100
            progress_bar['value'] = progress

            video_canvas.update()

            frame_end_time = time.time()
            elapsed_time = (frame_end_time - frame_start_time) * 1000
            remaining_delay = frame_delay - elapsed_time

            if remaining_delay > 0:
                video_canvas.after(int(remaining_delay))

        cap.release()

    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        reset_ui(status_label, start_button, accident_label, progress_bar)

# ========= SOUND FUNCTION =========
def play_alarm_sound():
    global is_muted
    if is_muted:
        return
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(SOUND_PATH))
        pygame.mixer.music.play()
    except Exception as e:
        print(f"🔔 Error playing sound: {e}")

def toggle_mute(mute_button):
    global is_muted
    is_muted = not is_muted
    if is_muted:
        mute_button.config(text="Unmute 🔊", bootstyle=WARNING)
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception as e:
            print(f"🔇 Error stopping sound: {e}")
    else:
        mute_button.config(text="Mute 🔇", bootstyle=WARNING)

# ========= UI RESET =========
def reset_ui(status_label, start_button, accident_label, progress_bar):
    status_label.config(text="Status: Waiting...")
    start_button.config(text="Start Detection", bootstyle=SUCCESS)
    accident_label.config(text="No Accident Reported")
    progress_bar['value'] = 0

# ========= GUI LOGIC =========
def start_detection(selected_video, status_label, start_button, accident_label, progress_bar, video_canvas):
    global stop_detection
    stop_detection = False
    if selected_video is None:
        messagebox.showwarning("Warning", "Please select a video first.")
        return
    threading.Thread(target=detect_accidents, args=(selected_video, status_label, start_button, accident_label, progress_bar, video_canvas), daemon=True).start()

def stop_detection_now():
    global stop_detection
    stop_detection = True

def select_video_and_start(listbox, status_label, start_button, accident_label, progress_bar, video_canvas):
    selected = listbox.curselection()
    if not selected:
        messagebox.showwarning("Warning", "Please select a video first.")
        return
    video_filename = listbox.get(selected[0])
    video_path = VIDEOS_DIR / video_filename

    if start_button['text'] == "Start Detection":
        start_button.config(text="Stop Detection", bootstyle=DANGER)
        status_label.config(text="Status: Starting 🚀")
        accident_label.config(text="No Accident Reported")
        progress_bar['value'] = 0
        start_detection(video_path, status_label, start_button, accident_label, progress_bar, video_canvas)
    else:
        stop_detection_now()

# ========= MAIN GUI =========
def main_gui():
    app = Style(theme='superhero').master
    app.title("CrashSenseAI 🚗 (Senses Crashes Instantly)")
    app.geometry("800x1000")

    Label(app, text="Select a Test Video:", font=("Helvetica", 16)).pack(pady=10)

    listbox = Listbox(app, selectmode=SINGLE, font=("Helvetica", 14), width=70)
    listbox.pack(pady=10)

    for file in os.listdir(VIDEOS_DIR):
        if file.endswith((".mp4", ".avi", ".mov")):
            listbox.insert(END, file)

    start_button = Button(app, text="Start Detection", bootstyle=SUCCESS)
    start_button.pack(pady=10)

    mute_button = Button(app, text="Mute 🔇", bootstyle=WARNING)
    mute_button.pack(pady=5)
    mute_button.config(
        command=lambda: toggle_mute(mute_button)
    )

    video_canvas = Canvas(app, width=VIDEO_CANVAS_WIDTH, height=VIDEO_CANVAS_HEIGHT, bg="black")
    video_canvas.pack(pady=10)

    progress_bar = Progressbar(app, length=500, mode='determinate')
    progress_bar.pack(pady=10)

    status_label = Label(app, text="Status: Waiting...", font=("Helvetica", 14))
    status_label.pack(pady=10)

    accident_label = Label(app, text="No Accident Reported", font=("Helvetica", 14))
    accident_label.pack(pady=5)

    start_button.config(
        command=lambda: select_video_and_start(listbox, status_label, start_button, accident_label, progress_bar, video_canvas)
    )

    app.mainloop()

if __name__ == "__main__":
    main_gui()
