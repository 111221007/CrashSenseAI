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

# ========= CONFIGURATION =========
MODEL_PATH = Path(r"C:/Users/cmpor/PycharmProjects/EdgeAI_Benchmark_Project/accident_detection/model/best.pt")
VIDEOS_DIR = Path(r"C:/Users/cmpor/PycharmProjects/EdgeAI_Benchmark_Project/accident_detection/data/input/test_videos")
SAVE_DIR = Path(r"C:/Users/cmpor/PycharmProjects/EdgeAI_Benchmark_Project/accident_detection/data/output/accident_frames")
SOUND_PATH = Path(r"C:/Users/cmpor/PycharmProjects/EdgeAI_Benchmark_Project/accident_detection/data/alarm.mp3")

CONFIDENCE_THRESHOLD = 0.5
CONSECUTIVE_ACCIDENT_FRAMES = 3
MAX_ACCIDENT_FRAMES = 5
VIDEO_CANVAS_WIDTH = 600
VIDEO_CANVAS_HEIGHT = 360

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ========= GLOBALS =========
stop_detection = False
is_muted = False
model = YOLO(str(MODEL_PATH)).to('cuda' if torch.cuda.is_available() else 'cpu')

# ========= SOUND AND EMAIL HELPERS =========
def play_alarm_sound():
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
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except:
        pass
    mute_button.config(text="Unmute 🔊" if is_muted else "Mute 🔇", bootstyle=WARNING)

def send_email_async(save_folder):
    from email_sender import send_accident_email
    threading.Thread(target=lambda: send_accident_email(save_folder), daemon=True).start()

# ========= UI HELPERS =========
def reset_ui(status_label, start_button, accident_label, progress_bar):
    status_label.config(text="Status: Waiting...")
    start_button.config(text="Start Detection", bootstyle=SUCCESS)
    accident_label.config(text="No Accident Reported")
    progress_bar['value'] = 0

def draw_text_box(frame, text, top_left, font_scale=0.7, thickness=2, padding=(15, 8)):
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x1, y1 = top_left
    x2, y2 = x1 + text_width + padding[0]*2, y1 + text_height + padding[1]*2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.putText(frame, text, (x1 + padding[0], y1 + text_height + padding[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

def draw_detection_overlay(frame, label_text, timestamp_text):
    draw_text_box(frame, label_text, top_left=(20, 20))
    draw_text_box(frame, timestamp_text, top_left=(20, 90))

# ========= CORE DETECTION =========
def detect_accidents(video_path, status_label, start_button, accident_label, progress_bar, video_canvas):
    global stop_detection
    try:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            messagebox.showerror("Error", "❌ Cannot open selected video.")
            reset_ui(status_label, start_button, accident_label, progress_bar)
            return

        video_filename = Path(video_path).stem
        save_path_folder = SAVE_DIR / "SelectedVideo" / video_filename
        save_path_folder.mkdir(parents=True, exist_ok=True)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = 1000 / video_fps

        status_label.config(text="Status: Running 🚀")
        accident_label.config(text="No Accident Reported")

        accident_streak = 0
        accident_frame_count = 0
        frame_count = 0
        email_sent = False

        while cap.isOpened():
            if stop_detection:
                break

            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results = model.predict(frame, imgsz=640, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
            detections = results.boxes

            detected_accident = any("accident" in model.names[int(box.cls[0])].lower() and float(box.conf[0]) > CONFIDENCE_THRESHOLD for box in detections)

            if detected_accident:
                accident_streak += 1
            else:
                accident_streak = 0

            if accident_streak >= CONSECUTIVE_ACCIDENT_FRAMES:
                if accident_frame_count < MAX_ACCIDENT_FRAMES:
                    accident_frame_count += 1
                    label_text = f"Accident Detected!"
                    timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    draw_detection_overlay(frame, label_text, timestamp_now)

                    frame_save_path = save_path_folder / f"accident_frame_{accident_frame_count:04d}.jpg"
                    cv2.imwrite(str(frame_save_path), frame)

                    accident_label.config(text="Accident Detected!")

                    play_alarm_sound()

                    if not email_sent:
                        send_email_async(save_path_folder)
                        email_sent = True
                accident_streak = 0

            color = (0, 0, 255) if detected_accident else (0, 255, 0)
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, h), color, 8)
            cv2.putText(frame, "Accident Detected!" if detected_accident else "No Accident",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb).resize((VIDEO_CANVAS_WIDTH, VIDEO_CANVAS_HEIGHT))
            imgtk = ImageTk.PhotoImage(image=img_pil)
            video_canvas.create_image(0, 0, anchor='nw', image=imgtk)
            video_canvas.imgtk = imgtk

            progress = (frame_count / total_frames) * 100
            progress_bar['value'] = progress
            video_canvas.update()

            elapsed_ms = (time.time() - start_time) * 1000
            delay = frame_delay - elapsed_ms
            if delay > 0:
                video_canvas.after(int(delay))

        cap.release()

    except Exception as e:
        messagebox.showerror("Error", str(e))

    finally:
        reset_ui(status_label, start_button, accident_label, progress_bar)

# ========= GUI UTILITY FUNCTIONS =========
def start_detection(selected_video, status_label, start_button, accident_label, progress_bar, video_canvas):
    global stop_detection
    stop_detection = False
    if selected_video:
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
    mute_button = Button(app, text="Mute 🔇", bootstyle=WARNING)
    video_canvas = Canvas(app, width=VIDEO_CANVAS_WIDTH, height=VIDEO_CANVAS_HEIGHT, bg="black")
    progress_bar = Progressbar(app, length=500, mode='determinate')
    status_label = Label(app, text="Status: Waiting...", font=("Helvetica", 14))
    accident_label = Label(app, text="No Accident Reported", font=("Helvetica", 14))

    start_button.pack(pady=10)
    mute_button.pack(pady=5)
    video_canvas.pack(pady=10)
    progress_bar.pack(pady=10)
    status_label.pack(pady=10)
    accident_label.pack(pady=5)

    start_button.config(command=lambda: select_video_and_start(listbox, status_label, start_button, accident_label, progress_bar, video_canvas))
    mute_button.config(command=lambda: toggle_mute(mute_button))

    app.mainloop()

if __name__ == "__main__":
    main_gui()
