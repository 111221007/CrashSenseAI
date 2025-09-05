# 🚗 CrashSenseAI
**Senses Crashes Instantly with AI**

---

## 🔥 Overview
CrashSenseAI is a real-time accident detection system using Edge AI models.  
It detects crashes from videos or streams, raises sound alarms, saves accident frames, and instantly sends email notifications.

### 🛠 Built With
- **Python 3.10.0**
- **YOLOv8 (Ultralytics)** — Deep Learning Object Detection
- **OpenCV** — Video processing
- **PyTorch** — Neural Network backend
- **Tkinter + ttkbootstrap** — Clean Dark GUI
- **Pygame** — Sound alarms
- **Automated Email** — With crash frame attachments

---

## 🚀 Features
- 🎯 Real-time accident detection
- 💾 Auto-save accident frames in organized folders
- 📩 Auto-send email alerts with crash images
- 🔊 Sound alarm on accident detection (Mute/Unmute option)
- 🎥 Original video playback speed (no artificial slowdown)
- 🖥️ Clean and modern dark GUI (Responsive and Lightweight)
- ⚡ Easy setup, lightweight, and customizable

---

## 📦 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/111221007/CrashSenseAI.git
cd CrashSenseAI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt** includes:
- opencv-python
- numpy
- torch
- ttkbootstrap
- pygame
- Pillow
- ultralytics

---

### 3. Prepare Required Folders and Files
✅ Place your trained YOLO model at:
```
CrashSenseAI/accident_detection/model/best.pt
```

✅ Add your **test videos** inside:
```
CrashSenseAI/accident_detection/data/input/test_videos/
```

✅ Add an **alarm sound file** (e.g., `alarm.mp3`) here:
```
CrashSenseAI/accident_detection/data/alarm.mp3
```

✅ Setup your **email credentials** inside `email_sender.py`:
```python
sender_email = "your_email@gmail.com"
sender_password = "your_app_password"
receiver_email = "receiver_email@gmail.com"
```
> ⚡ If using Gmail, you must enable **App Passwords** to use email sending. [Learn how to create App Password →](https://support.google.com/mail/answer/185833?hl=en)

---

### ▶️ How to Run CrashSenseAI
1. Launch the GUI:
   ```bash
   python gui.py
   ```

2. Select a test video from the GUI.

3. Click **Start Detection**.

The system will:
- Detect crashes in real-time
- Save crash frames inside:
  ```
  accident_detection/data/output/accident_frames/SelectedVideo/
  ```
- Play sound alarms when crash detected
- Send automatic emails with saved crash images attached
- Allow you to Mute/Unmute alarms anytime from the GUI

---

## 📂 Project Structure
```bash
CrashSenseAI/
├── accident_detection/
│   ├── data/
│   │   ├── input/test_videos/       # ➡️ Your test videos
│   │   ├── output/accident_frames/  # ➡️ Saved crash frames
│   │   └── alarm.mp3                # ➡️ Alarm sound
│   ├── model/
│   │   └── best.pt                  # ➡️ YOLOv8 trained model
│   ├── scripts/
│   │   └── gui.py                   # ➡️ GUI Application
│   └── email_sender.py              # ➡️ Email automation
├── requirements.txt
└── README.md
```

---

## 🌟 Future Roadmap
- 🔴 Live webcam & YouTube live stream crash detection
- 📊 Real-time dashboard monitoring accidents
- ☁️ Cloud-based remote alert system (IoT Integration)
- 📈 Performance optimization for embedded devices (Raspberry Pi / Jetson)

---

## 👨‍💻 Developed By
**Chandramohan Reddy Poreddy**  
👉 _Feel free to contribute by creating a pull request!_

---

## 📜 License
This project is licensed under the **MIT License**.  
Feel free to use, modify, and share responsibly.

---

> ⭐ **If you find this project helpful, please consider giving a star on GitHub!** ⭐
