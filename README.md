# 🚗 CrashSenseAI
**Senses Crashes Instantly with AI**

---

## 🔥 Overview
CrashSenseAI is a real-time accident detection system using Edge AI models.  
It detects crashes from videos or streams, raises sound alarms, saves accident frames, and instantly sends email notifications.

**Built with:**
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch
- Tkinter + ttkbootstrap (modern dark-themed GUI)
- Pygame for alarms
- Automated email sending with attachments

---

## 🚀 Features
- 🎯 Real-time accident detection from videos
- 💾 Auto-save accident frames to organized folders
- 📩 Automatic email alerts with attached crash images
- 🔊 Sound alarm on accident detection (Mute/Unmute support)
- 🎥 Original video playback speed (no artificial slowdowns)
- 🖥️ Clean and professional dark GUI
- ⚡ Lightweight, fast, and easy to deploy

---

## 🛠 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/111221007/CrashSenseAI.git
cd CrashSenseAI
2. Install Dependencies
Install all required packages:

bash
Copy
Edit
pip install -r requirements.txt
requirements.txt includes:

Copy
Edit
opencv-python
numpy
torch
ttkbootstrap
pygame
Pillow
ultralytics
3. Prepare Folders and Files
Download your trained YOLOv8 model and place it at:

bash
Copy
Edit
CrashSenseAI/accident_detection/model/best.pt
Prepare test videos and put them into:

swift
Copy
Edit
CrashSenseAI/accident_detection/data/input/test_videos/
Add an alarm sound file (e.g., alarm.mp3) at:

bash
Copy
Edit
CrashSenseAI/accident_detection/data/alarm.mp3
4. Configure Email Settings
Inside email_sender.py, update your email credentials:

python
Copy
Edit
sender_email = "your_email@gmail.com"
sender_password = "your_app_password"
receiver_email = "receiver_email@gmail.com"
⚡ Important: Use an App Password if using Gmail.
Check How to Generate App Password.

▶️ How to Run CrashSenseAI
Run the main GUI file:

bash
Copy
Edit
python gui.py
Select a test video from the list.

Click Start Detection.

The system will:

Detect accidents in real-time.

Save crash frames automatically inside:

swift
Copy
Edit
accident_detection/data/output/accident_frames/SelectedVideo/
Play alarm sounds upon accident detection.

Send an email with the saved accident images automatically after detection.

✅ You can Mute/Unmute the sound anytime from the GUI.

📂 Project Structure
bash
Copy
Edit
CrashSenseAI/
├── accident_detection/
│   ├── data/
│   │   ├── input/test_videos/       # Test videos here
│   │   ├── output/accident_frames/  # Detected crash images saved here
│   │   └── alarm.mp3                # Alarm sound file
│   ├── model/
│   │   └── best.pt                  # YOLOv8 trained model
│   ├── scripts/
│   │   └── gui.py                   # Main GUI application
│   └── email_sender.py              # Email sending logic
├── requirements.txt
└── README.md
🎯 Future Work
🚀 Live webcam / YouTube stream accident detection

📊 Dashboard for monitoring crash incidents

☁️ Cloud-based remote alert system

👨‍💻 Developed By
[Your Name Here]

⭐ Star this Repository if you find it useful! ⭐
yaml
Copy
Edit

---

✅ This is in **pure README.md format** and will render perfectly when you upload to GitHub.

---

Would you like me to also generate:
- a professional `.gitignore`
- a ready `LICENSE (MIT License)` file 
to complete your GitHub project? 🚀  
(Reply: **yes** if you want!)






