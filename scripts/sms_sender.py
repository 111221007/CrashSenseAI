# sms_sender.py
import requests
from pathlib import Path

# ====== CONFIGURATION ======
TEXTBEE_API_KEY = "6b5238f9-87de-43e9-b26c-3d34af55388a"   # Your real API key
DEVICE_ID = "680d679a60ed7610f7173dbc"                    # Your device ID
PHONE_NUMBER = "+886928316907"                             # Receiver's phone number (with country code)

TEXTBEE_API_URL = f"https://api.textbee.dev/api/v1/gateway/devices/{DEVICE_ID}/send-sms"

sms_message = (
    "🚨 Accident Detected - CrashSenseAI Alert 🚗\n\n"
    "Accident detected by CrashSenseAI system!\n"
    "Check attached camera feed immediately.\n\n"
    "- CrashSenseAI Team 🚀"
)

# ====== SMS SENDER FUNCTION ======
def send_accident_sms(attachments_folder: Path = None):
    """Send an SMS notification for accident detection."""
    try:
        headers = {
            "x-api-key": TEXTBEE_API_KEY,
            "Content-Type": "application/json"
        }

        payload = {
            "recipients": [PHONE_NUMBER],
            "message": sms_message
        }

        # Send SMS
        response = requests.post(TEXTBEE_API_URL, json=payload, headers=headers)

        if response.status_code == 200:
            print("✅ Accident alert SMS sent successfully!")
        else:
            print(f"❌ Error sending SMS: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Exception occurred: {e}")

# Example usage
if __name__ == "__main__":
    send_accident_sms()
