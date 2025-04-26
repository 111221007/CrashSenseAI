# email_sender.py
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path

# ====== CONFIGURATION ======
SENDER_EMAIL = "andi4work@gmail.com"
RECEIVER_EMAIL = "cmporeddy@gmail.com"

SENDER_PASSWORD = "xdav vhzi jrsn thdn"  # Use App Password for Gmail
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
# ====== EMAIL SENDER FUNCTION ======
def send_accident_email(attachments_folder: Path):
    """Send an email with all accident frame images attached."""
    try:
        message = EmailMessage()
        message['Subject'] = '🚨 Accident Detected - CrashSenseAI Alert'
        message['From'] = SENDER_EMAIL
        message['To'] = RECEIVER_EMAIL
        message.set_content('Attached are the accident frames captured by CrashSenseAI.')

        # Attach up to 5 images
        images = sorted(attachments_folder.glob("*.jpg"))
        for img in images[:5]:
            with open(img, 'rb') as file:
                file_data = file.read()
                message.add_attachment(file_data, maintype='image', subtype='jpeg', filename=img.name)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)

        print("✅ Accident alert email sent successfully!")

    except Exception as e:
        print(f"❌ Error sending email: {e}")
