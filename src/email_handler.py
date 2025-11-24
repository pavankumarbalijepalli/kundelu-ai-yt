from utils import log
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from utils import email_template

def send_email(youtube_post: str, topic: str, html_template: str = email_template):
    # Email credentials
    sender_email = os.getenv("EMAIL_FROM")
    app_password = os.getenv("EMAIL_PASSWORD")
    receiver_email = os.getenv("EMAIL_TO")

    # Create the email
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Your Daily Youtube Content: " + topic
    msg["From"] = f"Y.T.A Agent <{sender_email}>"
    msg["To"] = receiver_email

    # Email content (plain + HTML)
    # text = "Hi there!\nThis is a test email sent from Python."
    html = html_template.format(
        youtube_post=youtube_post,
        kundelu_ai="https://raw.githubusercontent.com/pavankumarbalijepalli/pavankumarbalijepalli/refs/heads/main/kundelu_ai.png"
    )

    # msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))

    # Send the email via Gmail SMTP
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())

    log("✅ Email sent successfully!")