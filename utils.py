import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

# Environment variables for SMTP credentials
FROM_ADDRESS = os.getenv('FROM_ADDRESS', "")
SMTP_USERNAME = os.getenv('SMTP_USERNAME', "")
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', "")

# AWS SES SMTP configuration
SES_SMTP_HOST = "email-smtp.us-east-1.amazonaws.com"  # Replace with your SES region
SES_SMTP_PORT = 587  # Use 587 for TLS or 465 for SSL

def send_email_with_attachment(subject: str, body: str, recipients: list, attachments: list):
    """
    Send email with attachments using AWS SES SMTP.
    """
    # Create message
    msg = MIMEMultipart()
    msg['From'] = FROM_ADDRESS
    msg['To'] = ", ".join(recipients)
    msg['Subject'] = subject
    
    # Add body
    msg.attach(MIMEText(body, 'plain'))
    
    # Add attachments
    for attachment_path in attachments:
        if os.path.exists(attachment_path):
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(attachment_path)}'
                )
                msg.attach(part)
    
    # Send email via SMTP
    try:
        # Create SMTP session
        server = smtplib.SMTP(SES_SMTP_HOST, SES_SMTP_PORT)
        server.starttls()  # Enable TLS encryption
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        
        # Send email
        text = msg.as_string()
        server.sendmail(FROM_ADDRESS, recipients, text)
        server.quit()
        
        print("Email sent successfully via SMTP")
        
    except Exception as e:
        print(f"Error sending email: {e}")
        raise

# Alternative function for HTML email body
def send_html_email_with_attachment(subject: str, html_body: str, recipients: list, attachments: list):
    """
    Send HTML email with attachments using AWS SES SMTP.
    """
    # Create message
    msg = MIMEMultipart()
    msg['From'] = FROM_ADDRESS
    msg['To'] = ", ".join(recipients)
    msg['Subject'] = subject
    
    # Add HTML body
    msg.attach(MIMEText(html_body, 'html'))
    
    # Add attachments
    for attachment_path in attachments:
        if os.path.exists(attachment_path):
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(attachment_path)}'
                )
                msg.attach(part)
    
    # Send email via SMTP
    try:
        server = smtplib.SMTP(SES_SMTP_HOST, SES_SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        
        text = msg.as_string()
        server.sendmail(FROM_ADDRESS, recipients, text)
        server.quit()
        
        print("HTML email sent successfully via SMTP")
        
    except Exception as e:
        print(f"Error sending email: {e}")
        raise