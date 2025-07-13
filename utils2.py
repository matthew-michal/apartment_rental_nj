import boto3
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

FROM_ADDRESS = os.getenv('FROM_ADDRESS',"")

def send_email_with_attachment(subject: str, body: str, recipients: list, attachments: list):
    """
    Send email with attachments using AWS SES.
    """
    ses_client = boto3.client('ses', region_name='us-east-1')  # Replace with your SES region
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = FROM_ADDRESS  # Replace with verified SES email
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
    
    # Send email
    try:
        response = ses_client.send_raw_email(
            Source=msg['From'],
            Destinations=recipients,
            RawMessage={'Data': msg.as_string()}
        )
        print(f"Email sent successfully. Message ID: {response['MessageId']}")
    except Exception as e:
        print(f"Error sending email: {e}")
        raise