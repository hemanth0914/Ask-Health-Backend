import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Optional
from fastapi import HTTPException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Email configuration
GMAIL_ADDRESS = os.getenv('GMAIL_ADDRESS', 'indalavenkatasrisaihemanth@gmail.com')
GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD', 'gydp dyzg rvsm towz')

def send_email(subject: str, body: str, to_emails: List[str], attachments: List[Dict] = None):
    """
    Send an email with optional attachments using Gmail's SMTP server
    
    Args:
        subject: Email subject
        body: Email body
        to_emails: List of recipient email addresses
        attachments: List of dictionaries with keys:
            - filename: Name of the file
            - content: Binary content of the file
            - content_type: MIME type of the file
    """
    try:
        # Create message container
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = GMAIL_ADDRESS
        msg['To'] = ", ".join(to_emails)

        # Add body
        msg.attach(MIMEText(body, 'plain'))

        # Add attachments if any
        if attachments:
            for attachment in attachments:
                if 'content_type' in attachment and attachment['content_type'].startswith('application/'):
                    part = MIMEApplication(
                        attachment['content'],
                        _subtype=attachment['content_type'].split('/')[-1]
                    )
                else:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment['content'])
                    encoders.encode_base64(part)
                
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename="{attachment["filename"]}"'
                )
                if 'content_type' in attachment:
                    part.set_type(attachment['content_type'])
                msg.attach(part)

        # Send email using Gmail's SMTP server
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.send_message(msg)
            print(f"Email sent successfully to {to_emails}")
            return True

    except smtplib.SMTPAuthenticationError:
        error_msg = "Failed to authenticate with Gmail. Please check your App Password."
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Failed to send email: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

def send_credentials_email(email: str, username: str, password: str):
    """Send account credentials via email"""
    subject = "Your Child's Account Credentials"
    body = f"""
    Hello,

    Your child's account has been created. Here are the login details:

    Username: {username}
    Password: {password}

    Please change your password after your first login for security purposes.

    Best regards,
    Your Health System
    """
    return send_email(subject, body, [email])

def send_appointment_confirmation_email(to_email: str, child_name: str, provider_name: str, appointment_date: str):
    """Send appointment confirmation email"""
    subject = "Appointment Confirmation"
    body = f"""
    Hello,

    This email confirms an appointment for {child_name} with {provider_name}.

    Details:
    Date and Time: {appointment_date}

    Please arrive 10 minutes before your scheduled time.

    Best regards,
    Your Health System
    """
    return send_email(subject, body, [to_email])

def send_ehr_email(provider_email: str, provider_name: str, child_name: str, pdf_content: bytes):
    """Send EHR PDF via email"""
    subject = f"Electronic Health Record - {child_name}"
    body = f"""
    Dear Dr. {provider_name},

    Please find attached the Electronic Health Record for {child_name}.

    Best regards,
    Your Health System
    """
    attachments = [{
        "filename": f"EHR_{child_name}.pdf",
        "content": pdf_content,
        "content_type": "application/pdf"
    }]
    return send_email(subject, body, [provider_email], attachments) 