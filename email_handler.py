import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Email configuration
GMAIL_ADDRESS = os.getenv('GMAIL_ADDRESS', 'indalavenkatasrisaihemanth@gmail.com')
GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD', 'gydp dyzg rvsm towz')

class EmailSender:
    def __init__(self):
        self.sender_email = GMAIL_ADDRESS
        self.app_password = GMAIL_APP_PASSWORD

    def _create_message(
        self, 
        subject: str, 
        body: str, 
        to_emails: List[str], 
        attachments: Optional[List[Dict]] = None
    ) -> MIMEMultipart:
        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = ", ".join(to_emails)
        msg["Subject"] = subject

        # Add body
        msg.attach(MIMEText(body, "plain"))

        # Add attachments if any
        if attachments:
            for attachment in attachments:
                filename = attachment.get('filename')
                content = attachment.get('content')
                mime_type = attachment.get('mime_type', 'application/pdf')
                
                if filename and content:
                    part = MIMEApplication(content, _subtype=mime_type.split('/')[-1])
                    part.add_header('Content-Disposition', 'attachment', filename=filename)
                    msg.attach(part)

        return msg

    def send_email(
        self, 
        subject: str, 
        body: str, 
        to_emails: List[str], 
        attachments: Optional[List[Dict]] = None
    ) -> bool:
        try:
            msg = self._create_message(subject, body, to_emails, attachments)

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(self.sender_email, self.app_password)
                server.send_message(msg)
                print(f"Email sent successfully to {', '.join(to_emails)}")
                return True

        except smtplib.SMTPAuthenticationError:
            print("Failed to authenticate with Gmail. Please check your App Password.")
            raise
        except Exception as e:
            print(f"Failed to send email: {e}")
            raise

    def send_credentials(self, email: str, username: str, password: str) -> bool:
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
        return self.send_email(subject, body, [email])

    def send_appointment_confirmation(
        self,
        to_email: str,
        child_name: str,
        provider_name: str,
        appointment_date: str,
        appointment_type: str = "General Checkup"
    ) -> bool:
        subject = "Appointment Confirmation"
        body = f"""
        Hello,

        This email confirms an appointment for {child_name} with {provider_name}.

        Details:
        Date and Time: {appointment_date}
        Type: {appointment_type}

        Please arrive 10 minutes before your scheduled time.

        Best regards,
        Your Health System
        """
        return self.send_email(subject, body, [to_email])

    def send_vaccination_record(
        self,
        to_email: str,
        child_name: str,
        vaccine_name: str,
        dose_number: int,
        administered_date: str,
        administered_by: str
    ) -> bool:
        subject = "Vaccination Record"
        body = f"""
        Hello,

        This email confirms that {child_name} has received the following vaccination:

        Vaccine: {vaccine_name}
        Dose Number: {dose_number}
        Date Administered: {administered_date}
        Administered By: {administered_by}

        Please keep this record for your files.

        Best regards,
        Your Health System
        """
        return self.send_email(subject, body, [to_email])

# Create a global instance
email_sender = EmailSender() 