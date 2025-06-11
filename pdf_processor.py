import mysql.connector
from mysql.connector import Error
import bcrypt
import pytesseract
from pdf2image import convert_from_path
import re
import random
import string
from datetime import datetime
from email_handler import email_sender
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'ChildHealth')
}

def connect_to_db():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        raise

def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def convert_date_to_mysql_format(date_str):
    try:
        date_obj = datetime.strptime(date_str.strip(), '%B %d, %Y')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError as e:
        print(f"Error parsing date: {e}")
        raise

def parse_text_to_data(text):
    data = {}
    try:
        data['first_name'] = re.search(r"First Name:\s*(.*)", text).group(1).strip()
        data['last_name'] = re.search(r"Last Name:\s*(.*)", text).group(1).strip()
        
        dob_match = re.search(r"Date of Birth:\s*(.*)", text)
        if dob_match:
            date_str = dob_match.group(1).strip()
            data['dob'] = convert_date_to_mysql_format(date_str)
        
        data['gender'] = re.search(r"Gender:\s*(.*)", text).group(1).strip()
        data['parent_id'] = '52'  # Hardcoded parent_id as requested
        data['blood_type'] = re.search(r"Blood Type:\s*(.*)", text).group(1).strip()
        data['pincode'] = re.search(r"Pincode:\s*(.*)", text).group(1).strip()
        data['email'] = re.search(r"Email:\s*(.*)", text).group(1).strip()
        data['vaccine_name'] = re.search(r"Vaccine Name:\s*(.*)", text).group(1).strip()
        data['dose_number'] = int(re.search(r"Dose Number:\s*(\d+)", text).group(1).strip())
        
        date_admin_match = re.search(r"Date Administered:\s*(.*)", text)
        if date_admin_match:
            date_str = date_admin_match.group(1).strip()
            data['date_administered'] = convert_date_to_mysql_format(date_str)
            
        data['administered_by'] = re.search(r"Administered By:\s*(.*)", text).group(1).strip()
        data['reminder_sent'] = 1 if "Yes" in re.search(r"Reminder Sent:\s*(.*)", text).group(1).strip() else 0
        data['notes'] = re.search(r"Notes:\s*(.*)", text).group(1).strip()

        return data
    except (AttributeError, ValueError) as e:
        print(f"Error parsing data: {e}")
        print("Raw text from PDF:")
        print(text)
        raise

def generate_username_password(first_name, last_name):
    username = first_name.lower() + '.' + last_name.lower() + str(random.randint(1000, 9999))
    password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return username, password

def hash_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password

def send_credentials_to_user(email, username, password):
    try:
        return email_sender.send_credentials(email, username, password)
    except Exception as e:
        print(f"Failed to send email: {e}")
        raise

def process_pdf_and_store_data(pdf_path):
    try:
        # Extract and parse data from PDF
        text = extract_text_from_pdf(pdf_path)
        data = parse_text_to_data(text)
        
        # Generate credentials
        username, password = generate_username_password(data['first_name'], data['last_name'])
        hashed_password = hash_password(password)

        # Connect to database and store data
        conn = connect_to_db()
        cursor = conn.cursor()

        try:
            # Insert into Children table
            insert_child = """
                INSERT INTO Children 
                (first_name, last_name, date_of_birth, gender, parent_id, blood_type, pincode, email)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            child_data = (
                data['first_name'], 
                data['last_name'], 
                data['dob'],
                data['gender'],
                data['parent_id'],
                data['blood_type'],
                data['pincode'],
                data['email']
            )
            cursor.execute(insert_child, child_data)
            child_id = cursor.lastrowid

            # Insert into VaccinationTakenRecords table
            insert_vaccination = """
                INSERT INTO VaccinationTakenRecords 
                (child_id, vaccine_name, dose_number, date_administered, administered_by, reminder_sent, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            vaccination_data = (
                child_id,
                data['vaccine_name'],
                data['dose_number'],
                data['date_administered'],
                data['administered_by'],
                data['reminder_sent'],
                data['notes']
            )
            cursor.execute(insert_vaccination, vaccination_data)

            # Insert into ChildUsers table
            insert_user = """
                INSERT INTO ChildUsers 
                (child_id, username, password_hash)
                VALUES (%s, %s, %s)
            """
            user_data = (child_id, username, hashed_password)
            cursor.execute(insert_user, user_data)

            conn.commit()

            # Send credentials via email
            send_credentials_to_user(data['email'], username, password)

            return {
                "success": True,
                "child_id": child_id,
                "message": "Data processed and stored successfully"
            }

        except mysql.connector.Error as err:
            conn.rollback()
            raise Exception(f"Database error: {str(err)}")
        finally:
            cursor.close()
            conn.close()

    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}") 