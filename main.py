from fastapi import FastAPI, HTTPException, Depends, status, Query, BackgroundTasks, WebSocket, Request
from pydantic import BaseModel, validator
from databases import Database
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, date, timezone
from typing import List, Optional, Dict, Literal, Tuple, Any
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta
import re
from fastapi.responses import StreamingResponse, FileResponse
import io
import openai
import json
import smtplib
from email.message import EmailMessage
import pandas as pd
from dateutil.relativedelta import relativedelta
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain.prompts.prompt import PromptTemplate
from sqlalchemy import create_engine, text
import base64
import tempfile
from app.database import SessionLocal, engine, Base
import traceback
from gtts import gTTS
import bcrypt
import pytesseract
from pdf2image import convert_from_path
import re
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import UploadFile, File
import tempfile
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from email import encoders
from email.mime.base import MIMEBase
import shutil
from fastapi.staticfiles import StaticFiles

# Database configuration
DATABASE_URL = "mysql+aiomysql://root:@localhost:3306/ChildHealth"
SYNC_DATABASE_URL = "mysql+pymysql://root:@localhost:3306/ChildHealth"

# Authentication configuration
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Initialize database
database = Database(DATABASE_URL)

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__default_rounds=12,
    bcrypt__min_rounds=12
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")

try:
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI client initialized successfully")
except Exception as e:
    print(f"Error initializing OpenAI client: {str(e)}")
    raise

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory for PDFs if it doesn't exist
os.makedirs("static/pdfs", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize SQLAlchemy engine
engine = create_engine(SYNC_DATABASE_URL)
db = SQLDatabase(engine)

# Initialize healthcare agent
healthcare_agent = None  # Will be initialized during startup

# Initialize ChatOpenAI with deterministic settings
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    temperature=0,  # Set to 0 for maximum consistency
    model="gpt-3.5-turbo",
    verbose=True,
    model_kwargs={
        "top_p": 1.0,  # Use greedy sampling
        "presence_penalty": 0,
        "frequency_penalty": 0
    }
)

# Custom prompt template for healthcare queries
_DEFAULT_TEMPLATE = """You are a friendly healthcare assistant. Your task is to help users understand their healthcare information in simple, clear terms.

IMPORTANT - These are the EXACT table names and structures you MUST use:
1. VaccinationTakenRecords (NOT VaccinationRecords):
   - record_id, child_id, vaccine_name, dose_number, date_administered, administered_by, reminder_sent, notes

2. VaccineSchedule:
   - schedule_id, vaccine_name, dose_number, recommended_age_months, mandatory, description, benefits

3. Children:
   - child_id, first_name, last_name, date_of_birth, gender, blood_type, pincode, email

4. HealthcareProviders:
   - provider_id, provider_name, specialty, address, pincode, phone, email

For vaccine-related queries, ALWAYS:
1. Use VaccinationTakenRecords (not VaccinationRecords)
2. Join with VaccineSchedule using both vaccine_name AND dose_number
3. Compare dates using CURDATE() for current date
4. Use child_id = :child_id in WHERE clause
5. Handle NULL cases in LEFT JOINs properly

RESPONSE FORMATTING RULES:
1. NEVER show SQL queries in the response
2. NEVER show database IDs or technical details
3. NEVER mention table names or database terms
4. Use natural, conversational language
5. Format dates as "Month Day, Year" (e.g., "March 15, 2024")
6. Use bullet points for lists
7. Group related information together
8. Highlight important warnings or alerts
9. Use consistent, non-technical terminology
10. Focus on what's relevant to the parent/child

Internal format (not shown to user):
Question: "{input}"
SQLQuery: <write the SQL query>
SQLResult: <show the query results>
Answer: <write ONLY the user-friendly response following the rules above>

Question: {input}"""

PROMPT = PromptTemplate(
    input_variables=["input", "table_info"],
    template=_DEFAULT_TEMPLATE
)

# Initialize SQLDatabaseChain with stricter settings
db_chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    prompt=PROMPT,
    verbose=True,
    return_intermediate_steps=True,
    use_query_checker=True,
    top_k=10  # Limit number of results for consistency
)

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "indalavenkatasrisaihemanth@gmail.com"
SMTP_PASSWORD = "gydp dyzg rvsm towz"
EMAIL_FROM = "indalavenkatasrisaihemanth@gmail.com"

# Schema definition
schema = {
    "Children": ["child_id", "first_name", "last_name", "date_of_birth", "gender", "blood_type", "pincode", "email"],
    "Appointments": ["appointment_id", "child_id", "provider_id", "provider_name", "appointment_date", "notes", "status", "created_at"],
    "HealthAlerts": ["alert_id", "child_id", "disease", "confidence", "matching_symptoms", "reported_at"],
    "HealthcareProviders": ["provider_id", "provider_name", "specialty", "address", "pincode", "phone", "email"],
}

all_tables = set(schema.keys())
all_columns = {col for cols in schema.values() for col in cols}

# Pydantic Models
class Provider(BaseModel):
    provider_id: int
    provider_name: str
    specialty: Optional[str]
    address: Optional[str]
    pincode: Optional[str]
    phone: Optional[str]
    email: Optional[str]

class SymptomDetail(BaseModel):
    name: str
    severity: Optional[str] = None
    duration: Optional[str] = None
    context: Optional[str] = None

class SymptomAnalysisRequest(BaseModel):
    callId: str
    symptoms: List[SymptomDetail]

    class Config:
        json_schema_extra = {
            "example": {
                "callId": "call_123",
                "symptoms": [
                    {
                        "name": "fever",
                        "severity": "high",
                        "duration": "2 days",
                        "context": "started after exposure to rain"
                    }
                ]
            }
        }

class VaccineScheduleCreate(BaseModel):
    vaccine_name: str
    dose_number: int
    recommended_age_months: int
    mandatory: bool
    description: str
    benefits: str

class ImmunizationSummaryResponse(BaseModel):
    child_id: int
    first_name: str
    last_name: str
    summary: str

class ChildCreate(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: str  # YYYY-MM-DD
    gender: str
    blood_type: str
    pincode: str
    email: str
    username: str
    password: str

    @validator('username')
    def username_must_be_valid(cls, v):
        if not v:
            raise ValueError('Username cannot be empty')
        if not re.match("^[a-zA-Z0-9_]{4,20}$", v):
            raise ValueError('Username must be 4-20 characters long and contain only letters, numbers, and underscores')
        return v

    @validator('password')
    def password_must_be_strong(cls, v):
        if not v:
            raise ValueError('Password cannot be empty')
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search("[A-Z]", v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search("[a-z]", v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search("[0-9]", v):
            raise ValueError('Password must contain at least one number')
        return v

    @validator('email')
    def email_must_be_valid(cls, v):
        if not v:
            raise ValueError('Email cannot be empty')
        if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", v):
            raise ValueError('Invalid email format')
        return v

    @validator('date_of_birth')
    def validate_date_of_birth(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Invalid date format. Use YYYY-MM-DD')

    @validator('gender')
    def validate_gender(cls, v):
        valid_genders = ['male', 'female', 'other']
        if v.lower() not in valid_genders:
            raise ValueError('Gender must be one of: male, female, other')
        return v.lower()

    @validator('blood_type')
    def validate_blood_type(cls, v):
        valid_blood_types = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        if v not in valid_blood_types:
            raise ValueError('Invalid blood type. Must be one of: ' + ', '.join(valid_blood_types))
        return v

    @validator('pincode')
    def validate_pincode(cls, v):
        if not v:
            raise ValueError('Pincode cannot be empty')
        if not re.match(r'^\d{6}$', v):
            raise ValueError('Pincode must be 6 digits')
        return v

class Summary(BaseModel):
    call_id: str
    summary: str
    startedAt: str  # ISO format
    endedAt: Optional[str] = None
    recordingUrl: Optional[str] = None
    stereoRecordingUrl: Optional[str] = None
    transcript: Optional[str] = None
    status: Optional[str] = None
    endedReason: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class Symptom(BaseModel):
    name: str
    severity: Optional[str]
    duration: Optional[str]
    context: Optional[str]

class HealthAlert(BaseModel):
    disease: str
    confidence: float
    matching_symptoms: List[Dict]
    recommendation: str

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class QueryRequest(BaseModel):
    query: str

class VaccineDetailResponse(BaseModel):
    vaccine_name: str
    description: Optional[str] = None
    benefits: Optional[str] = None

class AppointmentCreate(BaseModel):
    child_id: int
    appointment_date: str  # ISO format date string
    provider_id: int
    schedule_id: int

class UpcomingVaccine(BaseModel):
    child_id: int
    schedule_id: int
    first_name: str
    last_name: str
    vaccine_name: str
    dose_number: int
    due_date: str
    appointment_booked: bool

# Provider Models
class ProviderProfile(BaseModel):
    provider_id: int
    provider_name: str
    specialty: Optional[str]
    address: Optional[str]
    pincode: Optional[str]
    phone: Optional[str]
    email: Optional[str]

class AssignedPatient(BaseModel):
    child_id: int
    first_name: str
    last_name: str
    date_of_birth: date
    last_visit: Optional[datetime]
    has_alerts: bool

class PatientAlert(BaseModel):
    patient_name: str
    disease_name: str
    confidence_score: float
    matching_symptoms: List[Dict]
    created_at: datetime

# Add this with other Pydantic models
class ProviderSignup(BaseModel):
    username: str
    password: str
    provider_name: str
    specialty: Optional[str] = None
    address: Optional[str] = None
    pincode: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "dr_smith",
                "password": "SecurePass123",
                "provider_name": "Dr. John Smith",
                "specialty": "Pediatrician",
                "address": "123 Medical Center, Austin",
                "pincode": "78701",
                "phone": "512-555-0123",
                "email": "dr.smith@example.com"
            }
        }
    
    @validator('username')
    def username_must_be_valid(cls, v):
        if not v:
            raise ValueError('Username cannot be empty')
        if not re.match("^[a-zA-Z0-9_]{4,20}$", v):
            raise ValueError('Username must be 4-20 characters long and contain only letters, numbers, and underscores')
        return v
    
    @validator('password')
    def password_must_be_strong(cls, v):
        if not v:
            raise ValueError('Password cannot be empty')
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search("[A-Z]", v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search("[a-z]", v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search("[0-9]", v):
            raise ValueError('Password must contain at least one number')
        return v
    
    @validator('provider_name')
    def provider_name_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Provider name cannot be empty')
        if len(v) > 255:
            raise ValueError('Provider name must be less than 255 characters')
        return v.strip()
    
    @validator('specialty')
    def specialty_length(cls, v):
        if v and len(v) > 100:
            raise ValueError('Specialty must be less than 100 characters')
        return v if v else None
    
    @validator('email')
    def email_must_be_valid(cls, v):
        if v:
            if len(v) > 255:
                raise ValueError('Email must be less than 255 characters')
            if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", v):
                raise ValueError('Invalid email format. Please use a valid email address (e.g., doctor@hospital.com)')
        return v if v else None
    
    @validator('phone')
    def phone_must_be_valid(cls, v):
        if v:
            if len(v) > 20:
                raise ValueError('Phone number must be less than 20 characters')
            if not re.match(r"^[0-9+\-() ]{5,20}$", v):
                raise ValueError('Invalid phone number format. Examples: 512-555-0123, +1-512-555-0123')
        return v if v else None
    
    @validator('pincode')
    def pincode_length(cls, v):
        if v and len(v) > 10:
            raise ValueError('Pincode must be less than 10 characters')
        return v if v else None

# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Child/User authentication
async def get_child_user_by_username(username: str):
    query = "SELECT cu.child_id, cu.username, cu.password_hash, c.first_name, c.last_name FROM ChildUsers cu JOIN Children c ON cu.child_id = c.child_id WHERE cu.username = :username"
    return await database.fetch_one(query=query, values={"username": username})

async def authenticate_user(username: str, password: str):
    user = await get_child_user_by_username(username)
    if not user:
        return False
    if not verify_password(password, user["password_hash"]):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await get_child_user_by_username(username)
    if user is None:
        raise credentials_exception
    return user

# Provider authentication
async def get_provider_by_username(username: str):
    query = """
    SELECT pu.provider_user_id, pu.username, pu.password_hash, 
           hp.provider_id, hp.provider_name, hp.specialty, hp.address, hp.pincode, hp.phone, hp.email
    FROM HealthcareProviderUsers pu
    JOIN HealthcareProviders hp ON pu.provider_id = hp.provider_id
    WHERE pu.username = :username
    """
    return await database.fetch_one(query=query, values={"username": username})

async def authenticate_provider(username: str, password: str):
    provider = await get_provider_by_username(username)
    if not provider:
        return False
    if not verify_password(password, provider["password_hash"]):
        return False
    return provider

async def get_current_provider(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    provider = await get_provider_by_username(username)
    if provider is None:
        raise credentials_exception
    return provider

# Utility Functions

def months_between(start_date, end_date):
    rd = relativedelta(end_date, start_date)
    return rd.years * 12 + rd.months

def parse_iso_datetime(iso_str: str) -> str:
    if iso_str.endswith('Z'):
        iso_str = iso_str[:-1]
    dt = datetime.fromisoformat(iso_str)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

async def send_email(subject: str, body: str, to_emails: List[str], attachments: List[Dict] = None):
    """Send email with optional attachments"""
    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = EMAIL_FROM
        msg['To'] = ', '.join(to_emails)

        # Add body
        msg.attach(MIMEText(body, 'plain'))

        # Add attachments if any
        if attachments:
            for attachment in attachments:
                with open(attachment['path'], 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    
                    # Add header
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename="{attachment["filename"]}"'
                    )
                    msg.attach(part)

        # Connect to SMTP server and send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send email: {str(e)}"
        )

@validator("vaccine_name")
def vaccine_name_not_empty(cls, v):
    if not v.strip():
        raise ValueError("vaccine_name cannot be empty")
    return v

# Helper functions for disease matching
def calculate_match_score(symptoms: List[Dict], disease_pattern: Dict) -> float:
    try:
        # Debug print to see the actual data
        print(f"Disease pattern received: {disease_pattern}")
        print(f"Symptoms received: {symptoms}")

        # Ensure disease_pattern is a dictionary
        if not isinstance(disease_pattern, dict):
            print(f"Invalid disease_pattern type: {type(disease_pattern)}")
            return 0.0

        # Get the disease name for better error messages
        disease_name = disease_pattern.get('name', 'Unknown Disease')
            
        # Parse comma-separated symptoms into sets
        primary_symptoms = set(s.strip().lower() for s in disease_pattern.get('primary_symptoms', '').split(',') if s.strip())
        secondary_symptoms = set(s.strip().lower() for s in disease_pattern.get('secondary_symptoms', '').split(',') if s.strip())
        
        # Fixed weights
        weights = {
            "primary": 1.0,
            "secondary": 0.5
        }
        
        # Safely extract symptom names
        symptom_names = set()
        for symptom in symptoms:
            if isinstance(symptom, dict):
                name = symptom.get('symptom_name', '')
                if name:
                    symptom_names.add(name.lower())
            else:
                print(f"Invalid symptom format: {symptom}")
        
        # Debug print symptom sets
        print(f"Processed primary symptoms for {disease_name}: {primary_symptoms}")
        print(f"Processed secondary symptoms for {disease_name}: {secondary_symptoms}")
        print(f"Processed patient symptoms: {symptom_names}")
        
        if not primary_symptoms and not secondary_symptoms:
            print(f"No symptoms defined for disease: {disease_name}")
            return 0.0
            
        # Calculate matches
        primary_matches = len(primary_symptoms.intersection(symptom_names))
        secondary_matches = len(secondary_symptoms.intersection(symptom_names))
        
        # Calculate weighted scores
        primary_weight_sum = len(primary_symptoms) * weights["primary"]
        secondary_weight_sum = len(secondary_symptoms) * weights["secondary"]
        total_weight_sum = primary_weight_sum + secondary_weight_sum
        
        if total_weight_sum == 0:
            print(f"Total weight sum is 0 for disease: {disease_name}")
            return 0.0
            
        # Calculate final score
        total_score = (
            (primary_matches * weights["primary"]) + 
            (secondary_matches * weights["secondary"])
        ) / total_weight_sum
        
        print(f"Score calculation for {disease_name}:")
        print(f"Primary matches: {primary_matches}/{len(primary_symptoms)}")
        print(f"Secondary matches: {secondary_matches}/{len(secondary_symptoms)}")
        print(f"Final score: {total_score}")
        
        return total_score
        
    except Exception as e:
        print(f"Error calculating match score: {str(e)}")
        print(f"Disease pattern: {disease_pattern}")
        print(f"Symptoms: {symptoms}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return 0.0

def get_matching_symptoms(symptoms: List[Dict], pattern: Dict) -> List[Dict]:
    try:
        # Debug print
        print(f"Processing pattern: {pattern}")
        print(f"Processing symptoms: {symptoms}")
        
        # Parse comma-separated symptoms into sets
        primary_symptoms = set(s.strip().lower() for s in pattern.get('primary_symptoms', '').split(',') if s.strip())
        secondary_symptoms = set(s.strip().lower() for s in pattern.get('secondary_symptoms', '').split(',') if s.strip())
        
        pattern_symptoms = primary_symptoms.union(secondary_symptoms)
        print(f"Combined pattern symptoms: {pattern_symptoms}")
        
        # Only return symptoms that match and have all required fields
        matching_symptoms = []
        for symptom in symptoms:
            if not isinstance(symptom, dict):
                print(f"Invalid symptom format: {symptom}")
                continue
                
            symptom_name = symptom.get('symptom_name', '').lower()
            if symptom_name and symptom_name in pattern_symptoms:
                matching_symptoms.append({
                    "name": symptom_name,
                    "severity": symptom.get('severity', 'Not specified'),
                    "duration": symptom.get('duration', 'Not specified'),
                    "context": symptom.get('context', '')
                })
        
        print(f"Found matching symptoms: {matching_symptoms}")
        return matching_symptoms
        
    except Exception as e:
        print(f"Error getting matching symptoms: {str(e)}")
        print(f"Pattern: {pattern}")
        print(f"Symptoms: {symptoms}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return []

def generate_recommendation(disease: str, confidence: float) -> str:
    if confidence > 0.9:
        return "Urgent medical attention recommended"
    elif confidence > 0.8:
        return f"Schedule doctor consultation within 48 hours - Possible {disease}"
    elif confidence > 0.7:
        return f"Monitor symptoms and consult doctor if they persist - Potential {disease}"
    else:
        return "Continue monitoring symptoms"

async def immunization_status_summary_async(child_id: int, upcoming_window=2):
    # Fetch child info asynchronously
    query_child = """
    SELECT c.date_of_birth, c.first_name, c.last_name, c.gender
    FROM Children c
    WHERE c.child_id = :cid
    """
    child = await database.fetch_one(query=query_child, values={"cid": child_id})
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    dob = pd.to_datetime(child["date_of_birth"])
    first_name = child["first_name"]
    last_name = child["last_name"]
    gender = child["gender"]

    today = pd.to_datetime(datetime.today().date())
    print(f"Today: {today}")
    age_months = months_between(dob, today)

    # Fetch all vaccine schedule and taken records in one query for better performance
    schedule_query = """
    SELECT vs.vaccine_name, vs.dose_number, vs.recommended_age_months, vs.mandatory,
           vtr.date_administered, vtr.notes
    FROM VaccineSchedule vs
    LEFT JOIN VaccinationTakenRecords vtr 
        ON vs.vaccine_name = vtr.vaccine_name 
        AND vs.dose_number = vtr.dose_number 
        AND vtr.child_id = :child_id
    ORDER BY vs.recommended_age_months, vs.vaccine_name
    """
    
    vaccine_records = await database.fetch_all(
        schedule_query,
        values={"child_id": child_id}
    )

    # Initialize categorized vaccines and counts
    taken_on_time = []
    taken_late = []
    missed = []
    upcoming = []
    optional_missed = []
    
    # Initialize counters
    total_vaccines = 0
    completed_count = 0
    upcoming_count = 0

    for record in vaccine_records:
        total_vaccines += 1
        vaccine = record["vaccine_name"]
        dose = record["dose_number"]
        due_month = record["recommended_age_months"]
        is_mandatory = record["mandatory"]
        date_administered = record["date_administered"]
        notes = record["notes"]

        if pd.isna(due_month):
            continue

        vaccine_info = f"{vaccine} (Dose {dose})"
        if date_administered:
            completed_count += 1
            admin_date = pd.to_datetime(date_administered)
            admin_age = months_between(dob, admin_date)
            
            # Consider a 1-month grace period for "on time" vaccination
            if admin_age <= due_month + 1:
                status = "✅ Taken on time"
                if notes:
                    status += f" - Note: {notes}"
                taken_on_time.append(f"{vaccine_info}: {status}")
            else:
                delay = admin_age - due_month
                taken_late.append(f"{vaccine_info}: ⚠️ Taken {delay} months late")
        else:
            # Not taken yet
            if due_month <= age_months:
                if is_mandatory:
                    delay = age_months - due_month
                    missed.append(f"{vaccine_info}: 🚫 Overdue by {delay} months")
                else:
                    optional_missed.append(f"{vaccine_info}: ⚪ Optional vaccine not taken")
            elif due_month <= age_months + upcoming_window:
                upcoming_count += 1
                days_until_due = (dob + relativedelta(months=due_month) - today).days
                urgency = "❗" if days_until_due <= 7 else "🔜"
                upcoming.append(f"{vaccine_info}: {urgency} Due in {days_until_due} days")

    # Build comprehensive summary
    summary_parts = []
    
    # Add counts to the summary
    summary_parts.append(f"Statistics:\nTotal Vaccines: {total_vaccines}\nCompleted: {completed_count}\nUpcoming: {upcoming_count}")
    
    if upcoming:
        summary_parts.append("📅 Upcoming Vaccinations:\n" + "\n".join(upcoming))
    
    if taken_on_time:
        summary_parts.append("\n✅ Completed Vaccinations:\n" + "\n".join(taken_on_time))
    
    if taken_late:
        summary_parts.append("\n⚠️ Delayed Vaccinations:\n" + "\n".join(taken_late))
    
    if missed:
        summary_parts.append("\n🚫 Missed Mandatory Vaccinations:\n" + "\n".join(missed))
    
    if optional_missed:
        summary_parts.append("\n⚪ Optional Vaccinations Not Taken:\n" + "\n".join(optional_missed))

    if not summary_parts:
        summary = "No vaccination records available"
    else:
        child_info = f"Vaccination Summary for {first_name} {last_name} (Age: {age_months} months)"
        summary = child_info + "\n\n" + "\n\n".join(summary_parts)

    return {
        "first_name": first_name,
        "last_name": last_name,
        "summary": summary,
        "total_vaccines": total_vaccines,
        "completed_count": completed_count,
        "upcoming_count": upcoming_count
    }

# API Routes
@app.on_event("startup")
async def startup():
    try:
        # First connect to the database
        await database.connect()
        print("Database connected successfully")
        
        # Create database tables
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully")

        # Initialize healthcare agent
        global healthcare_agent
        healthcare_agent = {
            "process_query": process_agent_query,
            "process_provider_query": process_agent_query,
            "clear_conversation": lambda: None  # No-op since we don't maintain conversation state
        }
        print("Healthcare agent initialized successfully")
        
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

@app.on_event("shutdown")
async def shutdown():
    try:
        # Close database connection
        await database.disconnect()
        print("Database disconnected successfully")
    except Exception as e:
        print(f"Error during shutdown: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

@app.post("/signup", status_code=201)
async def signup(child: ChildCreate):
    # First create parent record
    insert_parent_query = """
    INSERT INTO Parents (email)
    VALUES (:email)
    """
    
    async with database.transaction():
        # Create parent first
        parent_id = await database.execute(
            query=insert_parent_query,
            values={"email": child.email}  # Using the email for parent record
        )
        
        # Now create child record with the new parent_id
        insert_child_query = """
        INSERT INTO Children (first_name, last_name, date_of_birth, gender, parent_id, blood_type, pincode, email)
        VALUES (:first_name, :last_name, :date_of_birth, :gender, :parent_id, :blood_type, :pincode, :email)
        """
        
        child_values = child.dict()
        child_values.pop("username")
        child_values.pop("password")
        child_values["parent_id"] = parent_id  # Use the newly created parent_id

        child_id = await database.execute(query=insert_child_query, values=child_values)
        
        # Create user credentials
        hashed_pw = get_password_hash(child.password)
        insert_user_query = """
        INSERT INTO ChildUsers (child_id, username, password_hash)
        VALUES (:child_id, :username, :password_hash)
        """
        await database.execute(query=insert_user_query, values={
            "child_id": child_id,
            "username": child.username,
            "password_hash": hashed_pw
        })

    return {"message": "Child user created successfully", "child_id": child_id, "parent_id": parent_id}

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"]}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/profile")
async def read_profile(current_user=Depends(get_current_user)):
    return {
        "child_id": current_user["child_id"],
        "username": current_user["username"],
        "first_name": current_user["first_name"],
        "last_name": current_user["last_name"],
    }

@app.post("/store-summary")
async def store_summary(
    summary_data: Summary,
    current_user=Depends(get_current_user)
):
    started_at = parse_iso_datetime(summary_data.startedAt)
    ended_at = parse_iso_datetime(summary_data.endedAt) if summary_data.endedAt else None

    insert_query = """
    INSERT INTO Summaries (
        call_id, 
        child_id,
        summary, 
        startedAt, 
        endedAt,
        recordingUrl,
        stereoRecordingUrl,
        transcript,
        status,
        endedReason
    ) VALUES (
        :call_id,
        :child_id,
        :summary,
        :startedAt,
        :endedAt,
        :recordingUrl,
        :stereoRecordingUrl,
        :transcript,
        :status,
        :endedReason
    )
    """
    
    values = {
        "call_id": summary_data.call_id,
        "child_id": current_user["child_id"],
        "summary": summary_data.summary,
        "startedAt": started_at,
        "endedAt": ended_at,
        "recordingUrl": summary_data.recordingUrl,
        "stereoRecordingUrl": summary_data.stereoRecordingUrl,
        "transcript": summary_data.transcript,
        "status": summary_data.status,
        "endedReason": summary_data.endedReason
    }

    try:
        await database.execute(query=insert_query, values=values)
        return {"message": "Summary stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summaries", response_model=List[Summary])
async def fetch_summaries(current_user=Depends(get_current_user)):
    query = """
    SELECT 
        call_id, 
        summary, 
        startedAt, 
        endedAt,
        recordingUrl,
        stereoRecordingUrl,
        transcript,
        status,
        endedReason
    FROM Summaries
    WHERE child_id = :child_id 
    AND summary IS NOT NULL 
    AND summary != '' 
    AND summary != 'No summary available.'
    ORDER BY startedAt DESC
    """
    results = await database.fetch_all(query=query, values={"child_id": current_user["child_id"]})
    
    summaries = []
    for row in results:
        summary = {
            "call_id": row["call_id"],
            "summary": row["summary"],
            "startedAt": row["startedAt"].isoformat() if row["startedAt"] else None,
            "endedAt": row["endedAt"].isoformat() if row["endedAt"] else None,
            "recordingUrl": row["recordingUrl"],
            "stereoRecordingUrl": row["stereoRecordingUrl"],
            "transcript": row["transcript"],
            "status": row["status"],
            "endedReason": row["endedReason"]
        }
        summaries.append(summary)

    return summaries

@app.get("/upcoming-vaccines", response_model=List[UpcomingVaccine])
async def get_upcoming_vaccines(current_user=Depends(get_current_user)):
    query_dob = "SELECT date_of_birth, first_name, last_name FROM Children WHERE child_id = :child_id"
    child = await database.fetch_one(query=query_dob, values={"child_id": current_user["child_id"]})
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    dob = pd.to_datetime(child["date_of_birth"])
    first_name = child["first_name"]
    last_name = child["last_name"]

    today = datetime.utcnow().date()
    thirty_days_later = today + timedelta(days=30)
    
    # Calculate the age range in months for the next 30 days
    current_age_months = months_between(dob.date(), today)
    age_in_30_days = months_between(dob.date(), thirty_days_later)

    query_schedule = """
    SELECT schedule_id, vaccine_name, dose_number, recommended_age_months, mandatory
    FROM VaccineSchedule
    WHERE recommended_age_months BETWEEN :current_age AND :future_age
    ORDER BY recommended_age_months
    """
    schedules = await database.fetch_all(
        query=query_schedule, 
        values={
            "current_age": current_age_months,
            "future_age": age_in_30_days
        }
    )

    query_taken = """
    SELECT vaccine_name, dose_number FROM VaccinationTakenRecords WHERE child_id = :child_id
    """
    taken_records = await database.fetch_all(query=query_taken, values={"child_id": current_user["child_id"]})
    taken_set = {(r["vaccine_name"], r["dose_number"]) for r in taken_records}

    query_appointments = """
    SELECT schedule_id FROM Appointments WHERE child_id = :child_id AND status = 'scheduled'
    """
    appointment_rows = await database.fetch_all(query=query_appointments, values={"child_id": current_user["child_id"]})
    appointment_ids = {r["schedule_id"] for r in appointment_rows}

    result = []
    for s in schedules:
        if (s["vaccine_name"], s["dose_number"]) in taken_set:
            continue
        due_date = (dob + relativedelta(months=+s["recommended_age_months"])).date().isoformat()
        # Only include if due date is within the next 30 days
        if today <= datetime.strptime(due_date, '%Y-%m-%d').date() <= thirty_days_later:
            result.append({
                "child_id": current_user["child_id"],
                "schedule_id": s["schedule_id"],
                "first_name": first_name,
                "last_name": last_name,
                "vaccine_name": s["vaccine_name"],
                "dose_number": s["dose_number"],
                "due_date": due_date,
                "appointment_booked": s["schedule_id"] in appointment_ids
            })

    return result

@app.get("/child/nearby-providers", response_model=List[Provider])
async def get_nearby_providers(current_user=Depends(get_current_user)):
    child_id = current_user["child_id"]
    query_child_pincode = "SELECT pincode FROM Children WHERE child_id = :child_id"
    child = await database.fetch_one(query=query_child_pincode, values={"child_id": child_id})
    if not child or not child["pincode"]:
        raise HTTPException(status_code=404, detail="Child or pincode not found")
    pincode_str = child["pincode"]

    query_providers_exact = """
    SELECT provider_id, provider_name, specialty, address, pincode, phone, email
    FROM HealthcareProviders
    WHERE pincode = :pincode
    """
    providers = await database.fetch_all(query=query_providers_exact, values={"pincode": pincode_str})
    if providers:
        return [Provider(**dict(p)) for p in providers]

    try:
        pincode_num = int(pincode_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pincode format for child")
    low = pincode_num - 2
    high = pincode_num + 2

    query_providers_range = """
    SELECT provider_id, provider_name, specialty, address, pincode, phone, email
    FROM HealthcareProviders
    WHERE CAST(pincode AS UNSIGNED) BETWEEN :low AND :high
    """
    providers_range = await database.fetch_all(query=query_providers_range, values={"low": low, "high": high})
    return [Provider(**dict(p)) for p in providers_range]

@app.get("/immunization-summary")
async def get_immunization_summary(current_user=Depends(get_current_user)):
    result = await immunization_status_summary_async(current_user["child_id"])
    print(result["completed_count"])
    print(result["upcoming_count"])
    
    # Enhanced response format for the frontend
    return {
        "child_id": current_user["child_id"],
        "first_name": result["first_name"],
        "last_name": result["last_name"],
        "summary": result["summary"],
        "statistics": {
            "total_vaccines": result["total_vaccines"],
            "completed_count": result["completed_count"],
            "upcoming_count": result["upcoming_count"],
            "completion_percentage": round((result["completed_count"] / result["total_vaccines"]) * 100 if result["total_vaccines"] > 0 else 0, 1)
        },
        "status": {
            "isUpToDate": result["completed_count"] == result["total_vaccines"],
            "hasUpcoming": result["upcoming_count"] > 0,
            "nextDueDate": result.get("next_due_date"),
            "lastVaccinationDate": result.get("last_vaccination_date")
        }
    } 

@app.post("/vaccine-schedule", status_code=201)
async def add_vaccine_schedule(
    vaccine_data: VaccineScheduleCreate,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
):
    insert_query = """
    INSERT INTO VaccineSchedule (vaccine_name, dose_number, recommended_age_months, mandatory)
    VALUES (:vaccine_name, :dose_number, :recommended_age_months, :mandatory)
    """
    values = vaccine_data.dict()
    await database.execute(query=insert_query, values=values)

    background_tasks.add_task(
        notify_all_children_new_vaccine,
        vaccine_name=vaccine_data.vaccine_name,
        dose_number=vaccine_data.dose_number,
        recommended_age_months=vaccine_data.recommended_age_months,
    )

    return {"message": "Vaccine schedule added and notifications sent."}

async def generate_sql_response(query: str, schema: Dict) -> Tuple[str, Any, str]:
    try:
        # Create schema information for context
        schema_info = "Available tables and their columns:\n"
        for table, columns in schema.items():
            schema_info += f"\n{table}: {', '.join(columns)}"
        
        # Generate SQL query using OpenAI
        sql_messages = [
            {"role": "system", "content": f"""You are a SQL query generator for a child healthcare system. Generate SQL queries based on natural language questions.
            {schema_info}
            
            Rules:
            1. ONLY return the SQL query, nothing else
            2. Use proper table names and columns from the schema
            3. Join tables when necessary using appropriate keys
            4. Use appropriate WHERE clauses for filtering
            5. Handle NULL values appropriately
            6. Format dates using MySQL date functions when needed
            7. Ensure child_id is properly filtered when querying personal data"""},
            {"role": "user", "content": query}
        ]

        sql_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=sql_messages,
            temperature=0,
        )
        
        generated_sql = sql_response.choices[0].message.content.strip()
        
        # Validate table and column names
        sql_lower = generated_sql.lower()
        for table in schema.keys():
            if table.lower() in sql_lower:
                # For each referenced table, validate its columns
                for column in schema[table]:
                    if column.lower() in sql_lower:
                        continue
        
        # Execute the validated query
        results = await database.fetch_all(query=generated_sql)
        
        # Format results for natural language response
        results_str = json.dumps([dict(row) for row in results], default=str)
        
        # Generate natural language response with full context
        response_messages = [
            {"role": "system", "content": """You are a helpful pediatric healthcare assistant that explains medical data in natural language.
            Format the response in a clear, friendly way that parents can understand.
            If there are multiple results, summarize them appropriately.
            If there are no results, explain that clearly.
            Always maintain a caring and supportive tone.
            
            When responding:
            1. Consider both the original question and the SQL results
            2. Explain the findings in a way that directly answers the original question
            3. Provide relevant context and explanations for medical terms
            4. If appropriate, suggest follow-up actions or recommendations
            5. Keep the tone professional but friendly"""},
            {"role": "user", "content": f"""Original Question: {query}
            
SQL Query Used: {generated_sql}

Query Results: {results_str}

Please provide a comprehensive response that answers the original question using this data."""}
        ]
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=response_messages,
            temperature=0.7,
        )
        
        return generated_sql, results, response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error in generate_sql_response: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/chat-assistant")
async def chat_assistant(request: QueryRequest, current_user=Depends(get_current_user)):
    try:
        # Get the last message from the user
        user_query = request.query
        
        # Add context about the current user
        contextualized_query = f"For child_id {current_user['child_id']}: {user_query}"
        
        # Generate SQL, execute it, and get natural language response
        sql_query, query_result, final_answer = await generate_sql_response(contextualized_query, schema)
        
        # Format the response to match frontend expectations
        response = {
            "response": final_answer,
            "generated_sql": sql_query,
            "query_result": query_result,
            "message_history": [{"role": "user", "content": user_query}, {"role": "assistant", "content": final_answer}]
        }

        return response

    except Exception as e:
        error_msg = str(e)
        if "invalid_api_key" in error_msg.lower():
            raise HTTPException(
                status_code=500,
                detail="Error with API key configuration. Please contact support."
            )
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {error_msg}"
        )

@app.get("/provider/disease-outbreak", response_model=List[dict])
async def disease_outbreak(current_provider=Depends(get_current_provider)):
    today = datetime.utcnow().date()
    thirty_days_ago = today - timedelta(days=30)

    query = """
    SELECT disease, COUNT(*) AS disease_count
    FROM diseasesDiagnosedChildren
    WHERE diagnosed_date BETWEEN :thirty_days_ago AND :today
    GROUP BY disease
    HAVING disease_count >= 5
    """

    try:
        results = await database.fetch_all(query, values={"thirty_days_ago": thirty_days_ago, "today": today})
        if not results:
            return []
        diseases = [{"disease": row["disease"], "count": row["disease_count"]} for row in results]
        return diseases
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/vaccine-details", response_model=VaccineDetailResponse)
async def get_vaccine_details(vaccine_name: str = Query(..., description="Name of the vaccine")):
    query = """
    SELECT vaccine_name, description, benefits
    FROM VaccineSchedule
    WHERE vaccine_name = :vaccine_name
    LIMIT 1
    """
    row = await database.fetch_one(query=query, values={"vaccine_name": vaccine_name})
    if not row:
        raise HTTPException(status_code=404, detail="Vaccine not found")

    return VaccineDetailResponse(
        vaccine_name=row["vaccine_name"],
        description=row["description"],
        benefits=row["benefits"],
    )

async def send_appointment_confirmation_emails(child_id: int, provider_id: int, appointment_date: str, schedule_id: int):
    # Get child details
    child_query = """
    SELECT c.first_name, c.last_name, c.email
    FROM Children c
    WHERE c.child_id = :child_id
    """
    child = await database.fetch_one(child_query, values={"child_id": child_id})
    
    # Get provider details
    provider_query = """
    SELECT provider_name, email
    FROM HealthcareProviders
    WHERE provider_id = :provider_id
    """
    provider = await database.fetch_one(provider_query, values={"provider_id": provider_id})
    
    # Get vaccine details
    vaccine_query = """
    SELECT vaccine_name, dose_number
    FROM VaccineSchedule
    WHERE schedule_id = :schedule_id
    """
    vaccine = await database.fetch_one(vaccine_query, values={"schedule_id": schedule_id})
    
    if child and provider and vaccine:
        # Format date for email
        appointment_datetime = datetime.strptime(appointment_date, "%Y-%m-%d")
        formatted_date = appointment_datetime.strftime("%B %d, %Y")
        
        # Email to child/parent
        child_subject = f"Appointment Confirmation - {vaccine['vaccine_name']} Vaccination"
        child_body = f"""
Dear {child['first_name']} {child['last_name']},

Your appointment has been scheduled for {vaccine['vaccine_name']} (Dose {vaccine['dose_number']}).

Details:
- Date: {formatted_date}
- Healthcare Provider: {provider['provider_name']}

Please arrive 10 minutes before your scheduled time.

Best regards,
AskHealth Team
        """
        
        # Email to provider
        provider_subject = f"New Appointment Scheduled - {vaccine['vaccine_name']}"
        provider_body = f"""
Dear {provider['provider_name']},

A new vaccination appointment has been scheduled.

Details:
- Patient: {child['first_name']} {child['last_name']}
- Vaccine: {vaccine['vaccine_name']} (Dose {vaccine['dose_number']})
- Date: {formatted_date}

Best regards,
AskHealth Team
        """
        
        # Send emails
        if child['email']:
            send_email(child_subject, child_body, [child['email']])
        if provider['email']:
            send_email(provider_subject, provider_body, [provider['email']])

@app.post("/appointments", status_code=201)
async def create_appointment(
    appointment: AppointmentCreate,
    current_user=Depends(get_current_user)
):
    insert_query = """
    INSERT INTO Appointments (child_id, appointment_date, provider_id, schedule_id)
    VALUES (:child_id, :appointment_date, :provider_id, :schedule_id)
    """
    
    async with database.transaction():
        await database.execute(query=insert_query, values=appointment.dict())
        # Send confirmation emails
        await send_appointment_confirmation_emails(
            appointment.child_id,
            appointment.provider_id,
            appointment.appointment_date,
            appointment.schedule_id
        )
    
    return {"message": "Appointment scheduled successfully"}

def calculate_age(birth_date):
    today = datetime.now().date()
    age = relativedelta(today, birth_date)
    if age.years > 0:
        return f"{age.years} years"
    elif age.months > 0:
        return f"{age.months} months"
    else:
        return f"{age.days} days"

@app.post("/analyze-symptoms")
async def analyze_symptoms(data: SymptomAnalysisRequest):
    try:
        # Extract symptoms from the conversation
        extracted_data = {
            "symptoms": []
        }
        
        for symptom in data.symptoms:
            extracted_data["symptoms"].append({
                "call_id": data.callId,
                "symptom_name": symptom.name.lower(),
                "severity": symptom.severity,
                "duration": symptom.duration,
                "context": symptom.context
            })
            
        return {"status": "success", "symptoms": extracted_data["symptoms"]}
        
    except Exception as e:
        print(f"Error in analyze_symptoms: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check-health-alerts")
async def check_health_alerts(current_user=Depends(get_current_user), call_id: str = None):
    try:
        # Get symptoms only from the current call
        query = """
        SELECT symptom_name, severity, duration, context, first_reported
        FROM ExtractedSymptoms
        WHERE child_id = :child_id
        AND call_id = :call_id
        ORDER BY first_reported DESC
        """
        recent_symptoms = await database.fetch_all(
            query=query, 
            values={
                "child_id": current_user["child_id"],
                "call_id": call_id
            }
        )
        
        # Convert database rows to list of dictionaries
        symptoms_list = [
            {
                "symptom_name": row["symptom_name"],
                "severity": row["severity"] or "Not specified",
                "duration": row["duration"] or "Not specified",
                "context": row["context"] or ""
            }
            for row in recent_symptoms
        ]
        
        # Get disease patterns
        patterns_query = """
        SELECT name, primary_symptoms, secondary_symptoms
        FROM Diseases
        """
        disease_patterns = await database.fetch_all(query=patterns_query)
        
        if not disease_patterns:
            print("No disease patterns found in database")
            return {"alerts": []}
        
        alerts = []
        for pattern in disease_patterns:
            # Convert Record object to dictionary
            pattern_dict = {
                "name": pattern["name"],
                "primary_symptoms": pattern["primary_symptoms"],
                "secondary_symptoms": pattern["secondary_symptoms"]
            }
            
            match_score = calculate_match_score(symptoms_list, pattern_dict)
            if match_score >= 0.7:  # Threshold for potential match
                matching_symptoms = get_matching_symptoms(symptoms_list, pattern_dict)
                if matching_symptoms:  # Only add alert if there are matching symptoms
                    alert = {
                        "disease": pattern_dict["name"],
                        "confidence": round(match_score * 100, 2),
                        "matching_symptoms": matching_symptoms,
                        "recommendation": generate_recommendation(pattern_dict["name"], match_score)
                    }
                    alerts.append(alert)
        
        if alerts:
            for alert in alerts:
                check_query = """
                SELECT 1 FROM HealthAlerts 
                WHERE child_id = :child_id 
                AND disease_name = :disease
                AND created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                """
                existing_alert = await database.fetch_one(
                    query=check_query,
                    values={
                        "child_id": current_user["child_id"],
                        "disease": alert["disease"]
                    }
                )
                
                if not existing_alert:
                    store_alert_query = """
                    INSERT INTO HealthAlerts 
                    (child_id, disease_name, confidence_score, matching_symptoms, created_at)
                    VALUES (:child_id, :disease, :confidence, :symptoms, NOW())
                    """
                    await database.execute(
                        query=store_alert_query,
                        values={
                            "child_id": current_user["child_id"],
                            "disease": alert["disease"],
                            "confidence": alert["confidence"],
                            "symptoms": json.dumps(alert["matching_symptoms"])
                        }
                    )
        
        return {"alerts": alerts}
        
    except Exception as e:
        print(f"Error in check_health_alerts: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health-alerts")
async def get_health_alerts(current_user=Depends(get_current_user)):
    query = """
    SELECT 
        ha.alert_id,
        ha.disease_name,
        ha.confidence_score,
        ha.matching_symptoms,
        ha.created_at,
        CASE 
            WHEN a.appointment_id IS NOT NULL THEN 'booked'
            ELSE 'pending'
        END as booking_status,
        CONVERT_TZ(a.appointment_date, '+00:00', @@session.time_zone) as appointment_date,
        hp.provider_name,
        hp.address,
        hp.phone
    FROM HealthAlerts ha
    LEFT JOIN Appointments a ON ha.alert_id = a.healthalert_id AND a.status = 'scheduled'
    LEFT JOIN HealthcareProviders hp ON a.provider_id = hp.provider_id
    WHERE ha.child_id = :child_id
    ORDER BY ha.created_at DESC
    LIMIT 10
    """
    alerts = await database.fetch_all(
        query=query,
        values={"child_id": current_user["child_id"]}
    )
    
    return {
        "alerts": [
            {
                "id": alert["alert_id"],
                "disease": alert["disease_name"],
                "confidence": alert["confidence_score"],
                "matching_symptoms": json.loads(alert["matching_symptoms"]),
                "created_at": alert["created_at"].isoformat(),
                "booking_status": alert["booking_status"],
                "appointment": {
                    "date": alert["appointment_date"].isoformat() if alert["appointment_date"] else None,
                    "provider_name": alert["provider_name"],
                    "address": alert["address"],
                    "phone": alert["phone"]
                } if alert["booking_status"] == "booked" else None
            }
            for alert in alerts
        ]
    }

async def notify_all_children_new_vaccine(vaccine_name: str, dose_number: int, recommended_age_months: int):
    # Fetch all children emails
    query = "SELECT email, first_name FROM Children WHERE email IS NOT NULL"
    children = await database.fetch_all(query=query)
    emails = [child["email"] for child in children if child["email"]]

    subject = f"New Vaccine Schedule Added: {vaccine_name} Dose {dose_number}"
    body = (
        f"Dear Parent,\n\n"
        f"We have added a new vaccine to the immunization schedule:\n"
        f"- Vaccine: {vaccine_name}\n"
        f"- Dose Number: {dose_number}\n"
        f"- Recommended Age (months): {recommended_age_months}\n\n"
        f"Please consult your pediatrician for more details.\n\n"
        f"Best regards,\n"
        f"Mother & Child Care Team"
    )

    if emails:
        # For large numbers of emails, consider batching or async email sending
        send_email(subject, body, emails)

# Provider routes
@app.post("/provider/login", response_model=Token)
async def provider_login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        print(f"\n=== Provider Login Debug ===")
        print(f"Login attempt for username: {form_data.username}")
        
        # Get provider details
        provider = await get_provider_by_username(form_data.username)
        
        if not provider:
            print("No provider found with this username")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        print(f"Found provider: {provider['username']}")
        print(f"Stored hash: {provider['password_hash']}")
        
        # Verify password
        is_password_correct = verify_password(form_data.password, provider['password_hash'])
        print(f"Password verification result: {is_password_correct}")
        
        if not is_password_correct:
            print("Password verification failed")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        print(f"Authentication successful for provider: {provider['username']}")
        
        # Generate token with provider-specific claims
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={
                "sub": provider["username"],
                "type": "provider",
                "provider_id": provider["provider_id"]
            },
            expires_delta=access_token_expires
        )
        
        print("Login successful - token generated")
        print("=== Debug End ===\n")
        
        return {"access_token": access_token, "token_type": "bearer"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in provider_login: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )

@app.get("/provider/profile", response_model=ProviderProfile)
async def read_provider_profile(current_provider=Depends(get_current_provider)):
    return {
        "provider_id": current_provider["provider_id"],
        "provider_name": current_provider["provider_name"],
        "specialty": current_provider["specialty"],
        "address": current_provider["address"],
        "pincode": current_provider["pincode"],
        "phone": current_provider["phone"],
        "email": current_provider["email"]
    }

@app.post("/provider/signup", status_code=201, response_model=dict)
async def provider_signup(provider: ProviderSignup):
    """
    Register a new healthcare provider.
    """
    try:
        # Debug logging
        print("\n=== Provider Signup Debug ===")
        print(f"Received request data: {provider.dict()}")
        
        # Check if username already exists
        check_query = "SELECT 1 FROM HealthcareProviderUsers WHERE username = :username"
        existing_user = await database.fetch_one(check_query, values={"username": provider.username})
        if existing_user:
            print(f"Username {provider.username} already exists")
            raise HTTPException(
                status_code=400,
                detail="Username already registered"
            )
            
        # Check if email already exists (only if email is provided)
        if provider.email:
            check_email_query = "SELECT 1 FROM HealthcareProviders WHERE email = :email"
            existing_email = await database.fetch_one(check_email_query, values={"email": provider.email})
            if existing_email:
                print(f"Email {provider.email} already exists")
                raise HTTPException(
                    status_code=400,
                    detail="Email already registered"
                )
            
        async with database.transaction():
            # First, insert into HealthcareProviders
            provider_query = """
            INSERT INTO HealthcareProviders 
            (provider_name, specialty, address, pincode, phone, email)
            VALUES (:provider_name, :specialty, :address, :pincode, :phone, :email)
            """
            provider_values = {
                "provider_name": provider.provider_name,
                "specialty": provider.specialty,
                "address": provider.address,
                "pincode": provider.pincode,
                "phone": provider.phone,
                "email": provider.email
            }
            print(f"Inserting provider with values: {provider_values}")
            provider_id = await database.execute(provider_query, values=provider_values)
            print(f"Created provider with ID: {provider_id}")
            
            # Then, insert into HealthcareProviderUsers
            user_query = """
            INSERT INTO HealthcareProviderUsers 
            (provider_id, username, password_hash)
            VALUES (:provider_id, :username, :password_hash)
            """
            user_values = {
                "provider_id": provider_id,
                "username": provider.username,
                "password_hash": get_password_hash(provider.password)
            }
            print(f"Inserting provider user with values: {user_values}")
            await database.execute(user_query, values=user_values)
            
        # Generate token for immediate login
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={
                "sub": provider.username,
                "type": "provider",
                "provider_id": provider_id
            },
            expires_delta=access_token_expires
        )
        
        print("Provider signup successful")
        print("=== Debug End ===\n")
        
        return {
            "message": "Provider registered successfully",
            "provider_id": provider_id,
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except ValidationError as e:
        print(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in provider_signup: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while registering the provider"
        )

@app.post("/provider/store-summary")
async def store_provider_summary(
    summary_data: Summary,
    current_provider=Depends(get_current_provider)
):
    started_at = parse_iso_datetime(summary_data.startedAt)
    ended_at = parse_iso_datetime(summary_data.endedAt) if summary_data.endedAt else None

    insert_query = """
    INSERT INTO ProviderSummaries (
        call_id, 
        provider_id,
        summary, 
        startedAt, 
        endedAt,
        recordingUrl,
        stereoRecordingUrl,
        transcript,
        status,
        endedReason
    ) VALUES (
        :call_id,
        :provider_id,
        :summary,
        :startedAt,
        :endedAt,
        :recordingUrl,
        :stereoRecordingUrl,
        :transcript,
        :status,
        :endedReason
    )
    """
    
    values = {
        "call_id": summary_data.call_id,
        "provider_id": current_provider["provider_id"],
        "summary": summary_data.summary,
        "startedAt": started_at,
        "endedAt": ended_at,
        "recordingUrl": summary_data.recordingUrl,
        "stereoRecordingUrl": summary_data.stereoRecordingUrl,
        "transcript": summary_data.transcript,
        "status": summary_data.status,
        "endedReason": summary_data.endedReason
    }

    try:
        await database.execute(query=insert_query, values=values)
        return {"message": "Summary stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/provider/summaries", response_model=List[Summary])
async def fetch_provider_summaries(current_provider=Depends(get_current_provider)):
    query = """
    SELECT 
        call_id, 
        summary, 
        startedAt, 
        endedAt,
        recordingUrl,
        stereoRecordingUrl,
        transcript,
        status,
        endedReason
    FROM ProviderSummaries
    WHERE provider_id = :provider_id 
    AND summary IS NOT NULL 
    AND summary != '' 
    AND summary != 'No summary available.'
    ORDER BY startedAt DESC
    """
    results = await database.fetch_all(query=query, values={"provider_id": current_provider["provider_id"]})
    
    summaries = []
    for row in results:
        summary = {
            "call_id": row["call_id"],
            "summary": row["summary"],
            "startedAt": row["startedAt"].isoformat() if row["startedAt"] else None,
            "endedAt": row["endedAt"].isoformat() if row["endedAt"] else None,
            "recordingUrl": row["recordingUrl"],
            "stereoRecordingUrl": row["stereoRecordingUrl"],
            "transcript": row["transcript"],
            "status": row["status"],
            "endedReason": row["endedReason"]
        }
        summaries.append(summary)

    return summaries

@app.get("/provider/local-health-alerts")
async def get_provider_local_health_alerts(current_provider=Depends(get_current_provider)):
    try:
        # Get provider's pincode
        provider_query = """
        SELECT pincode FROM HealthcareProviders WHERE provider_id = :provider_id
        """
        provider_data = await database.fetch_one(
            query=provider_query,
            values={"provider_id": current_provider["provider_id"]}
        )

        if not provider_data:
            raise HTTPException(status_code=404, detail="Provider not found")

        # Get alerts from the same pincode area that haven't been confirmed yet
        query = """
        SELECT 
            ha.alert_id,
            ha.disease_name as disease,
            CONCAT(c.first_name, ' ', c.last_name) as patient_name,
            c.child_id,
            c.email as patient_email,
            ha.confidence_score,
            ha.matching_symptoms,
            ha.created_at
        FROM HealthAlerts ha
        JOIN Children c ON ha.child_id = c.child_id
        LEFT JOIN diseasesDiagnosedChildren ddc ON ha.alert_id = ddc.alert_id
        WHERE c.pincode = :pincode
        AND ddc.alert_id IS NULL  # Only get unconfirmed alerts
        ORDER BY ha.created_at DESC
        """

        alerts = await database.fetch_all(
            query=query,
            values={"pincode": provider_data["pincode"]}
        )

        return {
            "alerts": [
                {
                    "alert_id": alert["alert_id"],
                    "disease": alert["disease"],
                    "patient_name": alert["patient_name"],
                    "child_id": alert["child_id"],
                    "patient_email": alert["patient_email"],
                    "matching_symptoms": json.loads(alert["matching_symptoms"]) if alert["matching_symptoms"] else [],
                    "reported_at": alert["created_at"].isoformat() if alert["created_at"] else None
                }
                for alert in alerts
            ]
        }
    except Exception as e:
        print(f"Error fetching local health alerts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching local health alerts: {str(e)}"
        )

class AgentQueryRequest(BaseModel):
    query: str
    return_audio: bool = True  # Whether to return audio response
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Show me my appointments for today",
                "return_audio": True
            }
        }

@app.post("/agent/query/provider")
async def agent_query_provider(
    request: AgentQueryRequest,
    background_tasks: BackgroundTasks,
    current_provider=Depends(get_current_provider)
):
    """Handle provider agent queries"""
    try:
        # Process the query
        response = await process_agent_query(
            query=request.query,
            current_provider=current_provider,
            background_tasks=background_tasks
        )
        
        # Initialize response data
        response_data = {
            "response": response.get("message", ""),
            "type": response.get("type", "text"),
            "data": response.get("data", {})
        }

        # If it's an EHR request with a PDF URL, format it for better display
        if response.get("type") == "ehr_request" and response.get("data", {}).get("pdf_url"):
            pdf_url = response["data"]["pdf_url"]
            patient_name = response["data"]["patient_name"]
            response_data["response"] = (
                f"I've generated the EHR for {patient_name} and it's being sent to your email.\n\n"
                f"📄 <a href='{pdf_url}' target='_blank'>Click here to view the PDF</a>"
            )
            response_data["has_html"] = True

        # Generate audio if requested
        if request.return_audio:
            try:
                # Use plain text version for TTS
                tts_text = response_data["response"]
                if response_data.get("has_html"):
                    # Strip HTML for TTS
                    tts_text = re.sub(r'<[^>]+>', '', tts_text)
                
                audio = openai_client.audio.speech.create(
                    model="tts-1",
                    voice="nova",
                    input=tts_text
                )
                response_data["audio"] = base64.b64encode(audio.content).decode()
            except Exception as e:
                print(f"Error generating audio: {str(e)}")
                # Continue without audio if TTS fails

        return response_data
        
    except Exception as e:
        print(f"Error in agent query: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/agent/query/user")
async def agent_query_user(
    request: AgentQueryRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user)
):
    """Process a user query through the agent"""
    try:
        # Process the query
        response = await process_agent_query(
            query=request.query,
            background_tasks=background_tasks,
            current_user=current_user
        )
        
        # Generate audio response if requested
        if request.return_audio:
            audio_response = await text_to_speech(TTSRequest(text=response["message"]))
            response["audio"] = audio_response.get("audio")
            
        return response
        
    except Exception as e:
        print(f"Error processing user query: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing user query: {str(e)}"
        )

@app.post("/agent/clear")
async def clear_agent_conversation(current_user=Depends(get_current_user)):
    """
    Clear the agent's conversation history
    """
    try:
        if not healthcare_agent:
            raise HTTPException(status_code=500, detail="Agent not initialized")
            
        healthcare_agent["clear_conversation"]()
        return {"message": "Conversation history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-JennyNeural"  # Default voice
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, how can I help you today?",
                "voice": "en-US-JennyNeural"  # Azure Neural voice
            }
        }

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using OpenAI's TTS API"""
    try:
        if not request.text:
            raise HTTPException(
                status_code=400,
                detail="Text content is required"
            )

        try:
            # Generate speech using OpenAI's TTS API
            audio_response = openai_client.audio.speech.create(
                model="tts-1",
                voice="nova",  # Professional female voice
                input=request.text
            )
            
            # Create a byte stream from the response
            audio_stream = io.BytesIO(audio_response.content)
            audio_stream.seek(0)
            
            return StreamingResponse(
                audio_stream,
                media_type="audio/mpeg",
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Disposition": "attachment;filename=speech.mp3"
                }
            )
            
        except Exception as e:
            print(f"TTS error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating speech: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in text_to_speech: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the request"
        )

async def process_other_queries(query: str, current_provider=None, current_user=None):
    """Process non-EHR queries"""
    cleaned_query = query.lower().strip()
    response_text = []
    response_data = {}

    # Define keywords for different query types
    appointment_keywords = ["appointment", "appointments", "schedule", "scheduled", "visit"]
    vaccination_keywords = ["vaccine", "vaccination", "overdue", "pending", "immunization"]
    health_alert_keywords = ["alert", "alerts", "health alert", "outbreak", "disease"]
    
    # Check query types
    is_appointment_query = any(keyword in cleaned_query for keyword in appointment_keywords)
    is_vaccination_query = any(keyword in cleaned_query for keyword in vaccination_keywords)
    is_health_alert_query = any(keyword in cleaned_query for keyword in health_alert_keywords)
    
    # Handle appointments
    if is_appointment_query and current_provider:
        appointments = await get_provider_appointments("week", current_provider)
        if appointments:
            response_text.append("\n📅 Upcoming Appointments:")
            for date, appts in appointments.items():
                response_text.append(f"\n{date}:")
                for appt in appts:
                    response_text.append(f"• {appt['time']} - {appt['patient_name']}: {appt['appointment_type']}")
        else:
            response_text.append("\nNo upcoming appointments found.")
        response_data["appointments"] = appointments
    
    # Handle vaccinations
    if is_vaccination_query and current_provider:
        overdue = await get_overdue_vaccinations(current_provider)
        if overdue:
            response_text.append("\n🚨 Overdue Vaccinations:")
            for vacc in overdue[:5]:
                response_text.append(f"• {vacc['child_name']}: {vacc['vaccine_name']} (Dose {vacc['dose_number']})")
        else:
            response_text.append("\nNo overdue vaccinations found.")
        response_data["vaccinations"] = overdue
    
    # Handle health alerts
    if is_health_alert_query and current_provider:
        alerts = await get_provider_local_health_alerts(current_provider)
        if alerts.get("alerts"):
            response_text.append("\n🏥 Recent Health Alerts:")
            for alert in alerts["alerts"][:5]:
                response_text.append(f"• {alert['disease']} - {alert['confidence']}% confidence")
        else:
            response_text.append("\nNo recent health alerts found.")
        response_data["alerts"] = alerts.get("alerts", [])
    
    # If no specific query type matched
    if not any([is_appointment_query, is_vaccination_query, is_health_alert_query]):
        return {
            "message": "I can help you with:\n• Appointments\n• Vaccinations\n• Health Alerts\n• Electronic Health Records (EHR)\n\nPlease specify what you'd like to know about.",
            "type": "help"
        }
    
    return {
        "message": "\n".join(response_text) if response_text else "I processed your request but found no relevant information.",
        "type": "combined",
        "data": response_data
    }

async def process_agent_query(
    query: str = Query(..., description="The query to process"),
    current_provider=None,
    current_user=None,
    background_tasks: BackgroundTasks = None
):
    """Process a query from the agent"""
    try:
        print("\n=== Processing Provider Query ===")
        print(f"Input query: {query}")
        
        # Clean up the query
        cleaned_query = query.lower().strip()
        
        # Define all keywords for different query types
        appointment_keywords = ["appointment", "appointments", "schedule", "scheduled", "visit"]
        today_keywords = ["today", "today's", "todays", "current day"]
        vaccination_keywords = ["vaccine", "vaccination", "overdue", "pending", "immunization"]
        health_alert_keywords = ["alert", "alerts", "health alert", "outbreak", "disease"]
        vaccination_record_keywords = ["has taken", "received", "administered", "given", "took", "got"]
        ehr_keywords = ["ehr", "electronic health record", "health record", "medical record", "patient record", "fetch", "get", "send"]
        mmr_check_keywords = ["measles", "not vaccinated", "haven't received", "missing mmr", "has not taken"]
        
        # Check query types
        is_appointment_query = any(keyword in cleaned_query for keyword in appointment_keywords)
        is_today_query = any(keyword in cleaned_query for keyword in today_keywords)
        is_vaccination_query = any(keyword in cleaned_query for keyword in vaccination_keywords)
        is_health_alert_query = any(keyword in cleaned_query for keyword in health_alert_keywords)
        is_vaccination_record = any(keyword in cleaned_query for keyword in vaccination_record_keywords)
        is_ehr_request = any(keyword in cleaned_query for keyword in ehr_keywords)
        is_mmr_query = (
            ("mmr" in cleaned_query or "measles" in cleaned_query) and
            not any(kw in cleaned_query for kw in vaccination_record_keywords)
        )

        # Initialize response
        response_text = []
        response_data = {}

        # Handle vaccination recording first
        if is_vaccination_record and current_provider:
            print("\n=== Processing Vaccination Record ===")
            # Extract SSN and vaccine details
            ssn_pattern = r"(?:with|ssn|client|patient)\s*(?:the|ssn|number|#)?\s*(\d{4})"
            vaccine_pattern = r"(?:has taken|received|got|took|given|administered)\s+(?:vaccination|vaccine)?\s*(\w+(?:\s*\w+)*?)(?:\s*(?:vaccine|vaccination|shot))?\s*(?:dose\s+(?:(\d+)|one|two|three|four|five|six))?\s*(?:today)?"
            
            print(f"Query: {cleaned_query}")
            ssn_match = re.search(ssn_pattern, cleaned_query)
            vaccine_match = re.search(vaccine_pattern, cleaned_query, re.IGNORECASE)
            
            print(f"SSN Match: {ssn_match.group(1) if ssn_match else None}")
            print(f"Vaccine Match: {vaccine_match.groups() if vaccine_match else None}")
            
            if ssn_match and vaccine_match:
                ssn = ssn_match.group(1)
                # Extract and clean vaccine name, preserving multi-word names
                vaccine_name = vaccine_match.group(1).strip().upper()
                print(f"Extracted vaccine name: {vaccine_name}")
                
                # Special handling for common vaccine names
                common_vaccines = ["MMR", "DTAP", "TDAP", "HEPB", "IPV"]
                if vaccine_name in common_vaccines:
                    print(f"Found common vaccine: {vaccine_name}")
                else:
                    # Remove extra words for other vaccines
                    vaccine_name = re.sub(r'\b(VACCINATION|VACCINE|SHOT)\b', '', vaccine_name).strip()
                    print(f"Cleaned vaccine name: {vaccine_name}")
                
                # Default to dose 1 if not specified
                dose_number = 1
                if len(vaccine_match.groups()) > 1 and vaccine_match.group(2):
                    dose_str = vaccine_match.group(2)
                    if dose_str and dose_str.isdigit():
                        dose_number = int(dose_str)
                    elif dose_str:
                        word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6}
                        dose_number = word_to_num.get(dose_str.lower(), 1)
                print(f"Dose number: {dose_number}")
                
                # Verify child exists and belongs to this provider
                child_query = """
                SELECT child_id, first_name, last_name 
                FROM Children 
                WHERE ssn LIKE :ssn_pattern 
                AND provider_id = :provider_id
                """
                child = await database.fetch_one(
                    child_query, 
                    values={
                        "ssn_pattern": f"%{ssn}",
                        "provider_id": current_provider["provider_id"]
                    }
                )
                print(f"Found child: {child['first_name'] if child else None}")
                
                if child:
                    # Check if this vaccination record already exists
                    check_query = """
                    SELECT record_id 
                    FROM VaccinationTakenRecords 
                    WHERE child_id = :child_id 
                    AND vaccine_name = :vaccine_name 
                    AND dose_number = :dose_number
                    AND DATE(date_administered) = CURDATE()
                    """
                    existing_record = await database.fetch_one(
                        check_query,
                        values={
                            "child_id": child["child_id"],
                            "vaccine_name": vaccine_name,
                            "dose_number": dose_number
                        }
                    )
                    
                    if existing_record:
                        print("Found existing record for today")
                        return {
                            "message": f"⚠️ This vaccination record already exists for {child['first_name']} {child['last_name']} (today's date).",
                            "type": "vaccination_record",
                            "data": {"vaccination_recorded": False}
                        }
                    
                    try:
                        # Insert vaccination record
                        insert_query = """
                        INSERT INTO VaccinationTakenRecords (
                            child_id, vaccine_name, dose_number, 
                            date_administered, administered_by, notes
                        ) VALUES (
                            :child_id, :vaccine_name, :dose_number,
                            CURDATE(), :administered_by, :notes
                        )
                        """
                        await database.execute(
                            insert_query,
                            values={
                                "child_id": child["child_id"],
                                "vaccine_name": vaccine_name,
                                "dose_number": dose_number,
                                "administered_by": current_provider["provider_name"],
                                "notes": f"Recorded via voice assistant by {current_provider['provider_name']}"
                            }
                        )
                        print("Successfully inserted vaccination record")
                        
                        return {
                            "message": f"✅ Successfully recorded that {child['first_name']} {child['last_name']} has received {vaccine_name} dose {dose_number} today.",
                            "type": "vaccination_record",
                            "data": {"vaccination_recorded": True}
                        }
                    except Exception as e:
                        print(f"Error inserting vaccination record: {str(e)}")
                        return {
                            "message": f"❌ Error recording vaccination: {str(e)}",
                            "type": "error",
                            "data": {"vaccination_recorded": False}
                        }
                else:
                    print(f"No child found with SSN {ssn}")
                    return {
                        "message": f"❌ I could not find a child with SSN ending in {ssn} in your records. Please verify the SSN and try again.",
                        "type": "vaccination_record",
                        "data": {"vaccination_recorded": False}
                    }
            else:
                missing = []
                if not ssn_match:
                    missing.append("SSN")
                if not vaccine_match:
                    missing.append("vaccine details")
                print(f"Missing information: {missing}")
                
                return {
                    "message": f"❌ Please provide both the child's {' and '.join(missing)} to record a vaccination.",
                    "type": "vaccination_record",
                    "data": {"vaccination_recorded": False}
                }
                
        # Continue with other query types...

        # Handle MMR vaccination check
        if is_mmr_query and current_provider:
            print("\n=== Processing MMR Vaccination Check ===")
            print(f"Provider ID: {current_provider['provider_id']}")
            
            try:
                # Query to find children who haven't received MMR vaccine
                unvaccinated_query = """
                SELECT 
                    c.child_id,
                    c.first_name,
                    c.last_name,
                    c.date_of_birth,
                    TIMESTAMPDIFF(MONTH, c.date_of_birth, CURDATE()) as age_months
                FROM Children c
                WHERE c.provider_id = :provider_id
                AND NOT EXISTS (
                    SELECT 1 
                    FROM VaccinationTakenRecords vtr 
                    WHERE vtr.child_id = c.child_id 
                    AND UPPER(vtr.vaccine_name) = 'MMR'
                )
                ORDER BY c.last_name, c.first_name;
                """
                
                print("Executing query to find unvaccinated children...")
                unvaccinated_children = await database.fetch_all(
                    unvaccinated_query,
                    values={"provider_id": current_provider["provider_id"]}
                )
                print(f"Query executed successfully. Found {len(unvaccinated_children) if unvaccinated_children else 0} results")
                
                if not unvaccinated_children:
                    return {
                        "message": "✅ All your registered clients have received their MMR vaccination.",
                        "type": "vaccination_check",
                        "data": {"unvaccinated_count": 0}
                    }
                
                # Format the response
                child_list = []
                for child in unvaccinated_children:
                    age_str = f"{child['age_months']} months" if child['age_months'] < 24 else f"{child['age_months'] // 12} years"
                    child_list.append(f"• {child['first_name']} {child['last_name']} (Age: {age_str})")
                    print(f"Processing child: {child['first_name']} {child['last_name']}")
                
                response_text = "⚠️ The following clients have not received MMR vaccination:\n\n" + "\n".join(child_list)
                
                return {
                    "message": response_text,
                    "type": "vaccination_check",
                    "data": {
                        "unvaccinated_count": len(unvaccinated_children),
                        "children": [
                            {
                                "name": f"{child['first_name']} {child['last_name']}",
                                "age_months": child['age_months']
                            }
                            for child in unvaccinated_children
                        ]
                    }
                }
                
            except Exception as e:
                print(f"\nError in MMR vaccination check:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"Provider details: {current_provider}")
                
                # Check if database is connected
                try:
                    await database.fetch_one("SELECT 1")
                    print("Database connection is working")
                except Exception as db_error:
                    print(f"Database connection error: {str(db_error)}")
                
                return {
                    "message": "❌ Error: Failed to fetch clients without MMR vaccine. Please try again.",
                    "type": "error",
                    "data": {
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                }

        # Handle EHR request
        if is_ehr_request and current_provider:
            ssn_pattern = r"(?:ssn|number|#)\s*(\d{4})"
            ssn_match = re.search(ssn_pattern, cleaned_query)
            
            if ssn_match:
                ssn = ssn_match.group(1)
                try:
                    # Get child details
                    query = """
                    SELECT c.* 
                    FROM Children c 
                    WHERE c.ssn = :ssn
                    """
                    child = await database.fetch_one(query=query, values={"ssn": ssn})
                    
                    if not child:
                        return {
                            "message": f"I couldn't find a patient with SSN ending in {ssn}. Please verify the SSN and try again.",
                            "type": "error"
                        }

                    # Generate PDF
                    pdf_url = await generate_ehr_pdf(child["child_id"], current_provider["provider_id"])
                    
                    # Get the absolute path of the PDF file
                    pdf_path = os.path.join(os.getcwd(), pdf_url.lstrip('/'))
                    
                    if not os.path.exists(pdf_path):
                        return {
                            "message": "I encountered an error generating the EHR. Please try again.",
                            "type": "error"
                        }

                    # Extract filename from pdf_url and create full URL
                    filename = os.path.basename(pdf_url)
                    full_pdf_url = f"http://localhost:8000/view-pdf/{filename}"

                    # Prepare email
                    subject = f"Electronic Health Record - {child['first_name']} {child['last_name']}"
                    body = f"""
                    Dear {current_provider['provider_name']},

                    Please find attached the Electronic Health Record for {child['first_name']} {child['last_name']}.
                    You can also view the report directly in your browser.

                    Best regards,
                    Ask Health Team
                    """

                    # Prepare attachments
                    attachments = [{
                        'path': pdf_path,
                        'filename': f"EHR_{child['first_name']}_{child['last_name']}.pdf"
                    }]
                    
                    # Add email sending to background tasks
                    if background_tasks:
                        background_tasks.add_task(
                            send_email,
                            subject=subject,
                            body=body,
                            to_emails=[current_provider['email']],
                            attachments=attachments
                        )

                    return {
                        "message": f"I've generated the EHR for {child['first_name']} {child['last_name']} and it's being sent to your email.\n\nYou can view the PDF directly here: {full_pdf_url}",
                        "type": "ehr_request",
                        "data": {
                            "ehr_generated": True,
                            "pdf_url": full_pdf_url,
                            "patient_name": f"{child['first_name']} {child['last_name']}"
                        }
                    }
                except Exception as e:
                    print(f"Error generating EHR: {str(e)}")
                    traceback.print_exc()
                    return {
                        "message": "I encountered an error generating the EHR. Please try again.",
                        "type": "error"
                    }

        # Handle today's appointments
        if is_today_query and current_provider:
            today = datetime.utcnow().date()
            query = """
            SELECT 
                DATE_FORMAT(a.appointment_date, '%Y-%m-%dT%H:%i:%s.%fZ') as appointment_date,
                c.first_name, c.last_name,
                COALESCE(vs.vaccine_name, 'General Checkup') as vaccine_name,
                COALESCE(vs.dose_number, 0) as dose_number,
                a.notes, a.is_general_appointment
            FROM Appointments a
            JOIN Children c ON a.child_id = c.child_id
            LEFT JOIN VaccineSchedule vs ON a.schedule_id = vs.schedule_id
            WHERE a.provider_id = :provider_id
            AND DATE(a.appointment_date) = :appointment_date
            AND a.status = 'scheduled'
            ORDER BY a.appointment_date ASC
            """
            
            today_appointments = await database.fetch_all(
                query=query,
                values={
                    "provider_id": current_provider["provider_id"],
                    "appointment_date": today.strftime('%Y-%m-%d')
                }
            )

            if today_appointments:
                response_text.append("\n📅 Today's Appointments:")
                for appt in today_appointments:
                    appt_time = datetime.strptime(appt["appointment_date"].split('T')[1][:5], '%H:%M').strftime('%I:%M %p')
                    appointment_type = "General Checkup" if appt["is_general_appointment"] else f"{appt['vaccine_name']} (Dose {appt['dose_number']})"
                    response_text.append(f"• {appt_time} - {appt['first_name']} {appt['last_name']}: {appointment_type}")
                    if appt["notes"]:
                        response_text.append(f"  Notes: {appt['notes']}")
            else:
                response_text.append("\nNo appointments scheduled for today.")
            
            response_data["today_appointments"] = [
                {
                    "time": datetime.strptime(appt["appointment_date"].split('T')[1][:5], '%H:%M').strftime('%I:%M %p'),
                    "patient_name": f"{appt['first_name']} {appt['last_name']}",
                    "appointment_type": "General Checkup" if appt["is_general_appointment"] else appt["vaccine_name"],
                    "dose_number": appt["dose_number"] if not appt["is_general_appointment"] else None,
                    "notes": appt["notes"],
                    "full_date": appt["appointment_date"]
                }
                for appt in today_appointments
            ]

        # Handle regular appointments
        if is_appointment_query and not is_today_query:
            time_range = "week"
            time_range_text = "upcoming week"
            
            if any(word in query.lower() for word in ["month", "monthly", "30 days"]):
                time_range = "month"
                time_range_text = "next 30 days"
            elif any(word in query.lower() for word in ["two months", "2 months", "60 days"]):
                time_range = "two_months"
                time_range_text = "next 60 days"

            appointments = await get_provider_appointments(time_range, current_provider)
            if appointments:
                response_text.append(f"\n📅 Appointments for the {time_range_text}:")
                for date, appts in appointments.items():
                    response_text.append(f"\n{date}:")
                    for appt in appts:
                        response_text.append(f"• {appt['time']} - {appt['patient_name']}: {appt['appointment_type']}")
                        if appt['notes']:
                            response_text.append(f"  Notes: {appt['notes']}")
            else:
                response_text.append(f"\nNo appointments scheduled for the {time_range_text}.")
            
            response_data["appointments"] = appointments

        # Handle overdue vaccinations
        if is_vaccination_query and current_provider and not is_vaccination_record:
            overdue_vaccinations = await get_overdue_vaccinations(current_provider)
            if overdue_vaccinations:
                response_text.append("\n🚨 Overdue Vaccinations in Your Area:")
                for vacc in overdue_vaccinations[:5]:  # Show top 5 most overdue
                    response_text.append(f"• {vacc['child_name']}: {vacc['vaccine_name']} (Dose {vacc['dose_number']}) - {vacc['days_overdue']} days overdue")
                if len(overdue_vaccinations) > 5:
                    response_text.append(f"... and {len(overdue_vaccinations) - 5} more overdue vaccinations.")
            else:
                response_text.append("\nNo overdue vaccinations in your area.")
            response_data["overdue_vaccinations"] = overdue_vaccinations

        # Handle health alerts
        if is_health_alert_query and current_provider:
            health_alerts = await get_provider_local_health_alerts(current_provider)
            alerts = health_alerts.get("alerts", [])
            if alerts:
                response_text.append("\n🏥 Recent Health Alerts in Your Area:")
                for alert in alerts[:5]:  # Show top 5 most recent alerts
                    response_text.append(f"• {alert['disease']} - {alert['confidence']}% confidence - {alert['total_cases_in_area']} cases")
                if len(alerts) > 5:
                    response_text.append(f"... and {len(alerts) - 5} more alerts.")
            else:
                response_text.append("\nNo recent health alerts in your area.")
            response_data["health_alerts"] = alerts

        # If no specific query type matched
        if not any([is_appointment_query, is_vaccination_query, is_health_alert_query, is_today_query, is_ehr_request, is_mmr_query]):
            return {
                "message": "I can help you with:\n• Today's appointments\n• Upcoming appointments\n• Overdue vaccinations\n• Health alerts\n• Electronic Health Records (EHR)\n• MMR vaccination status\n\nPlease specify what you'd like to know about.",
                "type": "help"
            }

        return {
            "message": "\n".join(response_text) if response_text else "I processed your request but found no relevant information.",
            "type": "combined",
            "data": response_data
        }

    except Exception as e:
        print(f"Error processing query: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

class ProviderVoiceQuery(BaseModel):
    audio_data: Optional[str] = None  # Base64 encoded audio
    text_query: Optional[str] = None  # Text query if no audio
    return_audio: bool = False  # Default to false - only generate audio if explicitly requested

@app.post("/provider/voice-assistant")
async def provider_voice_assistant(
    query: ProviderVoiceQuery,
    current_provider=Depends(get_current_provider)
):
    """Handle voice/text queries from healthcare providers"""
    try:
        # Initialize response variables
        text_response = ""
        audio_response = None
        
        # Process the input (either audio or text)
        if query.audio_data:
            # Convert audio to text using OpenAI Whisper
            audio_bytes = base64.b64decode(query.audio_data)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio.flush()
                
                with open(temp_audio.name, "rb") as audio_file:
                    transcript = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                text_query = transcript.text
        else:
            text_query = query.text_query
            
        if not text_query:
            raise HTTPException(status_code=400, detail="No query provided")
            
        # Create provider-specific context
        provider_context = f"""
        For healthcare provider {current_provider['provider_name']} (ID: {current_provider['provider_id']}).
        Focus on:
        - Upcoming appointments
        - Patient vaccination schedules
        - Health alerts and notifications
        """
        
        # Process the query
        messages = [
            {
                "role": "system",
                "content": f"""You are a specialized healthcare assistant for providers.
                {provider_context}
                Provide clear, professional responses focused on medical information.
                Format dates as Month Day, Year.
                Use bullet points for lists.
                Keep responses concise but informative."""
            },
            {"role": "user", "content": text_query}
        ]
        
        # Generate SQL query
        sql_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
        
        generated_sql = sql_response.choices[0].message.content.strip()
        
        # Execute query and get results
        results = await database.fetch_all(query=generated_sql)
        results_str = json.dumps([dict(row) for row in results], default=str)
        
        # Generate natural language response
        response_messages = messages + [
            {"role": "assistant", "content": generated_sql},
            {"role": "system", "content": f"Query results: {results_str}. Provide a natural language response."}
        ]
        
        final_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=response_messages,
            temperature=0.7
        )
        
        text_response = final_response.choices[0].message.content.strip()
        
        # Only generate audio if explicitly requested
        audio_base64 = None
        if query.return_audio:
            try:
                audio_response = openai_client.audio.speech.create(
                    model="tts-1",
                    voice="nova",  # Professional female voice
                    input=text_response
                )
                audio_base64 = base64.b64encode(audio_response.content).decode()
            except Exception as e:
                print(f"TTS error: {str(e)}")
                # Continue without audio if TTS fails
        
        return {
            "text_response": text_response,
            "audio_response": audio_base64,
            "query_understood": text_query,
            "success": True
        }
        
    except Exception as e:
        print(f"Error in provider voice assistant: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing voice query: {str(e)}"
        )

class VaccineDueResponse(BaseModel):
    child_name: str
    child_email: str
    vaccine_name: str
    dose_number: int
    recommended_age_months: int
    due_date: str
    days_overdue: int
    reported_at: str

@app.get("/provider/overdue-vaccinations", response_model=List[VaccineDueResponse])
async def get_overdue_vaccinations(current_provider=Depends(get_current_provider)):
    """
    Get all overdue vaccinations in the provider's area (pincode).
    A vaccination is considered overdue if the due date is before the current date.
    Results are ordered by most overdue first.
    """
    try:
        if not current_provider["pincode"]:
            raise HTTPException(
                status_code=400,
                detail="Provider pincode not found. Please update your profile with a valid pincode."
            )

        query = """
        SELECT 
            CONCAT(c.first_name, ' ', c.last_name) AS child_name,
            c.email as child_email,
            vs.vaccine_name,
            vs.dose_number,
            vs.recommended_age_months,
            DATE_FORMAT(DATE_ADD(c.date_of_birth, INTERVAL vs.recommended_age_months MONTH), '%Y-%m-%dT%H:%i:%s.%fZ') AS due_date,
            DATEDIFF(CURDATE(), DATE_ADD(c.date_of_birth, INTERVAL vs.recommended_age_months MONTH)) as days_overdue,
            DATE_FORMAT(NOW(), '%Y-%m-%dT%H:%i:%s.%fZ') as reported_at
        FROM Children c
        CROSS JOIN VaccineSchedule vs
        LEFT JOIN VaccinationTakenRecords vtr ON (
            c.child_id = vtr.child_id 
            AND vs.vaccine_name = vtr.vaccine_name 
            AND vs.dose_number = vtr.dose_number
        )
        WHERE 
            c.pincode = :pincode
            AND vtr.record_id IS NULL
            AND DATE_ADD(c.date_of_birth, INTERVAL vs.recommended_age_months MONTH) < CURDATE()
        ORDER BY days_overdue DESC
        """

        results = await database.fetch_all(
            query=query,
            values={"pincode": current_provider["pincode"]}
        )

        return [
            {
                "child_name": row["child_name"],
                "child_email": row["child_email"],
                "vaccine_name": row["vaccine_name"],
                "dose_number": row["dose_number"],
                "recommended_age_months": row["recommended_age_months"],
                "due_date": row["due_date"],
                "days_overdue": row["days_overdue"],
                "reported_at": row["reported_at"]
            }
            for row in results
        ]

    except Exception as e:
        print(f"Error fetching overdue vaccinations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching overdue vaccinations: {str(e)}"
        )

# Update Pydantic models
class GeneralAppointmentCreate(BaseModel):
    provider_id: int
    appointment_date: str  # Should be datetime or use proper date validation
    notes: Optional[str] = None
    child_id: Optional[int] = None  # Will be set from authentication
    healthalert_id: Optional[int] = None  # New field for health alert reference
    
    @validator('appointment_date')
    def validate_appointment_date(cls, v):
        try:
            # Parse the date string and validate format
            parsed_date = datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
            # Ensure appointment is not in the past (with some buffer)
            if parsed_date.date() < date.today():
                raise ValueError("Appointment date cannot be in the past")
            return v
        except ValueError as e:
            if "does not match format" in str(e):
                raise ValueError("Date must be in format YYYY-MM-DD HH:MM:SS")
            raise e

class AppointmentResponse(BaseModel):
    appointment_id: int
    child_id: int
    provider_id: int
    provider_name: str
    appointment_date: datetime
    notes: Optional[str]
    status: str = "scheduled"
    created_at: datetime
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

@app.post("/general-appointments", status_code=201, response_model=AppointmentResponse)
async def create_general_appointment(
    appointment: GeneralAppointmentCreate,
    current_user=Depends(get_current_user)
):
    try:
        # Convert appointment time to UTC before storing
        local_date = datetime.strptime(appointment.appointment_date, "%Y-%m-%d %H:%M:%S")
        utc_date = local_date.astimezone(timezone.utc)
        
        print(f"Converting local time {local_date} to UTC {utc_date}")
        
        child_id = current_user["child_id"]
        
        # Check if health alert exists and belongs to the user
        if appointment.healthalert_id:
            alert_query = """
            SELECT alert_id FROM HealthAlerts 
            WHERE alert_id = :alert_id AND child_id = :child_id
            """
            alert = await database.fetch_one(
                query=alert_query,
                values={"alert_id": appointment.healthalert_id, "child_id": child_id}
            )
            if not alert:
                raise HTTPException(
                    status_code=404,
                    detail="Health alert not found or does not belong to the user"
                )
        
        # Rest of the validation code...
        provider_query = """
        SELECT provider_id, provider_name, email, phone, address
        FROM HealthcareProviders
        WHERE provider_id = :provider_id
        """
        provider = await database.fetch_one(
            query=provider_query,
            values={"provider_id": appointment.provider_id}
        )
        
        if not provider:
            raise HTTPException(status_code=404, detail="Healthcare provider not found")

        # Check for conflicts in UTC time
        conflict_query = """
        SELECT appointment_id 
        FROM Appointments 
        WHERE provider_id = :provider_id 
        AND appointment_date = :appointment_date
        AND status != 'cancelled'
        """
        existing_appointment = await database.fetch_one(
            query=conflict_query,
            values={
                "provider_id": appointment.provider_id,
                "appointment_date": utc_date
            }
        )
        
        if existing_appointment:
            raise HTTPException(status_code=409, detail="This time slot is already booked")

        async with database.transaction():
            # Insert the appointment in UTC
            insert_query = """
            INSERT INTO Appointments (
                child_id,
                provider_id,
                appointment_date,
                notes,
                is_general_appointment,
                status,
                created_at,
                healthalert_id
            ) VALUES (
                :child_id,
                :provider_id,
                :appointment_date,
                :notes,
                TRUE,
                'scheduled',
                UTC_TIMESTAMP(),
                :healthalert_id
            )
            """
            
            await database.execute(
                query=insert_query,
                values={
                    "child_id": child_id,
                    "provider_id": appointment.provider_id,
                    "appointment_date": utc_date,
                    "notes": appointment.notes,
                    "healthalert_id": appointment.healthalert_id
                }
            )

            # Update health alert status if healthalert_id is provided
            if appointment.healthalert_id:
                update_alert_query = """
                UPDATE HealthAlerts 
                SET status = 'booked'
                WHERE alert_id = :alert_id
                """
                await database.execute(
                    query=update_alert_query,
                    values={"alert_id": appointment.healthalert_id}
                )

            get_last_id_query = "SELECT LAST_INSERT_ID() as appointment_id"
            last_id_result = await database.fetch_one(query=get_last_id_query)
            appointment_id = last_id_result["appointment_id"]

            # Get the appointment details and convert back to local time
            get_appointment_query = """
            SELECT 
                a.appointment_id,
                a.child_id,
                a.provider_id,
                hp.provider_name,
                CONVERT_TZ(a.appointment_date, '+00:00', @@session.time_zone) as appointment_date,
                a.notes,
                a.status,
                CONVERT_TZ(a.created_at, '+00:00', @@session.time_zone) as created_at,
                a.healthalert_id
            FROM Appointments a
            JOIN HealthcareProviders hp ON a.provider_id = hp.provider_id
            WHERE a.appointment_id = :appointment_id
            """
            
            result = await database.fetch_one(
                query=get_appointment_query,
                values={"appointment_id": appointment_id}
            )

            if not result:
                raise HTTPException(status_code=500, detail="Failed to retrieve created appointment")

            appointment_dict = dict(result)
            return AppointmentResponse(**appointment_dict)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating appointment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create appointment: {str(e)}")

@app.get("/provider/appointments")
async def get_provider_appointments(
    time_range: Literal["today", "week", "month", "two_months"] = Query(..., description="Time range for appointments"),
    current_provider=Depends(get_current_provider)
):
    try:
        print(f"\n=== Fetching Provider Appointments ===")
        print(f"Provider ID: {current_provider['provider_id']}")
        print(f"Time Range: {time_range}")
        
        # Get current time in UTC
        now_utc = datetime.now(timezone.utc)
        
        # For filtering, use current time (not start of day) to exclude past appointments
        if time_range == "today":
            start_date = now_utc  # Use current time, not start of day
            end_date = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif time_range == "week":
            start_date = now_utc
            end_date = now_utc + timedelta(days=7)
        elif time_range == "month":
            start_date = now_utc
            end_date = now_utc + timedelta(days=30)
        else:  # two_months
            start_date = now_utc
            end_date = now_utc + timedelta(days=60)
            
        print(f"UTC time range: {start_date} to {end_date}")

        # Query appointments - use UTC time for comparison
        query = """
        SELECT DISTINCT
            CONVERT_TZ(a.appointment_date, '+00:00', @@session.time_zone) as local_appointment_date,
            a.appointment_date as utc_appointment_date,
            c.first_name,
            c.last_name,
            COALESCE(vs.vaccine_name, 'General Checkup') as vaccine_name,
            COALESCE(vs.dose_number, 0) as dose_number,
            a.notes,
            a.is_general_appointment
        FROM Appointments a
        JOIN Children c ON a.child_id = c.child_id
        LEFT JOIN VaccineSchedule vs ON a.schedule_id = vs.schedule_id
        WHERE a.provider_id = :provider_id
            AND a.appointment_date >= :start_date  # Compare with UTC start_date (current time)
            AND a.appointment_date < :end_date
            AND a.status = 'scheduled'
        ORDER BY a.appointment_date ASC
        """

        appointments = await database.fetch_all(
            query=query,
            values={
                "provider_id": current_provider["provider_id"],
                "start_date": start_date,
                "end_date": end_date
            }
        )
        
        print(f"Found {len(appointments)} appointments")

        formatted_appointments = []
        for appointment in appointments:
            local_time = appointment["local_appointment_date"]
            utc_time = appointment["utc_appointment_date"]
            
            print(f"UTC time: {utc_time}, Local time: {local_time}")
            
            formatted_appointments.append({
                "appointment_date": local_time.isoformat(),
                "utc_appointment_date": utc_time.isoformat(),
                "patient_name": f"{appointment['first_name']} {appointment['last_name']}".strip(),
                "vaccine_name": "General Checkup" if appointment["is_general_appointment"] else appointment["vaccine_name"],
                "dose_number": appointment["dose_number"],
                "notes": appointment["notes"],
                "formatted_time": local_time.strftime('%I:%M %p')
            })
        
        return formatted_appointments

    except Exception as e:
        print(f"Error fetching provider appointments: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error fetching appointments: {str(e)}")

class ConfirmDiseaseRequest(BaseModel):
    child_id: int
    disease: str
    diagnosed_date: str  # Format: YYYY-MM-DD
    alert_id: int

    @validator('diagnosed_date')
    def validate_diagnosed_date(cls, v):
        try:
            # Parse the date string and validate format
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("diagnosed_date must be in YYYY-MM-DD format")

    @validator('disease')
    def validate_disease(cls, v):
        if not v.strip():
            raise ValueError("disease cannot be empty")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "child_id": 1,
                "disease": "Measles",
                "diagnosed_date": "2024-03-20",
                "alert_id": 1
            }
        }

@app.post("/provider/confirm-disease")
async def confirm_disease(
    request: ConfirmDiseaseRequest,
    current_provider=Depends(get_current_provider)
):
    """Add a confirmed disease diagnosis to the database"""
    try:
        # First verify that the alert exists and belongs to the child
        verify_query = """
        SELECT alert_id 
        FROM HealthAlerts 
        WHERE alert_id = :alert_id AND child_id = :child_id
        """
        
        alert = await database.fetch_one(
            query=verify_query,
            values={
                "alert_id": request.alert_id,
                "child_id": request.child_id
            }
        )
        
        if not alert:
            raise HTTPException(
                status_code=404,
                detail="Health alert not found or does not belong to this child"
            )

        # Insert the confirmed diagnosis
        query = """
        INSERT INTO diseasesDiagnosedChildren 
        (child_id, disease, diagnosed_date, alert_id)
        VALUES (:child_id, :disease, :diagnosed_date, :alert_id)
        """
        
        await database.execute(
            query=query,
            values={
                "child_id": request.child_id,
                "disease": request.disease,
                "diagnosed_date": request.diagnosed_date,
                "alert_id": request.alert_id
            }
        )
        
        return {"message": "Disease confirmed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error confirming disease: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error confirming disease: {str(e)}"
        )

@app.get("/disease-outbreak", response_model=List[dict])
async def disease_outbreak(current_user=Depends(get_current_user)):
    today = datetime.utcnow().date()
    thirty_days_ago = today - timedelta(days=30)

    query = """
    SELECT disease, COUNT(*) AS disease_count
    FROM diseasesDiagnosedChildren
    WHERE diagnosed_date BETWEEN :thirty_days_ago AND :today
    GROUP BY disease
    HAVING disease_count >= 5
    """

    try:
        results = await database.fetch_all(query, values={"thirty_days_ago": thirty_days_ago, "today": today})
        if not results:
            return []
        diseases = [{"disease": row["disease"], "count": row["disease_count"]} for row in results]
        return diseases
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

class ExecuteQueryRequest(BaseModel):
    query: str

@app.post("/execute-query")
async def execute_query(
    request: ExecuteQueryRequest,
    current_provider=Depends(get_current_provider)
):
    """Execute a read-only SQL query"""
    try:
        # Ensure the query is read-only
        if not request.query.lower().strip().startswith('select'):
            raise HTTPException(
                status_code=400,
                detail="Only SELECT queries are allowed"
            )
            
        # Execute the query
        result = await database.fetch_all(query=request.query)
        
        # Convert result to list of dicts
        return [dict(row) for row in result]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing query: {str(e)}"
        )

class VaccinationRecordCreate(BaseModel):
    child_id: int
    vaccine_name: str
    dose_number: int
    administered_by: str
    notes: Optional[str] = None

@app.post("/provider/record-vaccination")
async def record_vaccination(
    record: VaccinationRecordCreate,
    current_provider=Depends(get_current_provider)
):
    """Record a new vaccination for a child"""
    try:
        # Insert the vaccination record
        query = """
        INSERT INTO VaccinationTakenRecords 
        (child_id, vaccine_name, dose_number, date_administered, administered_by, reminder_sent, notes)
        VALUES (:child_id, :vaccine_name, :dose_number, CURDATE(), :administered_by, 0, :notes)
        """
        await database.execute(
            query=query,
            values={
                "child_id": record.child_id,
                "vaccine_name": record.vaccine_name,
                "dose_number": record.dose_number,
                "administered_by": record.administered_by,
                "notes": record.notes
            }
        )

        return {"message": "Vaccination record created successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create vaccination record: {str(e)}"
        )

# Add helper functions for PDF processing
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
    """Parse extracted text to structured data"""
    data = {}
    
    # Extract First Name
    first_name_match = re.search(r"First Name:\s*([^\n]+)", text)
    if first_name_match:
        data['first_name'] = first_name_match.group(1).strip()
    else:
        raise ValueError("First Name is required")

    # Extract Last Name from Name field
    name_match = re.search(r"Last Name:\s*([^\n]+)", text)
    if name_match:
        data['last_name'] = name_match.group(1).strip()
    else:
        raise ValueError("Last Name is required")

    # Extract DOB
    dob_match = re.search(r"Date of Birth:\s*([^\n]+)", text)
    if dob_match:
        dob_str = dob_match.group(1).strip()
        try:
            dob_date = datetime.strptime(dob_str, "%B %d, %Y")
            data['date_of_birth'] = dob_date.strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid Date of Birth format. Expected format: Month DD, YYYY")
    else:
        raise ValueError("Date of Birth is required")

    # Extract Gender
    gender_match = re.search(r"Gender:\s*([^\n]+)", text)
    if gender_match:
        data['gender'] = gender_match.group(1).strip()
    else:
        raise ValueError("Gender is required")

    # Extract Parent ID
    parent_id_match = re.search(r"Parent ID:\s*(\d+)", text)
    if parent_id_match:
        try:
            data['parent_id'] = int(parent_id_match.group(1).strip())
        except ValueError:
            data['parent_id'] = None
    else:
        data['parent_id'] = None

    # Extract Blood Type
    blood_type_match = re.search(r"Blood Type:\s*([^\n]+)", text)
    if blood_type_match:
        data['blood_type'] = blood_type_match.group(1).strip()
    else:
        # Try alternate format where O- might be split across lines
        blood_type_line = re.search(r"([ABO][-+])[^\n]*", text)
        if blood_type_line:
            data['blood_type'] = blood_type_line.group(1).strip()
        else:
            raise ValueError("Blood Type is required")

    # Extract Pincode
    pincode_match = re.search(r"Pincode:\s*([^\n]+)", text)
    if pincode_match:
        data['pincode'] = pincode_match.group(1).strip()
    else:
        raise ValueError("Pincode is required")

    # Extract Email
    email_match = re.search(r"Email:\s*([^\n]+)", text)
    if email_match:
        data['email'] = email_match.group(1).strip()
    else:
        raise ValueError("Email is required")

    # Extract SSN
    ssn_match = re.search(r"SSN:\s*([^\n]+)", text)
    if ssn_match:
        data['ssn'] = ssn_match.group(1).strip()
    else:
        raise ValueError("SSN is required")

    # Extract Vaccination Information
    if "Vaccination History:" in text:
        # Extract Vaccine Name
        vaccine_name_match = re.search(r"Vaccine Name:\s*([^\n]+)", text)
        if vaccine_name_match:
            data['vaccination'] = vaccine_name_match.group(1).strip()
            
            # Extract Dose Number
            dose_match = re.search(r"Dose Number:\s*(\d+)", text)
            if dose_match:
                data['dose_number'] = int(dose_match.group(1))
            
            # Extract Date Administered
            date_admin_match = re.search(r"Date Administered:\s*([^\n]+)", text)
            if date_admin_match:
                date_str = date_admin_match.group(1).strip()
                try:
                    # Handle date format "January 1, 2023"
                    admin_date = datetime.strptime(date_str, "%B %d, %Y")
                    data['vaccination_date'] = admin_date.strftime("%Y-%m-%d")
                except ValueError:
                    data['vaccination_date'] = datetime.now().strftime("%Y-%m-%d")
            
            # Extract Administered By
            admin_by_match = re.search(r"Administered By:\s*([^\n]+)", text)
            if admin_by_match:
                data['administered_by'] = admin_by_match.group(1).strip()
            
            # Extract Reminder Sent
            reminder_match = re.search(r"Reminder Sent:\s*([^\n]+)", text)
            if reminder_match:
                data['reminder_sent'] = 1 if reminder_match.group(1).strip().lower() == 'yes' else 0
            
            # Extract Notes
            notes_match = re.search(r"Notes:\s*([^\n]+)", text)
            if notes_match:
                data['notes'] = notes_match.group(1).strip()

    print("Parsed data:", data)  # Debug print
    return data

def generate_username_password(first_name, last_name):
    username = first_name.lower() + '.' + last_name.lower() + str(random.randint(1000, 9999))
    password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return username, password

def hash_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password

async def send_credentials_to_user(email, username, password):
    try:
        subject = "Welcome to Our Healthcare System"
        body = f"""
        Hello,

        This email is to inform you that you have been added as a client to our healthcare system.

        Our healthcare providers will maintain your records and keep track of your vaccinations and health status.

        Best regards,
        Your Health System
        """

        # Use the existing send_email function
        send_email(subject, body, [email])
        print(f"Welcome email sent successfully to {email}")
    except Exception as e:
        print(f"Failed to send email: {e}")
        # Don't raise the exception, just log it
        pass

# Add new endpoint for PDF upload
@app.post("/provider/upload-client-pdf")
async def upload_client_pdf(
    pdf_file: UploadFile = File(...),
    current_provider: dict = Depends(get_current_provider)
):
    """Upload a PDF file containing client information"""
    try:
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, pdf_file.filename)
        
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(pdf_file.file, buffer)
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        print("Extracted text:", text)  # Debug print
        
        # Parse text to data
        data = parse_text_to_data(text)
        print("Parsed data:", data)  # Debug print
        
        # Insert into Children table
        query = """
        INSERT INTO Children (
            first_name, last_name, date_of_birth, gender,
            blood_type, pincode, email, ssn, provider_id,
            parent_id
        ) VALUES (
            :first_name, :last_name, :date_of_birth, :gender,
            :blood_type, :pincode, :email, :ssn, :provider_id,
            :parent_id
        )
        """
        
        values = {
            "first_name": data["first_name"],
            "last_name": data["last_name"],
            "date_of_birth": data["date_of_birth"],
            "gender": data["gender"],
            "blood_type": data["blood_type"],
            "pincode": data["pincode"],
            "email": data["email"],
            "ssn": data["ssn"],
            "provider_id": current_provider["provider_id"],
            "parent_id": data.get("parent_id")
        }
        
        result = await database.execute(query, values)
        child_id = result  # Get the auto-generated child_id
        
        # If vaccination information is present, record it
        if "vaccination" in data:
            print("Inserting vaccination record:", data)  # Debug print
            vaccination_query = """
            INSERT INTO VaccinationTakenRecords (
                child_id, vaccine_name, dose_number,
                date_administered, administered_by,
                reminder_sent, notes
            ) VALUES (
                :child_id, :vaccine_name, :dose_number,
                :date_administered, :administered_by,
                :reminder_sent, :notes
            )
            """
            
            vaccination_values = {
                "child_id": child_id,
                "vaccine_name": data["vaccination"],
                "dose_number": data.get("dose_number", 1),
                "date_administered": data.get("vaccination_date", datetime.now().date()),
                "administered_by": data.get("administered_by", current_provider["provider_name"]),
                "reminder_sent": data.get("reminder_sent", 0),
                "notes": data.get("notes", "Added via PDF upload")
            }
            
            await database.execute(vaccination_query, vaccination_values)
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return {
            "status": "success",
            "message": f"Successfully added client {data['first_name']} {data['last_name']} with vaccination record",
            "child_id": child_id
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing PDF: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )

# Add middleware to ensure database connection
@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    try:
        # Ensure database is connected
        if not database.is_connected:
            await database.connect()
            print("Database reconnected in middleware")
        response = await call_next(request)
        return response
    except Exception as e:
        print(f"Error in database middleware: {str(e)}")
        raise
    finally:
        if database.is_connected:
            print("Database connection maintained")

async def generate_ehr_pdf(child_id: int, provider_id: int) -> str:
    try:
        # Get child information
        query = """
        SELECT c.*, p.provider_name, p.specialty
        FROM Children c
        LEFT JOIN HealthcareProviders p ON p.provider_id = :provider_id
        WHERE c.child_id = :child_id
        """
        result = await database.fetch_one(query, {"child_id": child_id, "provider_id": provider_id})
        
        if not result:
            raise HTTPException(status_code=404, detail="Child not found")
            
        # Get vaccination records
        vax_query = """
        SELECT vt.*, vs.recommended_age_months, vs.description
        FROM VaccinationTakenRecords vt
        LEFT JOIN VaccineSchedule vs ON vs.vaccine_name = vt.vaccine_name 
            AND vs.dose_number = vt.dose_number
        WHERE vt.child_id = :child_id
        ORDER BY vt.date_administered DESC
        """
        vax_records = await database.fetch_all(vax_query, {"child_id": child_id})
        
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ehr_{child_id}_{timestamp}.pdf"
        filepath = f"static/pdfs/{filename}"
        
        # Generate PDF
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        elements.append(Paragraph("Electronic Health Record", title_style))
        
        # Add child information
        elements.append(Paragraph("Patient Information", styles["Heading2"]))
        child_info = [
            ["Name:", f"{result.first_name} {result.last_name}"],
            ["Date of Birth:", result.date_of_birth.strftime("%B %d, %Y")],
            ["Gender:", result.gender],
            ["Blood Type:", result.blood_type],
            ["Email:", result.email]
        ]
        
        t = Table(child_info, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))
        
        # Add healthcare provider information
        elements.append(Paragraph("Healthcare Provider Information", styles["Heading2"]))
        provider_info = [
            ["Provider Name:", result.provider_name],
            ["Specialty:", result.specialty or "Not specified"]
        ]
        
        t = Table(provider_info, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))
        
        # Add vaccination records
        elements.append(Paragraph("Vaccination History", styles["Heading2"]))
        if vax_records:
            vax_data = [["Vaccine", "Dose", "Date", "Administered By", "Notes"]]
            for record in vax_records:
                vax_data.append([
                    record.vaccine_name,
                    str(record.dose_number),
                    record.date_administered.strftime("%B %d, %Y"),
                    record.administered_by,
                    record.notes or ""
                ])
            
            t = Table(vax_data, colWidths=[1.5*inch, 0.7*inch, 1.5*inch, 1.5*inch, 2*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(t)
        else:
            elements.append(Paragraph("No vaccination records found.", styles["Normal"]))
        
        # Build PDF
        doc.build(elements)
        
        # Return the URL path for the generated PDF
        return f"/static/pdfs/{filename}"
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to generate EHR PDF")

@app.post("/provider/generate-ehr")
async def generate_and_email_ehr(
    background_tasks: BackgroundTasks,
    current_provider: dict = Depends(get_current_provider),
    ssn: str = Query(..., description="Child's SSN")
):
    """Generate and email EHR for a child"""
    try:
        # Get child details
        query = """
        SELECT c.* 
        FROM Children c 
        WHERE c.ssn = :ssn
        """
        child = await database.fetch_one(query=query, values={"ssn": ssn})
        
        if not child:
            raise HTTPException(
                status_code=404,
                detail="Child not found"
            )

        # Generate PDF
        pdf_url = await generate_ehr_pdf(child["child_id"], current_provider["provider_id"])
        
        # Get the absolute path of the PDF file
        pdf_path = os.path.join(os.getcwd(), pdf_url.lstrip('/'))
        
        if not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=500,
                detail="Generated PDF file not found"
            )

        # Prepare email
        subject = f"Electronic Health Record - {child['first_name']} {child['last_name']}"
        body = f"""
        Dear {current_provider['provider_name']},

        Please find attached the Electronic Health Record for {child['first_name']} {child['last_name']}.
        You can also view the report directly in your browser.

        Best regards,
        Ask Health Team
        """

        # Send email with PDF attachment
        attachments = [{
            'path': pdf_path,
            'filename': f"EHR_{child['first_name']}_{child['last_name']}.pdf"
        }]
        
        # Add email sending to background tasks
        background_tasks.add_task(
            send_email,
            subject=subject,
            body=body,
            to_emails=[current_provider['email']],
            attachments=attachments
        )

        # Extract filename from pdf_url
        filename = os.path.basename(pdf_url)
        
        return {
            "status": "success",
            "message": f"EHR has been generated and email is being sent to {current_provider['email']}",
            "data": {
                "pdf_url": f"http://localhost:8000/view-pdf/{filename}",
                "patient_name": f"{child['first_name']} {child['last_name']}"
            }
        }

    except Exception as e:
        print(f"Error generating and emailing EHR: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error generating and emailing EHR: {str(e)}"
        )

# Add endpoint to check if PDF exists
@app.get("/check-pdf/{filename}")
async def check_pdf(filename: str):
    """Check if a PDF file exists"""
    file_path = f"static/pdfs/{filename}"
    return {"exists": os.path.exists(file_path)}

# Add endpoint to serve PDF directly
@app.get("/view-pdf/{filename}")
async def view_pdf(
    filename: str,
    current_provider: dict = Depends(get_current_provider)
):
    """Serve PDF file for viewing"""
    file_path = f"static/pdfs/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    
    headers = {
        'Content-Type': 'application/pdf',
        'Content-Disposition': f'inline; filename="{filename}"',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    }
    
    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename,
        headers=headers
    )