from fastapi import FastAPI, HTTPException, Depends, status, Query, BackgroundTasks
from pydantic import BaseModel, validator
from databases import Database
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Literal

import re
import smtplib
from email.message import EmailMessage
import pandas as pd
from dateutil.relativedelta import relativedelta
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain.prompts.prompt import PromptTemplate
from sqlalchemy import create_engine, text

# Database configuration
DATABASE_URL = "mysql+aiomysql://root:@localhost:3306/childhealth"
SYNC_DATABASE_URL = "mysql+pymysql://root:@localhost:3306/childhealth"

# Authentication configuration
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Initialize database
database = Database(DATABASE_URL)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
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

# Initialize SQLAlchemy engine
engine = create_engine(SYNC_DATABASE_URL)
db = SQLDatabase(engine)

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
db_chain = SQLDatabaseChain(
    llm=llm,
    database=db,
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
    "VaccinationTakenRecords": ["record_id", "child_id", "vaccine_name", "dose_number", "date_administered", "administered_by", "reminder_sent", "notes"],
    "VaccineSchedule": ["schedule_id", "vaccine_name", "dose_number", "recommended_age_months", "mandatory", "description", "benefits"],
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
    parent_id: int
    blood_type: str
    pincode: str
    username: str
    password: str

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

class SymptomAnalysis(BaseModel):
    transcript: str
    userId: str
    callId: str

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
    messages: List[ChatMessage]

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

# Utility Functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

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

def months_between(start_date, end_date):
    rd = relativedelta(end_date, start_date)
    return rd.years * 12 + rd.months

def parse_iso_datetime(iso_str: str) -> str:
    if iso_str.endswith('Z'):
        iso_str = iso_str[:-1]
    dt = datetime.fromisoformat(iso_str)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def send_email(subject: str, body: str, to_emails: List[str]):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(to_emails)
    msg.set_content(body)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)

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
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.post("/signup", status_code=201)
async def signup(child: ChildCreate):
    insert_child_query = """
    INSERT INTO Children (first_name, last_name, date_of_birth, gender, parent_id, blood_type, pincode)
    VALUES (:first_name, :last_name, :date_of_birth, :gender, :parent_id, :blood_type, :pincode)
    """
    child_values = child.dict()
    child_values.pop("username")
    child_values.pop("password")

    async with database.transaction():
        child_id = await database.execute(query=insert_child_query, values=child_values)
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

    return {"message": "Child user created successfully", "child_id": child_id}

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
    WHERE child_id = :child_id AND summary != 'No summary available.'
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
    age_in_months = months_between(dob.date(), today)
    upcoming_age_start = age_in_months
    upcoming_age_end = age_in_months + 1

    query_schedule = """
    SELECT schedule_id, vaccine_name, dose_number, recommended_age_months, mandatory
    FROM VaccineSchedule
    WHERE recommended_age_months BETWEEN :start_age AND :end_age
    ORDER BY recommended_age_months
    """
    schedules = await database.fetch_all(query=query_schedule, values={"start_age": upcoming_age_start, "end_age": upcoming_age_end})

    query_taken = """
    SELECT vaccine_name, dose_number FROM VaccinationTakenRecords WHERE child_id = :child_id
    """
    taken_records = await database.fetch_all(query=query_taken, values={"child_id": current_user["child_id"]})
    taken_set = {(r["vaccine_name"], r["dose_number"]) for r in taken_records}

    query_appointments = """
    SELECT schedule_id FROM Appointments WHERE child_id = :child_id
    """
    appointment_rows = await database.fetch_all(query=query_appointments, values={"child_id": current_user["child_id"]})
    appointment_ids = {r["schedule_id"] for r in appointment_rows}

    result = []
    for s in schedules:
        if (s["vaccine_name"], s["dose_number"]) in taken_set:
            continue
        due_date = (dob + relativedelta(months=+s["recommended_age_months"])).date().isoformat()
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

@app.post("/chat-assistant")
async def chat_assistant(request: QueryRequest, current_user=Depends(get_current_user)):
    try:
        # Get the last message from the user
        user_query = request.messages[-1].content
        
        # Add context about the current user
        contextualized_query = f"For child_id {current_user['child_id']}: {user_query}"
        
        result = db_chain(contextualized_query)
        
        # Extract components from the result
        sql_query = result["intermediate_steps"][0]
        query_result = result["intermediate_steps"][1]
        final_answer = result["result"]

        # Format the response to match frontend expectations
        response = {
            "response": final_answer,
            "generated_sql": sql_query,
            "query_result": query_result,
            "message_history": request.messages + [{"role": "assistant", "content": final_answer}]
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

@app.post("/appointments", status_code=201)
async def create_appointment(
    appointment: AppointmentCreate,
    current_user=Depends(get_current_user)
):
    insert_query = """
    INSERT INTO Appointments (child_id, appointment_date, provider_id, schedule_id)
    VALUES (:child_id, :appointment_date, :provider_id, :schedule_id)
    """
    await database.execute(query=insert_query, values=appointment.dict())
    return {"message": "Appointment scheduled successfully"}

@app.get("/appointments-for-child")
async def get_appointments_for_child(current_user=Depends(get_current_user)):
    query = """
    SELECT schedule_id FROM Appointments
    WHERE child_id = :child_id
    """
    rows = await database.fetch_all(query, values={"child_id": current_user["child_id"]})
    return [row["schedule_id"] for row in rows]

@app.post("/analyze-symptoms")
async def analyze_symptoms(data: SymptomAnalysis, current_user=Depends(get_current_user)):
    try:
        messages = [
            {"role": "system", "content": """
            You are a medical symptoms analyzer. Extract medical symptoms from the conversation. 
            For each symptom include:
            - Exact symptom name (be specific and consistent)
            - Severity (mild/moderate/severe if mentioned)
            - Duration (how long they've had it)
            - Context (relevant details about when it occurs, what makes it better/worse)
            
            Return ONLY valid JSON in this format:
            {
                "symptoms": [
                    {
                        "name": "symptom name",
                        "severity": "mild/moderate/severe",
                        "duration": "duration mentioned",
                        "context": "relevant context"
                    }
                ]
            }
            """},
            {"role": "user", "content": data.transcript}
        ]
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            response_format={ "type": "json_object" }
        )
        
        extracted_data = json.loads(response.choices[0].message.content)
        
        for symptom in extracted_data['symptoms']:
            query = """
            INSERT INTO ExtractedSymptoms 
            (child_id, call_id, symptom_name, severity, duration, first_reported, context)
            VALUES (:child_id, :call_id, :symptom_name, :severity, :duration, NOW(), :context)
            """
            await database.execute(query=query, values={
                "child_id": current_user["child_id"],
                "call_id": data.callId,
                "symptom_name": symptom["name"].lower(),
                "severity": symptom.get("severity"),
                "duration": symptom.get("duration"),
                "context": symptom.get("context")
            })
            
        return {"status": "success", "symptoms": extracted_data["symptoms"]}
        
    except Exception as e:
        print(f"Error in analyze_symptoms: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check-health-alerts")
async def check_health_alerts(current_user=Depends(get_current_user)):
    try:
        # Get recent symptoms
        query = """
        SELECT symptom_name, severity, duration, context, first_reported
        FROM ExtractedSymptoms
        WHERE child_id = :child_id
        AND first_reported >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        ORDER BY first_reported DESC
        """
        recent_symptoms = await database.fetch_all(
            query=query, 
            values={"child_id": current_user["child_id"]}
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
    SELECT disease_name, confidence_score, matching_symptoms, created_at
    FROM HealthAlerts
    WHERE child_id = :child_id
    ORDER BY created_at DESC
    LIMIT 10
    """
    alerts = await database.fetch_all(
        query=query,
        values={"child_id": current_user["child_id"]}
    )
    
    return {
        "alerts": [
            {
                "disease": alert["disease_name"],
                "confidence": alert["confidence_score"],
                "matching_symptoms": json.loads(alert["matching_symptoms"]),
                "created_at": alert["created_at"].isoformat()
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