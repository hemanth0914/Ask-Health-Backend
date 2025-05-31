from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from databases import Database
import re
from fastapi import Query, HTTPException

from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from fastapi import Depends
from pydantic import BaseModel, validator
from fastapi import BackgroundTasks
import smtplib
from email.message import EmailMessage
from typing import List, Literal
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import date

from openai import OpenAI
import os
from dotenv import load_dotenv

DATABASE_URL = "mysql+aiomysql://root:@localhost:3306/childhealth"

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

database = Database(DATABASE_URL)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow only specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    description: str  # Add vaccine description here
    benefits: str
class ImmunizationSummaryResponse(BaseModel):
    child_id: int
    first_name: str
    last_name: str
    summary: str

# Pydantic models
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
    startedAt: str  # or datetime if you want, adjust accordingly
    endedAt: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str




# Utility functions
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

# Routes
@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

class UpcomingVaccine(BaseModel):
    child_id: int
    schedule_id: int
    first_name: str
    last_name: str
    vaccine_name: str
    dose_number: int
    due_date: str
    appointment_booked: bool  # ✅ Add this line


@app.post("/signup", status_code=201)
async def signup(child: ChildCreate):
    # Insert child data
    insert_child_query = """
    INSERT INTO Children (first_name, last_name, date_of_birth, gender, parent_id, blood_type, pincode)
    VALUES (:first_name, :last_name, :date_of_birth, :gender, :parent_id, :blood_type, :pincode)
    """
    child_values = child.dict()
    child_values.pop("username")
    child_values.pop("password")

    async with database.transaction():
        child_id = await database.execute(query=insert_child_query, values=child_values)

        # Hash password
        hashed_pw = get_password_hash(child.password)

        # Insert child login
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

@app.get("/summaries", response_model=List[Summary])
async def fetch_summaries(current_user=Depends(get_current_user)):
    query = """
    SELECT call_id, summary, startedAt, endedAt
    FROM Summaries
    WHERE child_id = :child_id AND summary != 'No summary available.'
    ORDER BY startedAt DESC
    """
    results = await database.fetch_all(query=query, values={"child_id": current_user["child_id"]})
    print(results)
    # Convert datetime to string if needed
    summaries = []
    for row in results:
        summaries.append({
            "call_id": row["call_id"],
            "summary": row["summary"],
            "startedAt": row["startedAt"].isoformat() if row["startedAt"] else None,
            "endedAt": row["endedAt"].isoformat() if row["endedAt"] else None,
        })

    return summaries

from datetime import datetime

def parse_iso_datetime(iso_str: str) -> str:
    # Remove 'Z' suffix and parse as UTC
    if iso_str.endswith('Z'):
        iso_str = iso_str[:-1]
    # Parse ISO format string without timezone info
    dt = datetime.fromisoformat(iso_str)
    # Format to MySQL datetime string format
    return dt.strftime("%Y-%m-%d %H:%M:%S")

@app.post("/store-summary", response_model=Summary)
async def store_summary(
    summary_data: Summary,
    current_user=Depends(get_current_user),
):
    started_at = parse_iso_datetime(summary_data.startedAt)
    ended_at = parse_iso_datetime(summary_data.endedAt) if summary_data.endedAt else None
    insert_query = """
    INSERT INTO Summaries (child_id, call_id, summary, startedAt, endedAt)
    VALUES (:child_id, :call_id, :summary, :startedAt, :endedAt)
    """

    await database.execute(
        query=insert_query,
        values={
            "child_id": current_user["child_id"],
            "call_id": summary_data.call_id,
            "summary": summary_data.summary,
            "startedAt": started_at,
            "endedAt": ended_at,
        },
    )

    return summary_data

def months_between(start_date, end_date):
    rd = relativedelta(end_date, start_date)
    return rd.years * 12 + rd.months

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
    print(child_id)
    query_child_pincode = "SELECT pincode FROM Children WHERE child_id = :child_id"
    child = await database.fetch_one(query=query_child_pincode, values={"child_id": child_id})
    if not child or not child["pincode"]:
        raise HTTPException(status_code=404, detail="Child or pincode not found")
    pincode_str = child["pincode"]

    # Try exact pincode providers first
    query_providers_exact = """
    SELECT provider_id, provider_name, specialty, address, pincode, phone, email
    FROM HealthcareProviders
    WHERE pincode = :pincode
    """
    providers = await database.fetch_all(query=query_providers_exact, values={"pincode": pincode_str})
    print(providers)
    if providers:
        return [Provider(**dict(p)) for p in providers]

    # If none found, try ±2 numeric range if pincode is numeric
    try:
        pincode_num = int(pincode_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pincode format for child")
    print(pincode_num)
    low = pincode_num - 2
    high = pincode_num + 2

    query_providers_range = """
    SELECT provider_id, provider_name, specialty, address, pincode, phone, email
    FROM HealthcareProviders
    WHERE CAST(pincode AS UNSIGNED) BETWEEN :low AND :high
    """
    providers_range = await database.fetch_all(query=query_providers_range, values={"low": low, "high": high})

    return [Provider(**dict(p)) for p in providers_range]


import pandas as pd
from dateutil.relativedelta import relativedelta
from fastapi import Depends, HTTPException



def months_between(start, end):
    rd = relativedelta(end, start)
    return rd.years * 12 + rd.months


async def immunization_status_summary_async(child_id: int, upcoming_window=2):
    # Fetch child info asynchronously
    query_child = "SELECT date_of_birth, first_name, last_name FROM Children WHERE child_id = :cid"
    child = await database.fetch_one(query=query_child, values={"cid": child_id})
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    dob = pd.to_datetime(child["date_of_birth"])
    first_name = child["first_name"]
    last_name = child["last_name"]

    today = pd.to_datetime(datetime.today().date())
    age_months = months_between(dob, today)

    # Fetch VaccineSchedule rows
    schedule_rows = await database.fetch_all(
        "SELECT vaccine_name, dose_number, recommended_age_months FROM VaccineSchedule")
    schedule_data = [dict(row) for row in schedule_rows]

    # Important: explicitly specify columns when creating DataFrame
    schedule = pd.DataFrame(schedule_data, columns=["vaccine_name", "dose_number", "recommended_age_months"])
    schedule["recommended_age_months"] = pd.to_numeric(schedule["recommended_age_months"], errors='coerce')

    # Fetch VaccinationTakenRecords for child
    taken_rows = await database.fetch_all(
        query="SELECT vaccine_name, dose_number, date_administered FROM VaccinationTakenRecords WHERE child_id = :cid",
        values={"cid": child_id}
    )
    taken_data = [dict(row) for row in taken_rows]

    taken = pd.DataFrame(taken_data, columns=["vaccine_name", "dose_number", "date_administered"])
    if not taken.empty:
        taken["date_administered"] = pd.to_datetime(taken["date_administered"])

    taken_on_time = []
    taken_late = []
    missed = []
    upcoming = []

    for _, row in schedule.iterrows():
        vaccine = row["vaccine_name"]
        dose = row["dose_number"]
        due_month = row["recommended_age_months"]

        if pd.isna(due_month):
            continue

        record = taken[(taken["vaccine_name"] == vaccine) & (taken["dose_number"] == dose)]
        if record.empty:
            if due_month <= age_months:
                missed.append(f"{vaccine} dose {dose}")
            elif due_month <= age_months + upcoming_window:
                upcoming.append(f"{vaccine} dose {dose}")
        else:
            date_admin = record.iloc[0]["date_administered"]
            admin_age = months_between(dob, date_admin)
            if admin_age <= due_month:
                taken_on_time.append(f"{vaccine} dose {dose}")
            else:
                taken_late.append(f"{vaccine} dose {dose}")

    summary_parts = []
    if taken_on_time:
        summary_parts.append("✅ " + ", ".join(taken_on_time) + " have been taken on time")
    if taken_late:
        summary_parts.append("❗ " + ", ".join(taken_late) + " were late")
    if missed:
        summary_parts.append("🚫 " + ", ".join(missed) + " were missed")
    if upcoming:
        summary_parts.append("🔜 " + ", ".join(upcoming) + " are due soon")
    if not summary_parts:
        return first_name, last_name, "No vaccination history available"

    summary = "\n".join(summary_parts)
    return first_name, last_name, f"{summary}"


@app.get("/immunization-summary")
async def get_immunization_summary(current_user=Depends(get_current_user)):
    child_id = current_user["child_id"]
    first_name, last_name, summary = await immunization_status_summary_async(child_id)
    print("summary", summary)
    return {
        "child_id": child_id,
        "first_name": first_name,
        "last_name": last_name,
        "summary": summary
    }


# Email config (example, replace with your SMTP server details)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "indalavenkatasrisaihemanth@gmail.com"
SMTP_PASSWORD = "gydp dyzg rvsm towz"
EMAIL_FROM = "indalavenkatasrisaihemanth@gmail.com"

@validator("vaccine_name")
def vaccine_name_not_empty(cls, v):
    if not v.strip():
        raise ValueError("vaccine_name cannot be empty")
    return v

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

@app.post("/vaccine-schedule", status_code=201)
async def add_vaccine_schedule(
    vaccine_data: VaccineScheduleCreate,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
):
    # Optional: add authorization check to allow only admin users

    insert_query = """
    INSERT INTO VaccineSchedule (vaccine_name, dose_number, recommended_age_months, mandatory)
    VALUES (:vaccine_name, :dose_number, :recommended_age_months, :mandatory)
    """
    values = vaccine_data.dict()
    await database.execute(query=insert_query, values=values)

    # Enqueue email notification to run in the background (non-blocking)
    background_tasks.add_task(
        notify_all_children_new_vaccine,
        vaccine_name=vaccine_data.vaccine_name,
        dose_number=vaccine_data.dose_number,
        recommended_age_months=vaccine_data.recommended_age_months,
    )

    return {"message": "Vaccine schedule added and notifications sent."}

# schema = {
#     "Children": ["child_id", "first_name", "last_name", "date_of_birth", "gender", "parent_id", "blood_type", "pincode"],
#     "Allergies": ["allergy_id", "child_id", "allergen", "reaction", "severity", "first_observed", "triggered_by"],
#     "MedicalVisits": ["visit_id", "child_id", "visit_date", "doctor_name", "diagnosis", "treatment", "notes"],
#     "GrowthMetrics": ["entry_id", "child_id", "date_recorded", "height_cm", "weight_kg", "head_circumference_cm", "weight_percentile", "height_percentile", "notes"],
#     "SymptomsLog": ["log_id", "child_id", "date_reported", "symptom_description", "severity", "duration", "notes"],
#     "FollowUpSchedule": ["followup_id", "child_id", "visit_reason", "scheduled_date", "completed", "doctor_notes"],
#     "VaccinationTakenRecords": ["record_id", "child_id", "vaccine_name", "dose_number", "date_administered"],
#     "VaccineSchedule": ["schedule_id", "vaccine_name", "dose_number", "recommended_age_months", "mandatory"],
#     "Parents": ["parent_id", "full_name", "email", "phone_number", "address", "relationship"],
#     "Summaries": ["call_id", "child_id", "summary", "startedAt", "endedAt"],
#     "ChildUsers": ["child_id", "username", "password_hash", "created_at", "updated_at"],
#     "HealthcareProviders": ["provider_id", "provider_name", "speciality", "address", "pincode", "phone", "email"]
# }
#
# # Flatten for validation
# all_tables = set(schema.keys())
# all_columns = {col for cols in schema.values() for col in cols}
#
# # @app.post("/chat-assistant")
# # async def chat_assistant(request: QueryRequest, current_user=Depends(get_current_user)):
# #     import re
# #     from sqlalchemy import create_engine, text
# #
# #     schema_description = "\n".join([f"- {table}({', '.join(cols)})" for table, cols in schema.items()])
# #
# #     system_prompt = {
# #         "role": "system",
# #         "content": (
# #             "You are a helpful assistant for healthcare providers and parents. "
# #             "If the user asks a question that can be answered from the database, generate a MySQL SELECT query. "
# #             "If the question is conversational or unclear, respond politely. "
# #             "Only generate raw SQL when it's relevant and appropriate.\n"
# #             "Here is the database schema:\n"
# #             f"{schema_description}"
# #         )
# #     }
# #
# #     full_chat = [system_prompt] + request.messages
# #
# #     try:
# #         llm_response = groq_client.chat.completions.create(
# #             model="llama-3.1-8b-instant",
# #             messages=full_chat,
# #             temperature=0.3
# #         ).choices[0].message.content.strip()
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")
# #
# #     # ✅ STEP 1: Check if it's a valid SQL SELECT
# #     if not llm_response.lower().startswith("select") or "from" not in llm_response.lower():
# #         return {
# #             "response": llm_response,
# #             "generated_sql": None,
# #             "query_result": None,
# #             "message_history": request.messages + [{"role": "assistant", "content": llm_response}]
# #         }
# #
# #     # ✅ STEP 2: Validate tables/columns
# #     def extract_identifiers(sql):
# #         tables = re.findall(r'from\s+(\w+)', sql, re.IGNORECASE) + re.findall(r'join\s+(\w+)', sql, re.IGNORECASE)
# #         cols = re.findall(r'select\s+(.*?)\s+from', sql, re.IGNORECASE)
# #         col_list = []
# #         if cols:
# #             col_list = [c.strip().split(" as ")[0].strip() for c in cols[0].split(",")]
# #         return set(tables), set(col_list)
# #
# #     tables_used, columns_used = extract_identifiers(llm_response)
# #
# #     if not tables_used.issubset(all_tables) or (
# #         columns_used and not columns_used.issubset(all_columns | {"*"})
# #     ):
# #         return {
# #             "response": "I'm sorry, that query references unavailable data.",
# #             "generated_sql": llm_response,
# #             "query_result": None,
# #             "message_history": request.messages + [{"role": "assistant", "content": "I'm sorry, that query isn't supported."}]
# #         }
# #
# #     # ✅ STEP 3: Run SQL safely
# #     try:
# #         engine = create_engine("mysql+pymysql://root@localhost:3306/ChildHealth")
# #         with engine.connect() as conn:
# #             result = conn.execute(text(llm_response))
# #             rows = result.fetchall()
# #             columns = result.keys()
# #             data = [dict(zip(columns, row)) for row in rows]
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"SQL error: {str(e)}")
# #
# #     if not data:
# #         return {
# #             "response": "No relevant data found.",
# #             "generated_sql": llm_response,
# #             "query_result": [],
# #             "message_history": request.messages + [{"role": "assistant", "content": "No relevant data found."}]
# #         }
# #
# #     # ✅ STEP 4: Summarize result
# #     summary_prompt = [
# #         {"role": "system", "content": "Summarize this SQL result for a parent or healthcare provider in friendly language."},
# #         {"role": "user", "content": str(data)}
# #     ]
# #
# #     try:
# #         summary = groq_client.chat.completions.create(
# #             model="llama-3.1-8b-instant",
# #             messages=summary_prompt,
# #             temperature=0.3
# #         ).choices[0].message.content.strip()
# #
# #         return {
# #             "response": summary,
# #             "generated_sql": llm_response,
# #             "query_result": data,
# #             "message_history": request.messages + [{"role": "assistant", "content": summary}]
# #         }
# #
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")
# @app.post("/chat-assistant")
# async def chat_assistant(request: QueryRequest, current_user=Depends(get_current_user)):
#     schema_description = "\n".join([f"- {table}({', '.join(cols)})" for table, cols in schema.items()])
#
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are a helpful and precise healthcare SQL assistant.\n"
#                 "Always convert the user's message into a valid MySQL SELECT query strictly using only the following schema.\n"
#                 "If the user's request is unrelated or references unknown tables/fields, reply with:\n"
#                 "'I'm sorry, we don't have that information available.'\n\n"
#                 "Schema:\n" + schema_description
#             )
#         }
#     ] + request.messages
#
#     try:
#         sql_response = groq_client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=messages,
#             temperature=0.2
#         ).choices[0].message.content.strip()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"LLM error: {e}")
#
#     if "```" in sql_response:
#         sql_response = sql_response.split("```")[1].replace("sql", "").strip()
#
#     # Simple check: did it return SQL?
#     if not sql_response.lower().startswith("select") or "from" not in sql_response.lower():
#         return {
#             "response": "I'm sorry, we don't have that information available.",
#             "generated_sql": None,
#             "query_result": None,
#             "message_history": request.messages + [{"role": "assistant", "content": "I'm sorry, we don't have that information available."}]
#         }
#
#     def extract_identifiers(sql):
#         tables = re.findall(r'from\\s+(\\w+)', sql, re.IGNORECASE) + re.findall(r'join\\s+(\\w+)', sql, re.IGNORECASE)
#         cols = re.findall(r'select\\s+(.*?)\\s+from', sql, re.IGNORECASE)
#         col_list = []
#         if cols:
#             col_list = [c.strip().split(" as ")[0].strip() for c in cols[0].split(",")]
#         return set(tables), set(col_list)
#
#     tables_used, columns_used = extract_identifiers(sql_response)
#
#     if not tables_used.issubset(all_tables) or (columns_used and not columns_used.issubset(all_columns | {"*"})):
#         return {
#             "response": "I'm sorry, we don't have that information available.",
#             "generated_sql": sql_response,
#             "query_result": None,
#             "message_history": request.messages + [{"role": "assistant", "content": "I'm sorry, we don't have that information available."}]
#         }
#
#     try:
#         engine = create_engine("mysql+pymysql://root@localhost:3306/ChildHealth")
#         with engine.connect() as conn:
#             result = conn.execute(text(sql_response))
#             rows = result.fetchall()
#             columns = result.keys()
#             data = [dict(zip(columns, row)) for row in rows]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"SQL error: {e}")
#
#     if not data:
#         return {
#             "response": "I'm sorry, we don't have that information available.",
#             "generated_sql": sql_response,
#             "query_result": [],
#             "message_history": request.messages + [{"role": "assistant", "content": "I'm sorry, we don't have that information available."}]
#         }
#
#     summary_prompt = [
#         {"role": "system", "content": "Summarize this SQL result for a parent or provider:"},
#         {"role": "user", "content": str(data)}
#     ]
#
#     try:
#         summary_response = groq_client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=summary_prompt,
#             temperature=0.8
#         ).choices[0].message.content.strip()
#
#         print(summary_response)
#
#
#         return {
#             "response": summary_response,
#             "generated_sql": sql_response,
#             "query_result": data,
#             "message_history": request.messages + [{"role": "assistant", "content": summary_response}]
#         }
#
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Summary failed: {str(e)}")
schema = {
    "Children": ["child_id", "first_name", "last_name", "date_of_birth", "gender", "blood_type", "pincode"],
    "VaccinationTakenRecords": ["record_id", "child_id", "vaccine_name", "dose_number", "date_administered", "administered_by", "reminder_sent", "notes"],
    "VaccineSchedule": ["schedule_id", "vaccine_name", "dose_number", "recommended_age_months", "mandatory", "description", "benefits"],
    "Summaries": ["call_id", "child_id", "summary", "startedAt", "endedAt"],
    "ChildUsers": ["child_id", "username", "password_hash", "created_at", "updated_at"],
    "HealthcareProviders": ["provider_id", "provider_name", "specialty", "address", "pincode", "phone", "email"],
    "Appointments":["appointment_id", "child_id ", "appointment_date", "provider_id", "schedule_id"]
}

all_tables = set(schema.keys())
all_columns = {col for cols in schema.values() for col in cols}

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class QueryRequest(BaseModel):
    messages: List[ChatMessage]
class VaccineDetailResponse(BaseModel):
    vaccine_name: str
    description: Optional[str] = None
    benefits: Optional[str] = None

@app.post("/chat-assistant")
async def chat_assistant(request: QueryRequest):
    # Compose schema description string for prompt
    schema_description = "\n".join([f"- {table}({', '.join(cols)})" for table, cols in schema.items()])

    # Combine system prompt + conversation messages
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful and precise healthcare SQL assistant.\n"
            "Always convert the user's message into a valid MySQL SELECT query very strictly using only the following schema.\n"
            "If the user's request is unrelated or references unknown tables/fields, reply exactly with:\n"
            "'I'm sorry, we don't have that information available.'\n\n"
            "Schema:\n" + schema_description
        )
    }
    messages = [system_prompt] + request.messages

    # Generate SQL query using Groq LLM
    try:
        sql_response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.3
        ).choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # Strip markdown formatting if present
    if "```" in sql_response:
        sql_response = sql_response.split("```")[1].replace("sql", "").strip()

    # Basic validation: must start with SELECT and contain FROM
    if not sql_response.lower().startswith("select") or "from" not in sql_response.lower():
        return {
            "response": "I'm sorry, we don't have that information available.",
            "generated_sql": None,
            "query_result": None,
            "message_history": request.messages + [{"role": "assistant", "content": "I'm sorry, we don't have that information available."}]
        }

    # Extract referenced tables and columns from SQL for validation
    def extract_identifiers(sql):
        tables = re.findall(r'(from|join)\s+(\w+)', sql, re.IGNORECASE)
        base_tables = set([match[1] for match in tables])
        col_match = re.search(r'select\s+(.*?)\s+from', sql, re.IGNORECASE | re.DOTALL)
        col_list = []
        if col_match:
            cols = col_match.group(1).split(',')
            col_list = [c.strip().split(' as ')[0].strip().split('.')[-1] for c in cols]
        return base_tables, set(col_list)

    print(sql_response)
    referenced_tables, referenced_columns = extract_identifiers(sql_response)

    if not referenced_tables.issubset(all_tables):
        return {"response": "I'm sorry, we don't have that information available."}

    # Allow aggregate functions in columns
    aggregates = {"count", "sum", "avg", "min", "max"}
    cleaned_columns = set()
    for col in referenced_columns:
        lower_col = col.lower()
        if any(lower_col.startswith(f"{agg}(") for agg in aggregates):
            inner_col = re.findall(r'\((.*?)\)', lower_col)
            if inner_col:
                cleaned_columns.add(inner_col[0])
        else:
            cleaned_columns.add(col)

    if cleaned_columns and not cleaned_columns.issubset(all_columns | {"*"}):
        return {"response": "I'm sorry, we don't have that information available."}

    # Run SQL query
    try:
        engine = create_engine("mysql+pymysql://root@localhost:3306/ChildHealth")
        with engine.connect() as conn:
            result = conn.execute(text(sql_response))
            rows = result.fetchall()
            columns = result.keys()
            data = [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL error: {e}")

    if not data:
        return {"response": "I'm sorry, we don't have that information available."}

    # Generate natural language summary of results
    summary_prompt = [
        {"role": "system", "content": """You are a helpful assistant. Based on the following database query result, generate a short, natural, clear, and concise summary in third person, avoiding unnecessary pleasantries.

Only respond with essential facts. Use third person.

If data is irrelevant or empty, say: "I'm sorry, we don't have that information available."""},
        {"role": "user", "content": str(data)}
    ]

    try:
        summary_response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=summary_prompt,
            temperature=0.6
        ).choices[0].message.content.strip()

        return {
            "response": summary_response,
            "generated_sql": sql_response,
            "query_result": data,
            "message_history": request.messages + [{"role": "assistant", "content": summary_response}]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation error: {str(e)}")


# New disease outbreak function
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
            return []  # Return an empty list instead of raising an error

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

class AppointmentCreate(BaseModel):
    child_id: int
    appointment_date: str  # ISO format date string
    provider_id: int
    schedule_id: int


@app.post("/appointments", status_code=201)
async def create_appointment(
    appointment: AppointmentCreate,
    current_user=Depends(get_current_user)
):
    # You may validate child_id matches current_user.child_id here

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
