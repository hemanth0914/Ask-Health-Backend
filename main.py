from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from databases import Database
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


DATABASE_URL = "mysql+aiomysql://root:@localhost:3306/childhealth"

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

database = Database(DATABASE_URL)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

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
    first_name: str
    last_name: str
    vaccine_name: str
    next_due_date: str

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
    WHERE child_id = :child_id AND summary != 'No summary available'
    ORDER BY startedAt DESC
    """
    results = await database.fetch_all(query=query, values={"child_id": current_user["child_id"]})

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

@app.get("/upcoming-vaccines", response_model=List[UpcomingVaccine])
async def get_upcoming_vaccines(current_user=Depends(get_current_user)):
    query = """
    SELECT c.first_name, c.last_name, v.vaccine_name, v.next_due_date
    FROM Children c
    JOIN VaccinationRecords v ON c.child_id = v.child_id
    WHERE v.next_due_date BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL 30 DAY)
      AND v.reminder_sent = FALSE
      AND c.child_id = :child_id
    ORDER BY v.next_due_date
    """
    results = await database.fetch_all(query=query, values={"child_id": current_user["child_id"]})
    # Convert date fields to string if needed
    return [
        {
            "first_name": r["first_name"],
            "last_name": r["last_name"],
            "vaccine_name": r["vaccine_name"],
            "next_due_date": r["next_due_date"].isoformat() if r["next_due_date"] else None,
        }
        for r in results
    ]

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
        summary_parts.append(", ".join(taken_on_time) + " have been taken on time")
    if taken_late:
        summary_parts.append(", ".join(taken_late) + " were late")
    if missed:
        summary_parts.append(", ".join(missed) + " were missed")
    if upcoming:
        summary_parts.append(", ".join(upcoming) + " are due soon")
    if not summary_parts:
        return first_name, last_name, "No vaccination history available"

    summary = ". ".join(summary_parts) + "."
    return first_name, last_name, summary

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




