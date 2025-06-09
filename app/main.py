from fastapi import Depends, HTTPException, Query
from openai import OpenAI
import os
from dotenv import load_dotenv
from . import app
from .database import SessionLocal
from .auth import get_current_user

# Load environment variables
load_dotenv()

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

@app.on_event("startup")
async def startup():
    try:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized successfully")
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise 