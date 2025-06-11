from fastapi import FastAPI
from routes import router as pdf_router

app = FastAPI()

# Include the PDF upload router
app.include_router(pdf_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 