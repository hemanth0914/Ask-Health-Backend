from fastapi import APIRouter, HTTPException, File, UploadFile, status
import tempfile
import os
from pdf_processor import process_pdf_and_store_data
from typing import Dict

router = APIRouter()

@router.post("/provider/upload-client-pdf", status_code=status.HTTP_201_CREATED, response_model=Dict)
async def upload_client_pdf(pdf_file: UploadFile = File(...)):
    if not pdf_file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )

    try:
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            content = await pdf_file.read()
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name

        try:
            # Process the PDF and store data
            result = process_pdf_and_store_data(temp_pdf_path)
            return result

        finally:
            # Clean up temporary file
            if os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)

    except Exception as e:
        error_message = str(e)
        if "Error parsing data" in error_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not parse PDF data. Please check the PDF format."
            )
        elif "Database error" in error_message:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while processing the PDF."
            )
        elif "Failed to send email" in error_message:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="PDF processed successfully but failed to send email notification."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing PDF: {error_message}"
            ) 