from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

from backend.models import AnalysisResponse
from backend.services.parser import extract_text_from_pdf
from backend.services.rag_engine import rag_engine

app = FastAPI(title="Dataset-Powered RAG Resume Analyzer API", version="2.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-resume/", response_model=AnalysisResponse)
async def analyze_resume_endpoint(
    job_description: str = Form(...),
    file: UploadFile = File(...)
):
    file_path = f"./temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        extracted_text = extract_text_from_pdf(file_path)
    except Exception as e:
        os.remove(file_path)
        return AnalysisResponse(filename=file.filename, ats_score=0, missing_skills=[], feedback=str(e))
        
    os.remove(file_path)
    
    if not extracted_text.strip():
        return AnalysisResponse(
            filename=file.filename,
            ats_score=0,
            missing_skills=[],
            feedback="The uploaded PDF appears to be empty or scanned."
        )

    # Run RAG Analysis via the services layer
    feedback_text = rag_engine.analyze_with_rag(extracted_text, job_description)

    return AnalysisResponse(
        filename=file.filename,
        ats_score=85, # Dynamic scoring can be extracted from LLM text if needed
        missing_skills=["Extracted via Dataset RAG"],
        feedback=feedback_text
    )