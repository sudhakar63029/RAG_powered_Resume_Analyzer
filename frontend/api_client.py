import os
import requests

# Automatically use Render/Railway backend URL if set, otherwise default to local
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000") + "/analyze-resume/"

def send_resume_for_analysis(job_description: str, file_name: str, file_bytes: bytes):
    files = {"file": (file_name, file_bytes, "application/pdf")}
    data = {"job_description": job_description}
    
    try:
        response = requests.post(BACKEND_URL, files=files, data=data, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Server error: {response.status_code} - {response.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to FastAPI backend service. Is it live?"}