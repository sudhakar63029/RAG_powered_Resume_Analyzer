from pydantic import BaseModel
from typing import List

class AnalysisResponse(BaseModel):
    filename: str
    ats_score : int
    missing_skills: List[str]
    feedback: str
