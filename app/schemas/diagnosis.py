from pydantic import BaseModel
from typing import Optional

class DiagnosisCreate(BaseModel):
    visit_id: int
    diagnosis: str
    treatment_plan: str  # JSON string for array of treatments
    tests_recommended: Optional[str] = None  # JSON string for array of tests
    home_remedies: Optional[str] = None  # JSON string for array of remedies
    followup_instruction: Optional[str] = None

class DiagnosisResponse(BaseModel):
    id: int
    visit_id: int
    diagnosis: str
    treatment_plan: str
    tests_recommended: Optional[str]
    home_remedies: Optional[str]
    followup_instruction: Optional[str]
    created_at: str

    class Config:
        from_attributes = True

