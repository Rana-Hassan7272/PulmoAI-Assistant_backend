from pydantic import BaseModel
from typing import Optional

class VisitCreate(BaseModel):
    patient_id: int
    visit_id: Optional[str] = None
    symptoms: str
    doctor_notes: str = ""
    diagnosis: str = ""
    vitals: Optional[str] = None
    emergency_flag: Optional[bool] = False
    xray_result: Optional[str] = None
    spirometry_result: Optional[str] = None
    cbc_result: Optional[str] = None

class VisitResponse(BaseModel):
    id: int
    visit_id: Optional[str]
    patient_id: int
    symptoms: str
    doctor_notes: str
    diagnosis: str
    vitals: Optional[str]
    emergency_flag: bool
    xray_result: Optional[str]
    spirometry_result: Optional[str]
    cbc_result: Optional[str]
    created_at: str

    class Config:
        from_attributes = True
