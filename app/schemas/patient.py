from pydantic import BaseModel
from typing import Optional

class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    smoker: Optional[bool] = False
    chronic_conditions: Optional[str] = None
    occupation: Optional[str] = None

class PatientResponse(PatientCreate):
    id: int
    class Config:
        from_attributes = True
