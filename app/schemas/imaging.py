from pydantic import BaseModel
from typing import Optional, Dict

class ImagingResponse(BaseModel):
    id: int
    patient_id: int
    file_path: str
    description: str

    class Config:
        from_attributes = True


# Schemas for X-ray prediction
class XRayPredictionResult(BaseModel):
    """X-ray prediction result."""
    class_id: int
    class_name: str
    confidence: float


class XRayPredictionProbabilities(BaseModel):
    """Probabilities for all X-ray classes."""
    probabilities: Dict[str, float]


class XRayAnalysisResponse(BaseModel):
    """Complete X-ray analysis response."""
    prediction: XRayPredictionResult
    probabilities: XRayPredictionProbabilities
    message: Optional[str] = None
