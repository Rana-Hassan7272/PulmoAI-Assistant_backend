from pydantic import BaseModel
from typing import Optional, Dict

class SpirometryInput(BaseModel):
    """Input for spirometry prediction - all fields optional."""
    sex: Optional[str] = None  # 'Male' or 'Female'
    race: Optional[str] = None  # Race category
    age: Optional[float] = None
    height: Optional[float] = None  # in cm
    weight: Optional[float] = None  # in kg
    bmi: Optional[float] = None
    fev1: Optional[float] = None  # FEV1 in liters
    fvc: Optional[float] = None  # FVC in liters
    # Note: fev1_fvc will be calculated automatically as fev1/fvc


class SpirometryPredictionResult(BaseModel):
    """Spirometry prediction result."""
    obstruction: int  # 0 or 1
    restriction: int  # 0 or 1
    prism: int  # 0 or 1
    mixed: int  # 0 or 1


class SpirometryPredictionProbabilities(BaseModel):
    """Probabilities for all spirometry conditions."""
    obstruction: float
    restriction: float
    prism: float
    mixed: float


class SpirometryAnalysisResponse(BaseModel):
    """Complete spirometry analysis response."""
    input_values: SpirometryInput
    prediction: SpirometryPredictionResult
    probabilities: SpirometryPredictionProbabilities
    status: str  # "healthy" if all predictions are 0, otherwise "abnormal"
    message: Optional[str] = None

