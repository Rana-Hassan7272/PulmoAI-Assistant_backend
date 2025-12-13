from pydantic import BaseModel
from typing import Optional, Dict, List

class LabCreate(BaseModel):
    patient_id: int
    test_name: str
    value: float
    unit: str

class LabResponse(LabCreate):
    id: int
    class Config:
        from_attributes = True


# Schemas for OCR and prediction
class OCRResult(BaseModel):
    """OCR extraction result."""
    text: str
    success: bool
    message: Optional[str] = None


class ParsedValues(BaseModel):
    """Parsed blood count values."""
    WBC: Optional[float] = None
    LYMp: Optional[float] = None
    NEUTp: Optional[float] = None
    LYMn: Optional[float] = None
    NEUTn: Optional[float] = None
    RBC: Optional[float] = None
    HGB: Optional[float] = None
    HCT: Optional[float] = None
    MCV: Optional[float] = None
    MCH: Optional[float] = None
    MCHC: Optional[float] = None
    PLT: Optional[float] = None
    PDW: Optional[float] = None
    PCT: Optional[float] = None


class ValidationStatus(BaseModel):
    """Validation status for parsed values."""
    is_valid: bool
    missing_params: List[str]
    extracted_count: int
    total_count: int
    extracted_params: List[str]


class PredictionResult(BaseModel):
    """Disease prediction result."""
    class_id: int
    disease_name: str
    confidence: Optional[float] = None


class PredictionProbabilities(BaseModel):
    """Probabilities for all disease classes."""
    probabilities: Dict[str, float]


class LabReportAnalysisResponse(BaseModel):
    """Complete analysis response."""
    ocr_result: OCRResult
    parsed_values: ParsedValues
    validation: ValidationStatus
    prediction: Optional[PredictionResult] = None
    probabilities: Optional[PredictionProbabilities] = None
    error: Optional[str] = None


# Schema for manual input (all fields optional)
class BloodCountInput(BaseModel):
    """Manual input for blood count values - all fields optional."""
    WBC: Optional[float] = None  # White Blood Cell count
    LYMp: Optional[float] = None  # Lymphocyte percentage
    NEUTp: Optional[float] = None  # Neutrophil percentage
    LYMn: Optional[float] = None  # Lymphocyte absolute count
    NEUTn: Optional[float] = None  # Neutrophil absolute count
    RBC: Optional[float] = None  # Red Blood Cell count
    HGB: Optional[float] = None  # Hemoglobin
    HCT: Optional[float] = None  # Hematocrit
    MCV: Optional[float] = None  # Mean Corpuscular Volume
    MCH: Optional[float] = None  # Mean Corpuscular Hemoglobin
    MCHC: Optional[float] = None  # Mean Corpuscular Hemoglobin Concentration
    PLT: Optional[float] = None  # Platelet count
    PDW: Optional[float] = None  # Platelet Distribution Width
    PCT: Optional[float] = None  # Plateletcrit


class BloodCountAnalysisResponse(BaseModel):
    """Response for manual blood count analysis."""
    input_values: BloodCountInput  # What user entered
    filled_values: ParsedValues  # Values used for prediction (with defaults filled)
    prediction: PredictionResult
    probabilities: PredictionProbabilities
    message: Optional[str] = None
