from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..db_models.lab_results import LabResult
from ..schemas.lab_results import (
    LabCreate, LabResponse, BloodCountInput, BloodCountAnalysisResponse,
    ParsedValues, PredictionResult, PredictionProbabilities
)
from ..ml_models.bloodcount_report.feature import predict_blood_disease, predict_blood_disease_proba

router = APIRouter(prefix="/labs", tags=["Lab Results"])

@router.post("/", response_model=LabResponse)
def create_lab(data: LabCreate, db: Session = Depends(get_db)):
    lab = LabResult(**data.dict())
    db.add(lab)
    db.commit()
    db.refresh(lab)
    return lab

@router.get("/by_patient/{patient_id}")
def get_labs(patient_id: int, db: Session = Depends(get_db)):
    return db.query(LabResult).filter(LabResult.patient_id == patient_id).all()


# Healthy/normal range values (midpoints or typical healthy values)
HEALTHY_VALUES = {
    'WBC': 7.0,      # White Blood Cell count (x1000/µL) - normal range 4-10
    'LYMp': 35.0,    # Lymphocyte percentage - normal range 20-40
    'NEUTp': 60.0,   # Neutrophil percentage - normal range 50-70
    'LYMn': 2.0,     # Lymphocyte absolute count (x1000/µL) - normal range 1.0-3.0
    'NEUTn': 4.0,    # Neutrophil absolute count (x1000/µL) - normal range 2.0-7.0
    'RBC': 4.8,      # Red Blood Cell count (x10^6/µL) - normal range 4.5-5.5
    'HGB': 14.5,     # Hemoglobin (g/dL) - normal range 13.0-17.0
    'HCT': 42.0,     # Hematocrit (%) - normal range 40-50
    'MCV': 90.0,     # Mean Corpuscular Volume (fL) - normal range 83-101
    'MCH': 30.0,     # Mean Corpuscular Hemoglobin (pg) - normal range 27-32
    'MCHC': 33.0,    # Mean Corpuscular Hemoglobin Concentration (g/dL) - normal range 32.5-34.5
    'PLT': 250.0,    # Platelet count (x10^3/µL) - normal range 150-450
    'PDW': 12.0,     # Platelet Distribution Width - normal range 10-15
    'PCT': 0.25,     # Plateletcrit - normal range 0.15-0.35
}


def fill_missing_values(input_data: dict) -> dict:
    """
    Fill missing values with healthy range defaults.
    
    Args:
        input_data: Dictionary with user-provided values (may have None values)
        
    Returns:
        Dictionary with all values filled (user values + healthy defaults)
    """
    filled_data = {}
    for param in HEALTHY_VALUES.keys():
        if input_data.get(param) is not None:
            # Use user-provided value
            filled_data[param] = input_data[param]
        else:
            # Use healthy default value
            filled_data[param] = HEALTHY_VALUES[param]
    return filled_data


@router.post("/analyze-blood-count", response_model=BloodCountAnalysisResponse)
def analyze_blood_count(input_data: BloodCountInput):
    """
    Analyze blood count values and predict disease.
    
    This endpoint:
    1. Accepts manual input of blood count values (all fields optional)
    2. Fills missing values with healthy/normal range defaults
    3. Makes disease prediction using ML model
    
    If a value is not provided, it will be set to a healthy/normal range value.
    
    Returns prediction result with probabilities for all disease classes.
    """
    try:
        # Convert input to dictionary
        input_dict = input_data.dict()
        
        # Fill missing values with healthy defaults
        filled_dict = fill_missing_values(input_dict)
        
        # Track which values were provided by user vs defaults
        user_provided = [k for k, v in input_dict.items() if v is not None]
        default_filled = [k for k, v in input_dict.items() if v is None]
        
        # Make prediction with filled values
        pred_result = predict_blood_disease(filled_dict)
        prob_result = predict_blood_disease_proba(filled_dict)
        
        # Create response message
        message_parts = []
        if user_provided:
            message_parts.append(f"Analyzed with {len(user_provided)} user-provided values")
        if default_filled:
            message_parts.append(f"{len(default_filled)} values set to healthy defaults")
        message = ". ".join(message_parts) + "." if message_parts else "Analysis complete."
        
        return BloodCountAnalysisResponse(
            input_values=input_data,
            filled_values=ParsedValues(**filled_dict),
            prediction=PredictionResult(
                class_id=pred_result['class_id'],
                disease_name=pred_result['disease_name'],
                confidence=pred_result.get('confidence')
            ),
            probabilities=PredictionProbabilities(probabilities=prob_result),
            message=message
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
