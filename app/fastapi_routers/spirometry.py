from fastapi import APIRouter, HTTPException
from ..schemas.spirometry import (
    SpirometryInput, SpirometryAnalysisResponse,
    SpirometryPredictionResult, SpirometryPredictionProbabilities
)
from ..ml_models.spirometry.featurizer import predict_spirometry, predict_spirometry_proba

router = APIRouter(prefix="/spirometry", tags=["Spirometry"])

# Healthy/normal range values for spirometry (midpoints or typical healthy values)
HEALTHY_VALUES = {
    'sex': 'Male',  # Default
    'race': 'White',  # Default
    'age': 45.0,
    'height': 170.0,  # cm
    'weight': 70.0,   # kg
    'bmi': 24.0,
    'fev1': 3.5,     # liters
    'fvc': 4.5,      # liters
    # fev1_fvc will be calculated automatically
}


def fill_missing_spirometry_values(input_data: dict) -> dict:
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


@router.post("/analyze", response_model=SpirometryAnalysisResponse)
def analyze_spirometry(input_data: SpirometryInput):
    """
    Analyze spirometry values and predict conditions.
    
    This endpoint:
    1. Accepts manual input of spirometry values (all fields optional)
    2. Fills missing values with healthy/normal range defaults
    3. Makes predictions for 4 conditions:
       - Obstruction
       - Restriction
       - PRISm (Preserved Ratio Impaired Spirometry)
       - Mixed patterns
    
    If a value is not provided, it will be set to a healthy/normal range value.
    
    Returns prediction results with probabilities for all conditions.
    """
    try:
        # Convert input to dictionary
        input_dict = input_data.dict()
        
        # Fill missing values with healthy defaults
        filled_dict = fill_missing_spirometry_values(input_dict)
        
        # Track which values were provided by user vs defaults
        user_provided = [k for k, v in input_dict.items() if v is not None]
        default_filled = [k for k, v in input_dict.items() if v is None]
        
        # Make prediction with filled values
        pred_result = predict_spirometry(filled_dict)
        prob_result = predict_spirometry_proba(filled_dict)
        
        # Check if all predictions are 0 (healthy)
        all_zero = (
            pred_result['obstruction'] == 0 and
            pred_result['restriction'] == 0 and
            pred_result['prism'] == 0 and
            pred_result['mixed'] == 0
        )
        status = "healthy" if all_zero else "abnormal"
        
        # Create response message
        message_parts = []
        if user_provided:
            message_parts.append(f"Analyzed with {len(user_provided)} user-provided values")
        if default_filled:
            message_parts.append(f"{len(default_filled)} values set to healthy defaults")
        if status == "healthy":
            message_parts.append("All conditions normal - healthy status")
        message = ". ".join(message_parts) + "." if message_parts else "Analysis complete."
        
        return SpirometryAnalysisResponse(
            input_values=input_data,
            prediction=SpirometryPredictionResult(
                obstruction=pred_result['obstruction'],
                restriction=pred_result['restriction'],
                prism=pred_result['prism'],
                mixed=pred_result['mixed']
            ),
            probabilities=SpirometryPredictionProbabilities(
                obstruction=prob_result['obstruction'],
                restriction=prob_result['restriction'],
                prism=prob_result['prism'],
                mixed=prob_result['mixed']
            ),
            status=status,
            message=message
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

