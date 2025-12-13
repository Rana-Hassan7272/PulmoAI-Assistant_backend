from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import shutil
import io
import os
from PIL import Image
from ..core.database import get_db
from ..db_models.imaging import Imaging
from ..schemas.imaging import ImagingResponse, XRayAnalysisResponse, XRayPredictionResult, XRayPredictionProbabilities
from ..ml_models.xray.preprocessor import predict_xray, predict_xray_proba

router = APIRouter(prefix="/imaging", tags=["Imaging"])

# IMPORTANT: More specific routes must come before parameterized routes
@router.post("/analyze-xray", response_model=XRayAnalysisResponse)
async def analyze_xray(file: UploadFile = File(...)):
    """
    Analyze X-ray image and predict pneumonia.
    
    This endpoint:
    1. Accepts X-ray image upload
    2. Preprocesses the image
    3. Makes prediction using ML model
    4. Returns prediction result with probabilities
    
    Returns prediction for: No disease, Bacterial pneumonia, or Viral pneumonia.
    """
    try:
        # Read and validate image file
        contents = await file.read()
        
        # Check file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image file (JPEG, PNG, etc.)"
            )
        
        # Convert to PIL Image
        try:
            image = Image.open(io.BytesIO(contents))
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to open image: {str(e)}"
            )
        
        # Make prediction
        try:
            pred_result = predict_xray(image)
            prob_result = predict_xray_proba(image)
            
            return XRayAnalysisResponse(
                prediction=XRayPredictionResult(
                    class_id=pred_result['class_id'],
                    class_name=pred_result['class_name'],
                    confidence=pred_result['confidence']
                ),
                probabilities=XRayPredictionProbabilities(probabilities=prob_result),
                message="X-ray analysis complete"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/{patient_id}", response_model=ImagingResponse)
def upload_xray(patient_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload X-ray image and save to database.
    """
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    file_location = f"uploads/{file.filename}"

    with open(file_location, "wb+") as f:
        shutil.copyfileobj(file.file, f)

    record = Imaging(
        patient_id=patient_id,
        file_path=file_location,
        description="Uploaded X-ray"
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return record
