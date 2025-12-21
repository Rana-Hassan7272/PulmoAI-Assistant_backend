from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
import os
import logging
from ..core.database import get_db
from ..db_models.visit import Visit
from ..schemas.visit import VisitCreate, VisitResponse
from ..core.migrations import migrate_add_pdf_report_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visits", tags=["Visits"])

@router.post("/", response_model=VisitResponse)
def create_visit(data: VisitCreate, db: Session = Depends(get_db)):
    try:
        visit = Visit(**data.dict())
        db.add(visit)
        db.commit()
        db.refresh(visit)
        return visit
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating visit: {e}")
        raise HTTPException(status_code=500, detail="Failed to create visit")

@router.get("/by_patient/{patient_id}")
def get_patient_visits(patient_id: int, db: Session = Depends(get_db)):
    """
    Get all visits for a patient.
    Handles missing columns gracefully by running migration if needed.
    """
    try:
        visits = db.query(Visit).filter(Visit.patient_id == patient_id).all()
        return visits
    except OperationalError as e:
        error_msg = str(e).lower()
        if "no such column" in error_msg and "pdf_report_path" in error_msg:
            # Column missing - try to run migration
            logger.warning("pdf_report_path column missing. Running migration...")
            try:
                migrate_add_pdf_report_path()
                # Retry the query
                visits = db.query(Visit).filter(Visit.patient_id == patient_id).all()
                return visits
            except Exception as migration_error:
                logger.error(f"Migration failed: {migration_error}")
                # Return visits without pdf_report_path field
                visits = db.query(Visit).filter(Visit.patient_id == patient_id).all()
                # Manually set pdf_report_path to None for all visits
                for visit in visits:
                    if not hasattr(visit, 'pdf_report_path'):
                        visit.pdf_report_path = None
                return visits
        else:
            logger.error(f"Database error: {e}")
            raise HTTPException(status_code=500, detail="Database error occurred")
    except Exception as e:
        logger.error(f"Error fetching visits: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch visits")

@router.get("/{visit_id}/report")
def download_report(visit_id: str, db: Session = Depends(get_db)):
    """
    Download PDF report for a specific visit.
    """
    visit = db.query(Visit).filter(Visit.visit_id == visit_id).first()
    if not visit:
        raise HTTPException(status_code=404, detail="Visit not found")
    
    if not visit.pdf_report_path:
        raise HTTPException(status_code=404, detail="PDF report not available for this visit")
    
    # Get absolute path to PDF file
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pdf_absolute_path = os.path.join(base_dir, visit.pdf_report_path)
    
    if not os.path.exists(pdf_absolute_path):
        raise HTTPException(status_code=404, detail="PDF file not found on server")
    
    return FileResponse(
        pdf_absolute_path,
        media_type="application/pdf",
        filename=f"medical_report_{visit_id}.pdf"
    )
