from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..db_models.visit import Visit
from ..schemas.visit import VisitCreate, VisitResponse

router = APIRouter(prefix="/visits", tags=["Visits"])

@router.post("/", response_model=VisitResponse)
def create_visit(data: VisitCreate, db: Session = Depends(get_db)):
    visit = Visit(**data.dict())
    db.add(visit)
    db.commit()
    db.refresh(visit)
    return visit

@router.get("/by_patient/{patient_id}")
def get_patient_visits(patient_id: int, db: Session = Depends(get_db)):
    return db.query(Visit).filter(Visit.patient_id == patient_id).all()
