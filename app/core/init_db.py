from .database import Base, engine
from ..db_models.patient import Patient
from ..db_models.visit import Visit
from ..db_models.lab_results import LabResult
from ..db_models.imaging import Imaging
from ..db_models.diagnosis import Diagnosis

def init_db():
    Base.metadata.create_all(bind=engine)
