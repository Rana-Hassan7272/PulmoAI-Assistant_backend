from sqlalchemy import Column, Integer, String, ForeignKey, Float
from ..core.database import Base
class LabResult(Base):
    __tablename__ = "lab_results"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    test_name = Column(String)
    value = Column(Float)
    unit = Column(String)
