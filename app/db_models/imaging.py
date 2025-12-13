from sqlalchemy import Column, Integer, String, ForeignKey
from ..core.database import Base

class Imaging(Base):
    __tablename__ = "imaging"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    file_path = Column(String)
    description = Column(String)
