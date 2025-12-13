from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from ..core.database import Base

class Visit(Base):
    __tablename__ = "visits"

    id = Column(Integer, primary_key=True, index=True)
    visit_id = Column(String, unique=True, nullable=True, index=True)  # Unique visit identifier
    patient_id = Column(Integer, ForeignKey("patients.id"))
    symptoms = Column(Text)
    doctor_notes = Column(Text)
    diagnosis = Column(Text)
    vitals = Column(Text, nullable=True)  # JSON string for vitals data
    emergency_flag = Column(Boolean, default=False)
    xray_result = Column(Text, nullable=True)  # JSON string for X-ray analysis result
    spirometry_result = Column(Text, nullable=True)  # JSON string for spirometry result
    cbc_result = Column(Text, nullable=True)  # JSON string for CBC result
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient")
