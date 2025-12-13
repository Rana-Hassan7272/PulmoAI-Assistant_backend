from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from ..core.database import Base

class Diagnosis(Base):
    __tablename__ = "diagnosis"

    id = Column(Integer, primary_key=True, index=True)
    visit_id = Column(Integer, ForeignKey("visits.id"))
    diagnosis = Column(Text)
    treatment_plan = Column(Text)  # JSON string for array of treatments
    tests_recommended = Column(Text, nullable=True)  # JSON string for array of tests
    home_remedies = Column(Text, nullable=True)  # JSON string for array of remedies
    followup_instruction = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    visit = relationship("Visit")

