from sqlalchemy import Column, Integer, String, Boolean, Text
from ..core.database import Base

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    smoker = Column(Boolean, default=False)
    chronic_conditions = Column(Text, nullable=True)  # JSON string or comma-separated
    occupation = Column(String, nullable=True)
