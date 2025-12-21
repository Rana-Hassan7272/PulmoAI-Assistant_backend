import logging
from .database import Base, engine
from ..db_models.patient import Patient
from ..db_models.visit import Visit
from ..db_models.lab_results import LabResult
from ..db_models.imaging import Imaging
from ..db_models.diagnosis import Diagnosis
from ..db_models.user import User
from .migrations import run_migrations

logger = logging.getLogger(__name__)

def init_db():
    """Initialize database: create tables and run migrations"""
    try:
        # Create all tables (if they don't exist)
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")
        
        # Run migrations to add new columns to existing tables
        run_migrations()
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
