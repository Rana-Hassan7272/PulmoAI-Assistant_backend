"""
Pytest configuration and shared fixtures.
"""
import pytest
import os
import tempfile
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from typing import Generator

# Set test environment variables before importing app
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only"
os.environ["OPENAI_API_KEY"] = "test-openai-key"
os.environ["GROQ_API_KEY"] = "test-groq-key"

from app.core.database import Base, get_db
from app.main import app
from app.db_models.user import User
from app.db_models.patient import Patient
from app.core.auth import get_password_hash, create_access_token
from app.agents.state import AgentState


# Create test database
@pytest.fixture(scope="function")
def test_db():
    """Create a temporary SQLite database for testing."""
    # Create temporary database file
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Override get_db dependency
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    db_session = TestingSessionLocal()
    
    yield db_session
    
    # Cleanup - close all connections first (Windows file locking fix)
    db_session.close()
    engine.dispose()  # Close all connections
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    os.close(db_fd)
    
    # Try to delete, but don't fail if it's locked (Windows issue)
    try:
        os.unlink(db_path)
    except (PermissionError, OSError):
        # Windows file locking - file will be cleaned up later
        pass


@pytest.fixture
def client(test_db):
    """Create a test client for FastAPI."""
    return TestClient(app)


@pytest.fixture
def test_user(test_db):
    """Create a test user in the database."""
    # Create patient first
    patient = Patient(
        name="Test Patient",
        age=30,
        gender="Male",
        smoker=False
    )
    test_db.add(patient)
    test_db.flush()
    
    # Create user
    user = User(
        email="test@example.com",
        password_hash=get_password_hash("testpassword123"),
        patient_id=patient.id
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    test_db.refresh(patient)
    
    return user


@pytest.fixture
def auth_token(test_user):
    """Generate JWT token for test user."""
    return create_access_token(
        data={
            "user_id": test_user.id,
            "patient_id": test_user.patient_id,
            "email": test_user.email
        }
    )


@pytest.fixture
def auth_headers(auth_token):
    """Return headers with authentication token."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def sample_agent_state():
    """Create a sample agent state for testing."""
    return {
        "patient_id": 1,
        "patient_name": "John Doe",
        "patient_age": 30,
        "patient_gender": "Male",
        "patient_smoker": False,
        "patient_chronic_conditions": None,
        "patient_occupation": None,
        "visit_id": "test-visit-123",
        "symptoms": "Cough and chest pain",
        "symptom_duration": "3 days",
        "patient_weight": 70.0,
        "vitals": None,
        "emergency_flag": False,
        "emergency_reason": None,
        "emergency_checked": False,
        "doctor_note": None,
        "xray_result": None,
        "xray_available": False,
        "spirometry_result": None,
        "spirometry_available": False,
        "cbc_result": None,
        "cbc_available": False,
        "missing_tests": [],
        "tests_collected": [],
        "tests_skipped": [],
        "test_collector_findings": None,
        "rag_context": None,
        "diagnosis": None,
        "treatment_plan": None,
        "tests_recommended": [],
        "home_remedies": [],
        "followup_instruction": None,
        "previous_visits": [],
        "progress_summary": None,
        "patient_history_summary": None,
        "conversation_history": [],
        "current_step": None,
        "message": None,
        "patient_data_confirmed": False,
        "treatment_approved": False,
        "treatment_modifications": None,
        "calculated_dosages": None,
        "error_count": 0,
        "workflow_error": None,
        "next_step": None,
        "test_collection_complete": False,
        "visit_summary": None,
        "history_saved": False
    }


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    from PIL import Image
    import io
    
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes
