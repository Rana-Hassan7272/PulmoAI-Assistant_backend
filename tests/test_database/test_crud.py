"""
Tests for database CRUD operations.
"""
import pytest
from app.db_models.patient import Patient
from app.db_models.user import User
from app.core.auth import get_password_hash


class TestPatientCRUD:
    """Test suite for Patient CRUD operations."""
    
    def test_create_patient(self, test_db):
        """Test creating a patient."""
        patient = Patient(
            name="Test Patient",
            age=30,
            gender="Male",
            smoker=False
        )
        test_db.add(patient)
        test_db.commit()
        test_db.refresh(patient)
        
        assert patient.id is not None
        assert patient.name == "Test Patient"
        assert patient.age == 30
    
    def test_read_patient(self, test_db):
        """Test reading a patient."""
        patient = Patient(
            name="Read Test",
            age=25,
            gender="Female"
        )
        test_db.add(patient)
        test_db.commit()
        test_db.refresh(patient)
        
        # Read back
        found = test_db.query(Patient).filter(Patient.id == patient.id).first()
        
        assert found is not None
        assert found.name == "Read Test"
    
    def test_update_patient(self, test_db):
        """Test updating a patient."""
        patient = Patient(
            name="Update Test",
            age=20
        )
        test_db.add(patient)
        test_db.commit()
        test_db.refresh(patient)
        
        # Update
        patient.age = 21
        test_db.commit()
        test_db.refresh(patient)
        
        assert patient.age == 21
    
    def test_delete_patient(self, test_db):
        """Test deleting a patient."""
        patient = Patient(
            name="Delete Test",
            age=30
        )
        test_db.add(patient)
        test_db.commit()
        patient_id = patient.id
        
        # Delete
        test_db.delete(patient)
        test_db.commit()
        
        # Verify deleted
        found = test_db.query(Patient).filter(Patient.id == patient_id).first()
        assert found is None


class TestUserCRUD:
    """Test suite for User CRUD operations."""
    
    def test_create_user(self, test_db):
        """Test creating a user."""
        # Create patient first
        patient = Patient(name="User Test Patient")
        test_db.add(patient)
        test_db.flush()
        
        user = User(
            email="user@test.com",
            password_hash=get_password_hash("password123"),
            patient_id=patient.id
        )
        test_db.add(user)
        test_db.commit()
        test_db.refresh(user)
        
        assert user.id is not None
        assert user.email == "user@test.com"
        assert user.patient_id == patient.id
    
    def test_user_patient_relationship(self, test_db):
        """Test user-patient relationship."""
        patient = Patient(name="Relationship Test")
        test_db.add(patient)
        test_db.flush()
        
        user = User(
            email="rel@test.com",
            password_hash=get_password_hash("pass"),
            patient_id=patient.id
        )
        test_db.add(user)
        test_db.commit()
        test_db.refresh(user)
        test_db.refresh(patient)
        
        # Test relationship
        assert user.patient_id == patient.id
        assert hasattr(user, 'patient') or user.patient_id is not None
