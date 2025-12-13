"""
FastAPI router for authentication (signup, login, me).
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..core.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_user
)
from ..db_models.user import User
from ..db_models.patient import Patient
from ..schemas.auth import UserSignup, UserLogin, Token, UserResponse
from datetime import timedelta

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    """
    Create a new user account and assign a patient_id.
    
    Steps:
    1. Check if email already exists
    2. Create Patient record (if name/age/gender provided)
    3. Create User record with hashed password
    4. Link User to Patient
    5. Generate JWT token
    """
    # Check if email already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create Patient record first (if basic info provided)
    patient = None
    if user_data.name or user_data.age or user_data.gender:
        patient = Patient(
            name=user_data.name,
            age=user_data.age,
            gender=user_data.gender,
            smoker=False,
            chronic_conditions=None,
            occupation=None
        )
        db.add(patient)
        db.flush()  # Get patient.id without committing
    
    # Create User record
    user = User(
        email=user_data.email,
        password_hash=get_password_hash(user_data.password),
        patient_id=patient.id if patient else None
    )
    db.add(user)
    
    # If no patient was created, create a minimal one now
    if not patient:
        patient = Patient(
            name=None,
            age=None,
            gender=None,
            smoker=False,
            chronic_conditions=None,
            occupation=None
        )
        db.add(patient)
        db.flush()
        # Update user with patient_id
        user.patient_id = patient.id
    
    try:
        db.commit()
        db.refresh(user)
        db.refresh(patient)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )
    
    # Generate JWT token
    access_token_expires = timedelta(minutes=60 * 24)  # 24 hours
    access_token = create_access_token(
        data={
            "user_id": user.id,
            "patient_id": user.patient_id,
            "email": user.email
        },
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        patient_id=user.patient_id,
        user_id=user.id
    )


@router.post("/login", response_model=Token)
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate user and return JWT token.
    """
    # Find user by email
    user = db.query(User).filter(User.email == credentials.email).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Verify password
    if not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Generate JWT token
    access_token_expires = timedelta(minutes=60 * 24)  # 24 hours
    access_token = create_access_token(
        data={
            "user_id": user.id,
            "patient_id": user.patient_id,
            "email": user.email
        },
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        patient_id=user.patient_id,
        user_id=user.id
    )


@router.get("/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user information.
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        patient_id=current_user.patient_id
    )

