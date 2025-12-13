"""
Pydantic schemas for authentication.
"""
from pydantic import BaseModel, EmailStr
from typing import Optional


class UserSignup(BaseModel):
    """Schema for user signup."""
    email: EmailStr
    password: str
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str


class Token(BaseModel):
    """Schema for JWT token response."""
    access_token: str
    token_type: str = "bearer"
    patient_id: Optional[int] = None
    user_id: int


class UserResponse(BaseModel):
    """Schema for user information response."""
    id: int
    email: str
    patient_id: Optional[int] = None
    
    class Config:
        from_attributes = True

