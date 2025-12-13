"""
Schemas for diagnostic workflow API endpoints.
"""
from pydantic import BaseModel
from typing import Optional, Dict, List, Any


class DiagnosticMessage(BaseModel):
    """Message for diagnostic workflow."""
    message: str
    patient_id: Optional[int] = None
    visit_id: Optional[str] = None


class DiagnosticResponse(BaseModel):
    """Response from diagnostic workflow."""
    message: str
    current_step: Optional[str] = None
    patient_id: Optional[int] = None
    visit_id: Optional[str] = None
    emergency_flag: Optional[bool] = False
    emergency_reason: Optional[str] = None
    state: Optional[Dict[str, Any]] = None  # Full state for debugging

