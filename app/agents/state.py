"""
LangGraph State Definition for Pulmonologist Assistant

This defines the state structure that flows through the LangGraph agent pipeline.
"""
from typing import TypedDict, Optional, Dict, List, Any
from typing_extensions import Annotated


class AgentState(TypedDict):
    """
    State structure for the LangGraph agent pipeline.
    
    This state is passed between nodes and accumulates information throughout
    the diagnostic process.
    """
    # Patient Information
    patient_id: Optional[int]
    patient_name: Optional[str]
    patient_age: Optional[int]
    patient_gender: Optional[str]
    patient_smoker: Optional[bool]
    patient_chronic_conditions: Optional[str]
    patient_occupation: Optional[str]
    
    # Visit Information
    visit_id: Optional[str]
    symptoms: Optional[str]
    symptom_duration: Optional[str]
    patient_weight: Optional[float]  # Patient weight in kg (for dosage calculations)
    vitals: Optional[Dict[str, Any]]  # e.g., {"spo2": 95, "bp": "120/80", "temperature": 98.6}
    
    # Emergency Detection
    emergency_flag: bool
    emergency_reason: Optional[str]
    
    # Doctor Note
    doctor_note: Optional[str]
    
    # Test Results (stored as dicts, will be serialized to JSON for DB)
    xray_result: Optional[Dict[str, Any]]
    xray_available: bool
    
    spirometry_result: Optional[Dict[str, Any]]
    spirometry_available: bool
    
    cbc_result: Optional[Dict[str, Any]]
    cbc_available: bool
    
    # Missing Tests
    missing_tests: List[str]
    
    # RAG and Diagnosis
    rag_context: Optional[str]  # Retrieved context from FAISS vector DB
    diagnosis: Optional[str]
    treatment_plan: Optional[List[str]]
    tests_recommended: Optional[List[str]]
    home_remedies: Optional[List[str]]
    followup_instruction: Optional[str]
    
    # History Comparison (for follow-up visits)
    previous_visits: Optional[List[Dict[str, Any]]]
    progress_summary: Optional[str]
    
    # Conversation History (for agent interactions)
    conversation_history: Annotated[List[Dict[str, str]], "append"]  # List of {"role": "user/assistant", "content": "..."}
    
    # Current step/message
    current_step: Optional[str]
    message: Optional[str]  # Current message to return to user
    
    # Human-in-the-loop confirmations
    patient_data_confirmed: bool  # Patient confirmed their data
    treatment_approved: bool  # Patient approved treatment plan
    treatment_modifications: Optional[str]  # Patient's requested modifications
    
    # Dosage information
    calculated_dosages: Optional[Dict[str, Dict[str, Any]]]  # Calculated dosages for each medication
    
    # Error handling and retry tracking
    error_count: int  # Number of consecutive errors (to prevent infinite loops)
    workflow_error: Optional[str]  # Error message that stops the workflow

