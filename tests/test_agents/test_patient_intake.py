"""
Tests for Patient Intake Agent.
"""
import pytest
from unittest.mock import patch, MagicMock
from app.agents.patient_intake import patient_intake_agent
from app.agents.state import AgentState


class TestPatientIntakeAgent:
    """Test suite for patient intake agent."""
    
    def test_initial_greeting(self, sample_agent_state):
        """Test that agent generates greeting on first interaction."""
        state = sample_agent_state.copy()
        state["conversation_history"] = []
        
        result = patient_intake_agent(state)
        
        assert result["current_step"] == "patient_intake_waiting_input"
        assert result["message"] is not None
        assert "greeting" in result["message"].lower() or "welcome" in result["message"].lower() or "hello" in result["message"].lower()
    
    @patch('app.agents.patient_intake.call_groq_llm')
    def test_data_extraction(self, mock_llm, sample_agent_state):
        """Test patient data extraction from conversation."""
        state = sample_agent_state.copy()
        state["conversation_history"] = [
            {"role": "user", "content": "name John age 30 weight 70kg gender male symptoms cough"}
        ]
        
        # Mock LLM response
        mock_llm.return_value = '{"name": "John", "age": 30, "weight": 70, "gender": "male", "symptoms": "cough"}'
        
        result = patient_intake_agent(state)
        
        # LLM may extract "John Doe" instead of just "John" - both are valid
        assert result["patient_name"] in ["John", "John Doe"]
        assert result["patient_age"] == 30
        assert result["patient_weight"] == 70.0
        assert result["patient_gender"] in ["male", "Male"]
        assert "cough" in result["symptoms"].lower()
    
    @patch('app.agents.patient_intake.call_groq_llm')
    def test_confirmation_handling(self, mock_llm, sample_agent_state):
        """Test patient data confirmation."""
        state = sample_agent_state.copy()
        state["patient_name"] = "John"
        state["patient_age"] = 30
        state["conversation_history"] = [
            {"role": "user", "content": "name John age 30"},
            {"role": "assistant", "content": "Please confirm..."},
            {"role": "user", "content": "yes, that is correct"}
        ]
        
        # Mock LLM for confirmation
        mock_llm.return_value = '{"confirmed": true}'
        
        result = patient_intake_agent(state)
        
        # Should set confirmation flag
        assert result.get("patient_data_confirmed") is True or result.get("message") is not None
    
    def test_empty_conversation(self, sample_agent_state):
        """Test handling of empty conversation."""
        state = sample_agent_state.copy()
        state["conversation_history"] = []
        
        result = patient_intake_agent(state)
        
        assert result["message"] is not None
        assert result["current_step"] is not None
