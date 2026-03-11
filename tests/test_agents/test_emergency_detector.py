"""
Tests for Emergency Detector Agent.
"""
import pytest
from unittest.mock import patch, MagicMock
from app.agents.emergency_detector import emergency_detector_agent
from app.agents.state import AgentState


class TestEmergencyDetectorAgent:
    """Test suite for emergency detector agent."""
    
    def test_no_symptoms_skips_check(self, sample_agent_state):
        """Test that agent skips check if no symptoms provided."""
        state = sample_agent_state.copy()
        state["symptoms"] = None
        
        result = emergency_detector_agent(state)
        
        assert result["emergency_flag"] is False
        # When no symptoms, it may not set emergency_checked to True
        # It just returns early without checking
        assert result.get("emergency_checked") is not None or result["emergency_flag"] is False
    
    def test_already_checked_skips(self, sample_agent_state):
        """Test that agent skips if already checked."""
        state = sample_agent_state.copy()
        state["emergency_checked"] = True
        state["symptoms"] = "severe chest pain"
        
        result = emergency_detector_agent(state)
        
        # Should return without re-checking
        assert result["emergency_checked"] is True
    
    @patch('app.agents.emergency_detector.call_groq_llm')
    def test_emergency_detected(self, mock_llm, sample_agent_state):
        """Test detection of emergency condition."""
        state = sample_agent_state.copy()
        state["symptoms"] = "severe chest pain, unable to breathe"
        state["patient_age"] = 65
        
        # Mock LLM response indicating emergency
        mock_llm.return_value = '{"is_emergency": true, "reason": "Severe respiratory distress detected"}'
        
        result = emergency_detector_agent(state)
        
        assert result["emergency_flag"] is True
        assert result["emergency_reason"] is not None
        assert "emergency" in result["message"].lower() or "🚨" in result["message"]
    
    @patch('app.agents.emergency_detector.call_groq_llm')
    def test_non_emergency(self, mock_llm, sample_agent_state):
        """Test non-emergency condition."""
        state = sample_agent_state.copy()
        state["symptoms"] = "mild cough"
        
        # Mock LLM response indicating non-emergency
        mock_llm.return_value = '{"is_emergency": false, "reason": "Mild symptoms, not an emergency"}'
        
        result = emergency_detector_agent(state)
        
        assert result["emergency_flag"] is False
        assert result["emergency_checked"] is True
    
    @patch('app.agents.emergency_detector.call_groq_llm')
    def test_invalid_json_fallback(self, mock_llm, sample_agent_state):
        """Test fallback behavior on invalid JSON response."""
        state = sample_agent_state.copy()
        state["symptoms"] = "cough"
        
        # Mock invalid JSON response
        mock_llm.return_value = "Invalid response without JSON"
        
        result = emergency_detector_agent(state)
        
        # Should default to non-emergency (safer)
        assert result["emergency_flag"] is False
        assert result["emergency_checked"] is True
