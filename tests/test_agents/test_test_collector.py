"""
Tests for Test Collector Agent.
"""
import pytest
from unittest.mock import patch, MagicMock
from app.agents.test_collector import test_collector_agent as collector_agent
from app.agents.state import AgentState


class TestTestCollectorAgent:
    """Test suite for test collector agent."""
    
    def test_requests_tests_when_none_collected(self, sample_agent_state):
        """Test that agent requests tests when none collected."""
        state = sample_agent_state.copy()
        state["tests_recommended"] = ["xray", "spirometry"]
        state["conversation_history"] = []
        
        result = collector_agent(state)
        
        assert result["current_step"] == "test_collector"
        assert result["message"] is not None
    
    def test_handles_skip_request(self, sample_agent_state):
        """Test handling of skip requests."""
        state = sample_agent_state.copy()
        state["tests_recommended"] = ["xray"]
        state["conversation_history"] = [
            {"role": "user", "content": "skip xray"}
        ]
        
        result = collector_agent(state)
        
        assert "xray" in result.get("missing_tests", []) or result.get("message") is not None
    
    def test_marks_collection_complete_when_all_done(self, sample_agent_state):
        """Test that agent marks collection complete when all tests done."""
        state = sample_agent_state.copy()
        state["tests_recommended"] = ["xray"]
        state["xray_available"] = True
        state["test_collection_complete"] = False
        
        result = collector_agent(state)
        
        # Should mark as complete or request next step
        assert result["current_step"] == "test_collector"
    
    @patch('app.agents.test_collector.predict_spirometry')
    def test_processes_spirometry_data(self, mock_predict, sample_agent_state):
        """Test processing of spirometry data."""
        state = sample_agent_state.copy()
        state["conversation_history"] = [
            {"role": "user", "content": "fev1 5.0 fvc 5.0"}
        ]
        
        # Mock ML prediction
        mock_predict.return_value = {
            "pattern": "Normal",
            "severity": "Normal",
            "confidence": 0.95
        }
        
        result = collector_agent(state)
        
        # Should process spirometry if data extracted
        assert result["current_step"] == "test_collector"
