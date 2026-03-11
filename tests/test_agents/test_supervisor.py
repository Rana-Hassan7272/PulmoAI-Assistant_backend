"""
Tests for Supervisor Agent.
"""
import pytest
from app.agents.supervisor import supervisor_agent, _fallback_routing, check_supervisor_routing
from app.agents.state import AgentState


class TestSupervisorAgent:
    """Test suite for supervisor agent."""
    
    def test_routes_to_emergency_detector_when_not_checked(self, sample_agent_state):
        """Test routing to emergency detector when not checked."""
        state = sample_agent_state.copy()
        state["patient_data_confirmed"] = True
        state["emergency_checked"] = False
        
        result = supervisor_agent(state)
        
        assert result["next_step"] == "emergency_detector"
    
    def test_routes_to_doctor_note_after_emergency_check(self, sample_agent_state):
        """Test routing to doctor note generator after emergency check."""
        state = sample_agent_state.copy()
        state["patient_data_confirmed"] = True
        state["emergency_checked"] = True
        state["doctor_note"] = None
        
        result = supervisor_agent(state)
        
        assert result["next_step"] == "doctor_note_generator"
    
    def test_routes_to_test_collector_after_doctor_note(self, sample_agent_state):
        """Test routing to test collector after doctor note."""
        state = sample_agent_state.copy()
        state["patient_data_confirmed"] = True
        state["emergency_checked"] = True
        state["doctor_note"] = "Clinical assessment"
        state["tests_recommended"] = ["xray", "spirometry"]
        state["test_collection_complete"] = False
        
        result = supervisor_agent(state)
        
        assert result["next_step"] == "test_collector"
    
    def test_routes_to_rag_after_tests_complete(self, sample_agent_state):
        """Test routing to RAG specialist after tests complete."""
        state = sample_agent_state.copy()
        state["patient_data_confirmed"] = True
        state["emergency_checked"] = True
        state["doctor_note"] = "Clinical assessment"
        state["tests_recommended"] = ["xray"]
        state["test_collection_complete"] = True
        state["xray_available"] = True  # Mark test as available
        state["treatment_plan"] = None
        
        result = supervisor_agent(state)
        
        assert result["next_step"] == "rag_treatment_planner"
    
    def test_routes_to_treatment_approval_after_rag(self, sample_agent_state):
        """Test routing to treatment approval after RAG."""
        state = sample_agent_state.copy()
        state["patient_data_confirmed"] = True
        state["emergency_checked"] = True
        state["doctor_note"] = "Clinical assessment"
        state["test_collection_complete"] = True
        state["treatment_plan"] = ["Medication A", "Medication B"]
        state["treatment_approved"] = False
        
        result = supervisor_agent(state)
        
        assert result["next_step"] == "treatment_approval"
    
    def test_routes_to_end_when_complete(self, sample_agent_state):
        """Test routing to END when workflow is complete."""
        state = sample_agent_state.copy()
        state["patient_data_confirmed"] = True
        state["emergency_checked"] = True
        state["doctor_note"] = "Clinical assessment"
        state["test_collection_complete"] = True
        state["treatment_plan"] = ["Medication"]
        state["treatment_approved"] = True
        state["calculated_dosages"] = {"med1": {"dose": "500mg"}}
        state["final_report"] = "Report generated"
        state["history_saved"] = True
        
        result = supervisor_agent(state)
        
        assert result["next_step"] == "end"


class TestFallbackRouting:
    """Test suite for fallback routing logic."""
    
    def test_fallback_emergency_not_checked(self, sample_agent_state):
        """Test fallback routing when emergency not checked."""
        state = sample_agent_state.copy()
        state["emergency_checked"] = False
        
        result = _fallback_routing(state)
        
        assert result == "emergency_detector"
    
    def test_fallback_doctor_note_missing(self, sample_agent_state):
        """Test fallback routing when doctor note missing."""
        state = sample_agent_state.copy()
        state["emergency_checked"] = True
        state["doctor_note"] = None
        
        result = _fallback_routing(state)
        
        assert result == "doctor_note_generator"
    
    def test_fallback_tests_not_complete(self, sample_agent_state):
        """Test fallback routing when tests not complete."""
        state = sample_agent_state.copy()
        state["emergency_checked"] = True
        state["doctor_note"] = "Note"
        state["tests_recommended"] = ["xray"]
        state["test_collection_complete"] = False
        
        result = _fallback_routing(state)
        
        assert result == "test_collector"


class TestSupervisorRouting:
    """Test suite for supervisor routing function."""
    
    def test_check_supervisor_routing_valid_step(self, sample_agent_state):
        """Test routing with valid step."""
        state = sample_agent_state.copy()
        state["next_step"] = "test_collector"
        
        result = check_supervisor_routing(state)
        
        assert result == "test_collector"
    
    def test_check_supervisor_routing_invalid_step(self, sample_agent_state):
        """Test routing with invalid step defaults to end."""
        state = sample_agent_state.copy()
        state["next_step"] = "invalid_step"
        
        result = check_supervisor_routing(state)
        
        assert result == "end"
