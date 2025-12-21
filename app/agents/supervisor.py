"""
Supervisor Agent - Master Orchestrator

This agent decides which agent/tool to call next based on the current workflow state.
Uses reactive routing to determine the next step in the diagnostic process.
"""
from typing import Literal
from .state import AgentState
from .config import call_groq_llm


def supervisor_agent(state: AgentState) -> AgentState:
    """
    Supervisor Agent - decides which agent/tool to call next.
    
    This is the master orchestrator that routes the workflow based on current state.
    Uses LLM to make intelligent routing decisions.
    
    Returns:
        Updated state with next_step field indicating which agent to call
    """
    state["current_step"] = "supervisor"
    
    # Get current workflow state
    patient_confirmed = state.get("patient_data_confirmed", False)
    emergency_detected = state.get("emergency_flag", False)
    doctor_note_ready = state.get("doctor_note") is not None
    tests_recommended = state.get("tests_recommended", [])
    
    # Check test availability
    xray_available = state.get("xray_available", False)
    spirometry_available = state.get("spirometry_available", False)
    cbc_available = state.get("cbc_available", False)
    
    # Check if tests are complete
    tests_complete = state.get("test_collection_complete", False)
    # Double check manually
    if tests_recommended:
        missing_tests = state.get("missing_tests", [])
        really_done = True
        for t in tests_recommended:
            if t == "xray" and not xray_available and "xray" not in missing_tests: really_done = False
            if t == "spirometry" and not spirometry_available and "spirometry" not in missing_tests: really_done = False
            if t == "cbc" and not cbc_available and "cbc" not in missing_tests: really_done = False
        
        if not really_done:
            tests_complete = False
            state["test_collection_complete"] = False
        elif tests_complete:
            # If manually verified and already True, keep it
            pass
        else:
            # If manually verified and False, update it
            tests_complete = True
            state["test_collection_complete"] = True
    
    # Check if treatment plan exists
    treatment_plan_ready = state.get("treatment_plan") is not None and len(state.get("treatment_plan", [])) > 0
    treatment_approved = state.get("treatment_approved", False)
    dosage_calculated = state.get("calculated_dosages") is not None
    final_report_ready = state.get("final_report") is not None
    
    # Check emergency checked status
    emergency_checked = state.get("emergency_checked", False)
    
    # Check history saved status
    history_saved = state.get("history_saved", False)
    
    # Build context for LLM decision
    workflow_context = f"""
Current Workflow State:
- Patient confirmed: {patient_confirmed}
- Emergency detected: {emergency_detected}
- Emergency checked: {emergency_checked} (if True, DO NOT route to emergency_detector)
- Doctor note ready: {doctor_note_ready}
- Tests recommended: {tests_recommended}
- Tests complete: {tests_complete}
  - X-ray available: {xray_available}
  - Spirometry available: {spirometry_available}
  - CBC available: {cbc_available}
- Treatment plan ready: {treatment_plan_ready}
- Treatment approved: {treatment_approved}
- Dosage calculated: {dosage_calculated}
- Final report ready: {final_report_ready}
- History saved: {history_saved} (if True, DO NOT route to history_saver - workflow is complete)
- Current step: {state.get('current_step', 'unknown')}
"""
    
    # Define available agents/tools
    available_agents = """
Available Agents/Tools:
1. emergency_detector - Check if patient has emergency symptoms
2. doctor_note_generator - Generate clinical assessment and recommend tests
3. test_collector - Collect diagnostic tests sequentially (X-ray, CBC, Spirometry)
4. rag_treatment_planner - Generate treatment plan using RAG
5. dosage_calculator - Calculate medication dosages
6. report_generator - Generate final comprehensive report
7. history_saver - Save visit to database
8. END - Complete the workflow
"""
    
    # Use LLM to decide next step
    messages = [
        {
            "role": "system",
            "content": """You are a Supervisor Agent that orchestrates a medical diagnostic workflow.
Your job is to decide which agent/tool should be called next based on the current workflow state.

Workflow Order (STRICT SEQUENCE - MUST FOLLOW IN ORDER):
1. Patient intake (already done if patient_confirmed=True)
2. Emergency detector (ONLY if emergency_checked=False - should only run once per visit)
3. Doctor note generator (MUST run first after emergency check - generates clinical note and recommends tests)
4. Test collector (MUST run after doctor note - collects X-ray, CBC, Spirometry sequentially)
5. RAG treatment planner (ONLY after tests are complete - generates treatment plan)
6. Treatment approval (wait for user to approve)
7. Dosage calculator (after treatment approved - calculates medication dosages)
8. Report generator (after dosages calculated - generates final comprehensive report)
9. History saver (ONLY if final_report exists AND history_saved=False - saves visit to database)
10. Followup agent (if history saved - compares with previous visits)
11. END (if everything complete)

CRITICAL RULES (MUST FOLLOW):
- NEVER skip doctor_note_generator - it MUST run first after emergency check
- NEVER skip test_collector - it MUST run after doctor note (if tests recommended)
- NEVER route to rag_treatment_planner before tests are complete
- Never route to emergency_detector if emergency_checked=True
- Never route to history_saver if history_saved=True (already saved)
- Never route to history_saver if final_report is missing (must have final report before saving)
- Follow the sequence strictly: doctor_note → test_collector → rag_treatment_planner → etc.

Return ONLY the agent name (e.g., "test_collector", "rag_treatment_planner", "END") as your response.
Do not include any explanation or additional text."""
        },
        {
            "role": "user",
            "content": f"{workflow_context}\n\n{available_agents}\n\nBased on the current state, which agent should be called next? Return only the agent name."
        }
    ]
    
    # CRITICAL: Use rule-based fallback routing FIRST to enforce strict sequence
    # This ensures we NEVER skip doctor_note_generator or test_collector
    # LLM routing can be unreliable and skip steps, so we prioritize rule-based routing
    try:
        found_agent = _fallback_routing(state)
        print(f"DEBUG: Supervisor - Using fallback routing (strict sequence): {found_agent}")
        
        # Additional validation to double-check (shouldn't be needed, but safety net)
        if found_agent == "rag_treatment_planner":
            if not doctor_note_ready:
                print("DEBUG: Supervisor - VALIDATION FAILED: Doctor note missing, forcing doctor_note_generator")
                found_agent = "doctor_note_generator"
            elif tests_recommended and not tests_complete and not state.get("test_collection_complete", False):
                print("DEBUG: Supervisor - VALIDATION FAILED: Tests not complete, forcing test_collector")
                found_agent = "test_collector"
        
        if found_agent == "history_saver":
            if history_saved:
                print("DEBUG: Supervisor - VALIDATION FAILED: History already saved, forcing END")
                found_agent = "end"
            elif not final_report_ready:
                print("DEBUG: Supervisor - VALIDATION FAILED: No final report yet, forcing report_generator")
                found_agent = "report_generator"
        
        if found_agent == "emergency_detector" and emergency_checked:
            print("DEBUG: Supervisor - VALIDATION FAILED: Emergency already checked, forcing doctor_note_generator")
            found_agent = "doctor_note_generator"
        
        # Set next step in state
        state["next_step"] = found_agent
        # Don't set generic message - let the next agent set its own medical instruction
        
    except Exception as e:
        print(f"Warning: Supervisor routing failed: {e}")
        import traceback
        print(traceback.format_exc())
        # Fallback to rule-based routing
        state["next_step"] = _fallback_routing(state)
        # Don't overwrite existing message
        if not state.get("message"):
            state["message"] = f"Routing to {state['next_step']}..."
    
    return state


def _fallback_routing(state: AgentState) -> str:
    """
    Fallback rule-based routing to enforce strict linear workflow.
    
    Returns:
        Next agent name to call
    """
    # STEP 1: Emergency detector - MUST run once after patient confirmation
    # If emergency is detected, we go to history_saver then END
    if not state.get("emergency_checked", False):
        print("DEBUG: _fallback_routing - Routing to emergency_detector (not checked yet)")
        return "emergency_detector"
    
    # If emergency was detected, we should NOT proceed to clinical notes
    if state.get("emergency_flag", False):
        if not state.get("history_saved", False):
            print("DEBUG: _fallback_routing - Emergency detected, routing to history_saver")
            return "history_saver"
        else:
            print("DEBUG: _fallback_routing - Emergency detected and saved, routing to END")
            return "end"

    # STEP 2: Doctor note generator - MUST run after emergency check
    # This agent generates clinical assessment + recommends tests
    if state.get("doctor_note") is None:
        print("DEBUG: _fallback_routing - Routing to doctor_note_generator (doctor note missing)")
        return "doctor_note_generator"
    
    # STEP 3: Test collector - MUST run after doctor note
    # Collects X-ray, CBC, Spirometry sequentially (user can provide or skip)
    tests_recommended = state.get("tests_recommended", [])
    test_collection_complete = state.get("test_collection_complete", False)
    
    print(f"DEBUG: _fallback_routing - Checking tests: recommended={tests_recommended}, complete={test_collection_complete}")
    
    # If tests are recommended, we MUST complete collection or skip them all
    if tests_recommended and len(tests_recommended) > 0 and not test_collection_complete:
        print(f"DEBUG: _fallback_routing - Tests recommended and NOT complete, routing to test_collector")
        return "test_collector"
    elif tests_recommended and len(tests_recommended) > 0 and test_collection_complete:
        print(f"DEBUG: _fallback_routing - Tests recommended but ALREADY complete, skipping collector")
    else:
        print(f"DEBUG: _fallback_routing - No tests recommended, skipping collector")
    
    # STEP 4: RAG treatment planner - ONLY after tests are complete
    # Generates diagnosis and treatment plan using both doctor_note and test_collector_findings
    if not state.get("treatment_plan"):
        # CRITICAL: Do NOT proceed to RAG if tests are recommended but not complete
        if tests_recommended and len(tests_recommended) > 0 and not test_collection_complete:
            print(f"DEBUG: _fallback_routing - BLOCKING RAG: Tests recommended {tests_recommended} but not complete. Routing back to collector.")
            return "test_collector"
            
        print("DEBUG: _fallback_routing - All tests handled, routing to rag_treatment_planner")
        return "rag_treatment_planner"
    
    # STEP 5: Treatment approval - check if user approved the RAG plan
    if state.get("treatment_plan") and not state.get("treatment_approved"):
        print("DEBUG: _fallback_routing - Routing to treatment_approval")
        return "treatment_approval"
    
    # STEP 6: Dosage calculator - ONLY after treatment is approved
    if state.get("treatment_approved") and not state.get("calculated_dosages"):
        print("DEBUG: _fallback_routing - Routing to dosage_calculator (dosages not calculated)")
        return "dosage_calculator"
    
    # STEP 7: Report generator - after dosages are calculated
    if not state.get("final_report"):
        print("DEBUG: _fallback_routing - Routing to report_generator (no final report)")
        return "report_generator"
    
    # STEP 8: History saver - ONLY at the very end to save visit
    if not state.get("history_saved", False):
        print("DEBUG: _fallback_routing - Routing to history_saver (saving final visit)")
        return "history_saver"
    
    # STEP 9: Followup agent - optional progress comparison after save
    if not state.get("progress_summary") and state.get("previous_visits"):
        print("DEBUG: _fallback_routing - Routing to followup_agent")
        return "followup_agent"
    
    # STEP 10: END - everything complete
    print("DEBUG: _fallback_routing - Routing to END (workflow complete)")
    return "end"


def check_supervisor_routing(state: AgentState) -> Literal[
    "emergency_detector",
    "doctor_note_generator",
    "test_collector",
    "rag_treatment_planner",
    "treatment_approval",
    "dosage_calculator",
    "report_generator",
    "history_saver",
    "end"
]:
    """
    Routing function for supervisor conditional edges.
    
    Returns:
        Next agent to call based on supervisor decision
    """
    next_step = state.get("next_step", "end")
    
    # Map to valid routing options
    valid_steps = {
        "emergency_detector": "emergency_detector",
        "doctor_note_generator": "doctor_note_generator",
        "test_collector": "test_collector",
        "rag_treatment_planner": "rag_treatment_planner",
        "treatment_approval": "treatment_approval",
        "dosage_calculator": "dosage_calculator",
        "report_generator": "report_generator",
        "history_saver": "history_saver",
        "end": "end"
    }
    
    return valid_steps.get(next_step.lower(), "end")

