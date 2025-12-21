"""
Emergency Detector Agent

Detects life-threatening symptoms and conditions that require immediate medical attention.
"""
import json
import signal
from .state import AgentState
from .config import call_groq_llm


def emergency_detector_agent(state: AgentState) -> AgentState:
    """
    Emergency Detector Agent - checks for life-threatening symptoms.
    
    This agent analyzes patient information to identify emergency conditions
    that require immediate medical attention.
    
    Args:
        state: Current agent state with patient information
        
    Returns:
        Updated state with emergency_flag and emergency_reason set if emergency detected
    """
    print(f"DEBUG: emergency_detector_agent - Starting emergency check")
    
    # If we've already performed an emergency check for this visit, skip re-checking.
    # This avoids re-calling the LLM and makes later turns (like test collection)
    # fast and non-blocking.
    if state.get("emergency_checked", False):
        print("DEBUG: emergency_detector_agent - Emergency already checked, skipping.")
        return state
    state["current_step"] = "emergency_detector"
    
    # Get patient information from state
    symptoms = state.get("symptoms", "")
    patient_age = state.get("patient_age")
    patient_gender = state.get("patient_gender", "")
    patient_smoker = state.get("patient_smoker")
    chronic_conditions = state.get("patient_chronic_conditions", "")
    symptom_duration = state.get("symptom_duration", "")
    
    # If no symptoms collected yet, skip emergency check
    if not symptoms:
        state["emergency_flag"] = False
        state["emergency_reason"] = None
        return state
    
    # Prepare patient information for LLM analysis
    patient_info = f"""Patient Information:
- Age: {patient_age if patient_age else "Unknown"}
- Gender: {patient_gender if patient_gender else "Unknown"}
- Smoker: {"Yes" if patient_smoker else "No" if patient_smoker is False else "Unknown"}
- Symptoms: {symptoms}
- Symptom Duration: {symptom_duration if symptom_duration else "Unknown"}
- Medical History: {chronic_conditions if chronic_conditions else "None"}"""
    
    # Use LLM to analyze for emergency conditions
    messages = [
        {
            "role": "system",
            "content": """You are a pulmonologist emergency triage system. Your task is to identify ONLY life-threatening respiratory conditions that require IMMEDIATE medical attention.

IMPORTANT: Only flag as EMERGENCY if symptoms are SEVERE, UNBEARABLE, or LIFE-THREATENING.

EMERGENCY CONDITIONS (ONLY severe cases):
- SEVERE shortness of breath (unable to speak, gasping for air, severe respiratory distress)
- UNBEARABLE chest pain (severe, crushing, radiating, or associated with other emergency symptoms)
- Hemoptysis (coughing up blood - significant amount)
- SEVERE asthma attack (severe respiratory distress, unable to breathe)
- Signs of respiratory failure (severe hypoxia, confusion, cyanosis)
- Very high fever (>103°F/39.4°C) with SEVERE respiratory symptoms
- Sudden onset SEVERE symptoms (severe chest pain, severe breathing difficulty)
- Signs of pulmonary embolism (sudden severe chest pain, severe dyspnea, hemoptysis)
- SEVERE allergic reaction affecting breathing (severe airway obstruction, anaphylaxis)
- Pneumothorax symptoms (severe chest pain, severe dyspnea)
- SEVERE COPD exacerbation (severe respiratory distress)

NON-EMERGENCY conditions (do NOT flag as emergency):
- Mild to moderate cough
- Mild to moderate shortness of breath
- Mild chest pain or discomfort
- Common cold symptoms
- Mild to moderate fever (<103°F/39.4°C)
- Chronic stable conditions (mild asthma, stable COPD)
- Mild breathing difficulty during exercise
- Mild allergy symptoms
- Stable chronic conditions

CRITICAL: Only return is_emergency: true if the condition is SEVERE, UNBEARABLE, or LIFE-THREATENING. 
Mild or moderate symptoms should return is_emergency: false.

Analyze the patient information and determine if this is a SEVERE EMERGENCY situation requiring immediate medical attention.

Return ONLY a valid JSON object with these exact keys:
{
    "is_emergency": true or false,
    "reason": "Brief explanation of why it is or isn't an emergency (2-3 sentences max)"
}

If is_emergency is true, the reason should clearly state the life-threatening condition detected."""
        },
        {
            "role": "user",
            "content": patient_info
        }
    ]
    
    try:
        import logging
        from ..core.error_handling import LLMError, LLMInvalidResponseError, log_error_with_context
        
        logger = logging.getLogger(__name__)
        logger.info("Calling LLM for emergency check...")
        
        response_text = call_groq_llm(messages, temperature=0.3)  # Lower temperature for more consistent emergency detection
        
        # Parse JSON response
        response_text = response_text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        try:
            emergency_analysis = json.loads(response_text)
        except json.JSONDecodeError as e:
            log_error_with_context(e, {"operation": "emergency_detector", "response_preview": response_text[:100]})
            # Default to non-emergency if parsing fails (safer)
            emergency_analysis = {"is_emergency": False, "reason": "Unable to analyze emergency status. Proceeding with standard care."}
            logger.warning("Failed to parse emergency analysis JSON. Defaulting to non-emergency.")
        
        is_emergency = emergency_analysis.get("is_emergency", False)
        reason = emergency_analysis.get("reason", "")
        
        state["emergency_flag"] = is_emergency
        state["emergency_reason"] = reason if is_emergency else None
        state["emergency_checked"] = True
        
        logger.info(f"Emergency check complete: is_emergency={is_emergency}")
        
        # Update message if emergency detected
        if is_emergency:
            state["message"] = (
                "🚨 EMERGENCY DETECTED 🚨\n\n"
                f"{reason}\n\n"
                "Please seek IMMEDIATE medical attention. "
                "Go to the nearest emergency room or call emergency services immediately. "
                "This system cannot provide emergency care."
            )
        # If not an emergency, keep whatever message was already set by previous agents
        # If no message exists, set a default one to prevent UI from showing "loading"
        if not state.get("message"):
            # Preserve the doctor note or set a default message
            if state.get("doctor_note"):
                state["message"] = state["doctor_note"]
            else:
                state["message"] = "Emergency check complete. Proceeding with diagnostic workflow..."
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, default to non-emergency (safer default)
        print(f"Warning: Failed to parse emergency detection response: {e}")
        import traceback
        print(traceback.format_exc())
        state["emergency_flag"] = False
        state["emergency_reason"] = None
        state["emergency_checked"] = True
        # Keep existing message from previous agents so the user still sees a response
        if not state.get("message"):
            if state.get("doctor_note"):
                state["message"] = state["doctor_note"]
            else:
                state["message"] = "Emergency check complete. Proceeding with diagnostic workflow..."
        # Continue with workflow (safer to proceed than block)
    
    except LLMError as e:
        import logging
        from ..core.error_handling import log_error_with_context
        
        logger = logging.getLogger(__name__)
        log_error_with_context(e, {"operation": "emergency_detector"})
        
        # Default to non-emergency on LLM error (safer - don't block patient care)
        state["emergency_flag"] = False
        state["emergency_reason"] = None
        state["emergency_checked"] = True
        logger.warning("Emergency detector LLM error. Proceeding with standard care.")
        
    except Exception as e:
        import logging
        from ..core.error_handling import log_error_with_context
        
        logger = logging.getLogger(__name__)
        log_error_with_context(e, {"operation": "emergency_detector"})
        
        # Default to non-emergency on any error (safer)
        state["emergency_flag"] = False
        state["emergency_reason"] = None
        state["emergency_checked"] = True
        logger.error("Emergency detector unexpected error. Proceeding with standard care.")
        if not state.get("message"):
            if state.get("doctor_note"):
                state["message"] = state["doctor_note"]
            else:
                state["message"] = "Emergency check complete. Proceeding with diagnostic workflow..."
        # Continue with workflow (safer to proceed than block)
    
    print(f"DEBUG: emergency_detector_agent - Returning state with message: {state.get('message', 'None')[:100] if state.get('message') else 'None'}...")
    return state

