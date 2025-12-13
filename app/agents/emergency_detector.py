"""
Emergency Detector Agent

Detects life-threatening symptoms and conditions that require immediate medical attention.
"""
import json
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
            "content": """You are a pulmonologist emergency triage system. Your task is to identify life-threatening respiratory conditions that require IMMEDIATE medical attention.

EMERGENCY CONDITIONS include (but not limited to):
- Severe shortness of breath or difficulty breathing
- Chest pain (especially if severe or radiating)
- Hemoptysis (coughing up blood)
- Severe asthma attack or respiratory distress
- Signs of respiratory failure (low oxygen saturation, confusion)
- High fever with severe respiratory symptoms
- Sudden onset severe symptoms
- Signs of pulmonary embolism
- Severe allergic reaction affecting breathing
- Pneumothorax symptoms
- Severe COPD exacerbation

NON-EMERGENCY conditions:
- Mild cough
- Mild shortness of breath
- Common cold symptoms
- Mild fever
- Chronic stable conditions

Analyze the patient information and determine if this is an EMERGENCY situation.

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
        response_text = call_groq_llm(messages, temperature=0.3)  # Lower temperature for more consistent emergency detection
        
        # Parse JSON response
        response_text = response_text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        emergency_analysis = json.loads(response_text)
        
        is_emergency = emergency_analysis.get("is_emergency", False)
        reason = emergency_analysis.get("reason", "")
        
        state["emergency_flag"] = is_emergency
        state["emergency_reason"] = reason if is_emergency else None
        
        # Update message if emergency detected
        if is_emergency:
            state["message"] = (
                "🚨 EMERGENCY DETECTED 🚨\n\n"
                f"{reason}\n\n"
                "Please seek IMMEDIATE medical attention. "
                "Go to the nearest emergency room or call emergency services immediately. "
                "This system cannot provide emergency care."
            )
        else:
            # No emergency - continue with normal workflow
            state["message"] = None  # No message needed, workflow continues
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, default to non-emergency (safer default)
        state["emergency_flag"] = False
        state["emergency_reason"] = None
        state["message"] = None
        print(f"Warning: Failed to parse emergency detection response: {e}")
        import traceback
        print(traceback.format_exc())
        # Continue with workflow (safer to proceed than block)
    
    except Exception as e:
        # On any error, default to non-emergency (safer default)
        state["emergency_flag"] = False
        state["emergency_reason"] = None
        state["message"] = None
        print(f"Warning: Emergency detection failed: {e}")
        import traceback
        print(traceback.format_exc())
        # Continue with workflow (safer to proceed than block)
    
    return state

