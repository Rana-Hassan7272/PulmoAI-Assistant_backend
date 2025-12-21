"""
Patient Intake Agent
Collects initial patient information and handles returning patients by loading history.
"""
import json
import re
import logging
from typing import Dict, Any, Optional, List
from .state import AgentState
from .config import call_groq_llm
from .tools import fetch_patient_history_tool
from ..core.error_handling import (
    LLMError, LLMInvalidResponseError, format_error_for_user,
    safe_execute, log_error_with_context
)

logger = logging.getLogger(__name__)


def patient_intake_agent(state: AgentState) -> AgentState:
    """
    Patient Intake Agent - collects patient info and loads history for returning patients.
    """
    # Capture the previous step before we update it
    previous_step = state.get("current_step")
    state["current_step"] = "patient_intake"
    
    # 1. LOAD HISTORY FOR RETURNING PATIENTS
    patient_id = state.get("patient_id")
    if patient_id and not state.get("patient_history_summary"):
        history_result = fetch_patient_history_tool(state)
        if history_result["status"] == "success" and history_result["visits"]:
            state["previous_visits"] = history_result["visits"]
            state["patient_history_summary"] = history_result["formatted_history"]

    # 2. GET CONVERSATION CONTEXT
    conversation = state.get("conversation_history", [])
    
    # If no conversation yet, show greeting
    if not conversation:
        state["message"] = _generate_greeting(state)
        state["current_step"] = "patient_intake_waiting_input"
        return state

    # 3. DATA EXTRACTION
    # Only extract if we haven't confirmed yet
    if not state.get("patient_data_confirmed"):
        data = {}  # Initialize data to empty dict
        
        messages = [
            {
                "role": "system",
                "content": """You are a medical intake assistant. Extract the following patient details into a JSON object:
                - name (string)
                - age (integer)
                - gender (string)
                - weight (number in kg)
                - smoker (boolean)
                - symptoms (string)
                - duration (string)
                - history (string, chronic conditions)
                - occupation (string)

                Return ONLY the JSON object. Use null for any missing information."""
            },
            {
                "role": "user",
                "content": f"Extract info from this conversation:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in conversation])
            }
        ]
        
        try:
            response = call_groq_llm(messages, json_mode=True)
            
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response.split("```json")[1].split("```")[0].strip()
            elif clean_response.startswith("```"):
                clean_response = clean_response.split("```")[1].split("```")[0].strip()
                
            data = json.loads(clean_response)
            logger.info(f"Successfully extracted patient data: {list(data.keys())}")
            
        except LLMError as e:
            log_error_with_context(e, {"operation": "patient_data_extraction"})
            # Don't fail completely - try to extract basic info from conversation
            data = _extract_basic_info_fallback(conversation)
            logger.warning("Using fallback data extraction due to LLM error")
            
        except (json.JSONDecodeError, LLMInvalidResponseError) as e:
            log_error_with_context(e, {"operation": "patient_data_extraction"})
            # Try fallback extraction
            data = _extract_basic_info_fallback(conversation)
            logger.warning("Using fallback data extraction due to JSON parse error")
            
        except Exception as e:
            log_error_with_context(e, {"operation": "patient_data_extraction"})
            # Final fallback
            data = _extract_basic_info_fallback(conversation)
            logger.warning("Using fallback data extraction due to unexpected error")
        
        # Update state with extracted data (only if not already set)
        if data.get("name") and not state.get("patient_name"): 
            state["patient_name"] = str(data["name"])
        if data.get("age") is not None and state.get("patient_age") is None: 
            try: 
                state["patient_age"] = int(data["age"])
            except: 
                pass
        if data.get("gender") and not state.get("patient_gender"): 
            state["patient_gender"] = str(data["gender"])
        if data.get("weight") is not None and state.get("patient_weight") is None:
            try: 
                state["patient_weight"] = float(data["weight"])
            except: 
                pass
        if data.get("smoker") is not None and state.get("patient_smoker") is None:
            # Handle string "yes"/"no" or boolean
            smoker_val = data.get("smoker")
            if isinstance(smoker_val, str):
                state["patient_smoker"] = smoker_val.lower() in ["yes", "true", "1"]
            else:
                state["patient_smoker"] = bool(smoker_val)
        if data.get("symptoms") and not state.get("symptoms"): 
            state["symptoms"] = str(data["symptoms"])
        if data.get("duration") and not state.get("symptom_duration"): 
            state["symptom_duration"] = str(data["duration"])
        if data.get("history") and not state.get("patient_chronic_conditions"): 
            state["patient_chronic_conditions"] = str(data["history"])
        if data.get("occupation") and not state.get("patient_occupation"): 
            state["patient_occupation"] = str(data["occupation"])
        
        logger.info(f"State after extraction - name: {state.get('patient_name')}, age: {state.get('patient_age')}, symptoms: {state.get('symptoms')}")

    # 4. WORKFLOW LOGIC
    last_msg = conversation[-1]["content"].lower() if conversation else ""
    
    # Use the previous_step to check if we were waiting for confirmation
    if previous_step == "patient_intake_awaiting_confirmation":
        if any(word in last_msg for word in ["yes", "correct", "confirm", "approve", "yep", "ok", "right"]):
            state["patient_data_confirmed"] = True
            state["current_step"] = "patient_intake_complete"
            logger.info("DEBUG: Confirmation received.")
            return state
        elif any(word in last_msg for word in ["no", "wrong", "change", "incorrect", "not correct"]):
            state["message"] = "I'm sorry. Please tell me what information is incorrect."
            state["current_step"] = "patient_intake_waiting_input"
            return state

    # If already confirmed, move on immediately
    if state.get("patient_data_confirmed"):
        state["current_step"] = "patient_intake_complete"
        return state

    # Check for minimum required data
    has_name = bool(state.get("patient_name"))
    has_age = state.get("patient_age") is not None
    has_symptoms = bool(state.get("symptoms"))

    if has_name and has_age and has_symptoms:
        state["message"] = _generate_confirmation_request(state)
        state["current_step"] = "patient_intake_awaiting_confirmation"
        logger.info("Generated confirmation request - waiting for user confirmation")
    else:
        missing_fields = []
        if not has_name: missing_fields.append("name")
        if not has_age: missing_fields.append("age")
        if not has_symptoms: missing_fields.append("symptoms")
        
        state["message"] = _generate_followup_question(state, missing_fields)
        state["current_step"] = "patient_intake_waiting_input"
        logger.info(f"Missing fields: {missing_fields} - asking for more info")

    return state


def _extract_basic_info_fallback(conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Fallback function to extract basic patient info using simple pattern matching.
    Used when LLM extraction fails.
    """
    import re
    data = {}
    full_text = " ".join([msg.get("content", "") for msg in conversation]).lower()
    
    # Extract name (look for "name is X" or "name: X")
    name_match = re.search(r'name\s*(?:is|:)?\s+([a-z]+)', full_text)
    if name_match:
        data["name"] = name_match.group(1).title()
    
    # Extract age
    age_match = re.search(r'age\s*(?:is|:)?\s+(\d+)', full_text)
    if age_match:
        try:
            data["age"] = int(age_match.group(1))
        except:
            pass
    
    # Extract gender
    if "male" in full_text:
        data["gender"] = "male"
    elif "female" in full_text:
        data["gender"] = "female"
    
    # Extract weight
    weight_match = re.search(r'weight\s*(?:is|:)?\s+(\d+(?:\.\d+)?)\s*(?:kg|lbs)?', full_text)
    if weight_match:
        try:
            data["weight"] = float(weight_match.group(1))
        except:
            pass
    
    # Extract smoker status
    if "smoker" in full_text:
        # Look for "smoker yes" or "smoker: yes" pattern
        smoker_match = re.search(r'smoker\s*(?:is|:)?\s*(yes|no|true|false)', full_text)
        if smoker_match:
            data["smoker"] = smoker_match.group(1).lower() in ["yes", "true"]
        elif "non-smoker" in full_text or "non smoker" in full_text:
            data["smoker"] = False
        else:
            # Default: if "smoker" mentioned, check if "yes" or "no" follows
            smoker_index = full_text.find("smoker")
            next_words = full_text[smoker_index:smoker_index+15]
            data["smoker"] = "yes" in next_words or "true" in next_words
    elif "non-smoker" in full_text or "non smoker" in full_text:
        data["smoker"] = False
    
    # Extract symptoms - try to get full symptom text
    symptom_match = re.search(r'symptoms?\s*(?:is|:)?\s*([^\.]+?)(?:\s+for\s+|\s+medical|\s+occupation|$)', full_text, re.IGNORECASE)
    if symptom_match:
        data["symptoms"] = symptom_match.group(1).strip()
    else:
        # Fallback: look for common symptom keywords
        symptoms = []
        symptom_keywords = ["cough", "fever", "pain", "breath", "chest", "headache", "nausea", "breathing"]
        for keyword in symptom_keywords:
            if keyword in full_text:
                symptoms.append(keyword)
        if symptoms:
            data["symptoms"] = ", ".join(symptoms)
    
    # Extract duration
    duration_match = re.search(r'(?:for|duration|last)\s+(\d+\s*(?:days?|weeks?|months?|hours?))', full_text, re.IGNORECASE)
    if duration_match:
        data["duration"] = duration_match.group(1).strip()
    
    # Extract medical history
    history_match = re.search(r'(?:medical\s+history|history)\s*(?:is|:)?\s*([^\.]+?)(?:\s+occupation|$)', full_text, re.IGNORECASE)
    if history_match:
        data["history"] = history_match.group(1).strip()
    
    # Extract occupation
    occupation_match = re.search(r'occupation\s*(?:is|:)?\s*([a-z]+)', full_text, re.IGNORECASE)
    if occupation_match:
        data["occupation"] = occupation_match.group(1).strip()
    
    return data


def _generate_greeting(state: AgentState) -> str:
    name = state.get("patient_name", "")
    history = state.get("patient_history_summary")
    if history:
        return f"Welcome back! I've reviewed your previous visits:\n{history}\n\nHow are you feeling today? Are you here for a follow-up or a new concern?"
    return """Hello! How are you, sir? I am your pulmonology doctor. 

To provide you with the best care, could you please tell me these details:

• **Name**
• **Age**
• **Gender**
• **Weight** (in kg or lbs)
• **Smoker or not**
• **Symptoms**
• **Medical history**
• **Symptom duration**

You can share all this information in one message, and I'll organize it for you."""


def _generate_confirmation_request(state: AgentState) -> str:
    info = [
        f"**Name:** {state.get('patient_name')}",
        f"**Age:** {state.get('patient_age')}",
        f"**Symptoms:** {state.get('symptoms')}",
        f"**Weight:** {state.get('patient_weight', 'Not provided')}kg",
        f"**Smoker:** {'Yes' if state.get('patient_smoker') else 'No'}"
    ]
    return "Thank you! Please review your information and confirm if it's correct:\n\n" + "\n".join(info) + "\n\nReply **'Yes'** to confirm or tell me what to change."


def _generate_followup_question(state: AgentState, missing: List[str]) -> str:
    if "name" in missing or "age" in missing:
        return "Could you please tell me your name and age?"
    if "symptoms" in missing:
        return f"Thanks {state.get('patient_name')}. What symptoms are you experiencing today and for how long?"
    return "I see. Could you also tell me your weight and if you are a smoker?"
