"""
Patient Intake Agent

Collects patient information and generates initial doctor note.
"""
import json
import re
from .state import AgentState
from .config import call_groq_llm


def extract_json_from_text(text: str) -> dict:
    """
    Extract JSON object from text that may contain extra content.
    
    Handles:
    - Markdown code blocks (```json ... ```)
    - Plain JSON with extra text before/after
    - Multiple JSON objects (returns first valid one)
    """
    text = text.strip()
    
    # Try to extract from markdown code blocks first
    if "```json" in text:
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass
    
    if "```" in text:
        json_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass
    
    # Try to find JSON object in the text (look for { ... })
    # Find the first { and try to parse from there
    brace_start = text.find('{')
    if brace_start != -1:
        # Try to find the matching closing brace
        brace_count = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found complete JSON object
                    json_str = text[brace_start:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
                    break
    
    # If all else fails, try parsing the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If that fails, try to find and parse just the first line that looks like JSON
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
    
    # If nothing works, raise an error
    raise json.JSONDecodeError("Could not extract valid JSON from response", text, 0)


def patient_intake_agent(state: AgentState) -> AgentState:
    """
    Patient Intake Agent - collects basic patient information.
    
    This agent:
    1. Extracts patient information from conversation history
    2. Generates a structured patient profile
    3. Creates a 2-3 line doctor note summary
    """
    # Check for existing errors first - exit immediately if already in error state
    if state.get("workflow_error") or state.get("current_step") == "error":
        return state
    
    # Check error count at the very start
    error_count = state.get("error_count", 0)
    if error_count >= 3:
        state["workflow_error"] = "Unable to process patient information due to repeated errors. Please check your API key configuration."
        state["message"] = "I'm experiencing technical difficulties. Please contact support or try again later."
        state["current_step"] = "error"
        return state
    
    # Get conversation history
    conversation = state.get("conversation_history", [])
    
    # Check if we're waiting for confirmation
    if state.get("current_step") == "patient_intake_awaiting_confirmation":
        # Check user's response for confirmation
        last_message = conversation[-1].get("content", "").lower() if conversation else ""
        
        if any(word in last_message for word in ["yes", "confirm", "correct", "right", "ok", "okay", "proceed"]):
            # Patient confirmed, proceed
            state["patient_data_confirmed"] = True
            doctor_note = generate_doctor_note(state)
            state["doctor_note"] = doctor_note
            state["message"] = f"Thank you for confirming! Here's the initial assessment:\n\n{doctor_note}"
            state["current_step"] = "patient_intake_complete"
            return state
        elif any(word in last_message for word in ["no", "wrong", "incorrect", "change", "modify", "update"]):
            # Patient wants to correct, ask what to change
            state["message"] = "I understand. Please tell me what information needs to be corrected, and I'll update it."
            state["current_step"] = "patient_intake"
            # Reset confirmation flag so we re-extract
            state["patient_data_confirmed"] = False
            return state
        else:
            # Unclear response, ask again
            confirmation_summary = format_patient_confirmation(state)
            state["message"] = (
                "I need your confirmation. Please review:\n\n"
                f"{confirmation_summary}\n\n"
                "Reply 'Yes' if correct, or tell me what needs to be changed."
            )
            state["current_step"] = "patient_intake_awaiting_confirmation"
            return state
    
    # If no conversation yet, ask initial questions
    if not conversation or len(conversation) == 0:
        state["message"] = (
            "Welcome! I'm your Pulmonologist Assistant. Let me collect some information.\n\n"
            "Please provide:\n"
            "- Your name, age, and gender\n"
            "- Your weight (in kg or lbs)\n"
            "- Are you a smoker? (Yes/No)\n"
            "- Your current symptoms (e.g., cough, fever, chest pain, breathing difficulty)\n"
            "- How long have you had these symptoms?\n"
            "- Any medical history (asthma, TB, allergies, diabetes, etc.)\n"
            "- Your occupation"
        )
        # Set step to indicate we've shown the greeting and are waiting for user input
        # This should route to END (workflow pauses until user sends message via /chat)
        state["current_step"] = "patient_intake_waiting_input"
        # Reset error count when starting fresh
        state["error_count"] = 0
        return state
    
    # Extract information from conversation using LLM
    # Format conversation history as text
    conv_text = "\n".join([
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
        for msg in conversation[-10:]  # Last 10 messages for context
    ])
    
    messages = [
        {
            "role": "system",
            "content": """You are a medical assistant collecting patient intake information.
Extract the following information from the conversation:
- Patient name
- Age (must be an integer, e.g., if user says "21 age" or "age 21", extract 21)
- Gender (Male, Female, or null if not mentioned)
- Weight (in kg or lbs - convert to kg if in lbs. If user says "65 weight" or "weight 65", extract 65 and assume kg unless specified as lbs)
- Smoker status (true if user says "smoker" or "yes", false if "non-smoker" or "no", null if not mentioned)
- Symptoms (list all symptoms mentioned)
- Symptom duration (how long symptoms have been present)
- Medical history (asthma, TB, allergies, diabetes, etc.)
- Occupation

IMPORTANT EXTRACTION RULES:
- If user says "21 age 65 weight", extract age=21 and weight=65 (assume kg unless specified as lbs)
- If user says "age 65" or "65 years old", extract age=65
- If user says "65 kg" or "65kg", extract weight=65
- If user says "65 lbs" or "65 pounds", convert to kg: 65 * 0.453592 = 29.48 kg
- For gender: Look for "male", "female", "man", "woman", "M", "F" - if not found, use null (not the string "null")
- For smoker: Look for keywords like "smoker", "smoking", "yes" (in context of smoking) = true; "non-smoker", "no" = false

If any information is missing, use null (not the string "null").
For weight: If provided in lbs, convert to kg (1 lb = 0.453592 kg). If not provided, use null.

Return ONLY a valid JSON object with these exact keys (use actual null, not string "null"):
{
    "name": "string or null",
    "age": integer or null,
    "gender": "string or null",
    "weight": float or null (in kg),
    "smoker": boolean or null,
    "symptoms": "string or null",
    "symptom_duration": "string or null",
    "chronic_conditions": "string or null",
    "occupation": "string or null"
}"""
        },
        {
            "role": "user",
            "content": f"Conversation history:\n{conv_text}"
        }
    ]
    
    # Check retry counter to prevent infinite loops
    error_count = state.get("error_count", 0)
    if error_count >= 3:
        state["workflow_error"] = "Unable to process patient information due to repeated errors. Please check your API key configuration."
        state["message"] = "I'm experiencing technical difficulties. Please contact support or try again later."
        state["current_step"] = "error"
        return state
    
    try:
        response_text = call_groq_llm(messages)
        # Reset error count on success
        state["error_count"] = 0
    except Exception as llm_error:
        # LLM call failed - increment error count
        error_count = state.get("error_count", 0) + 1
        state["error_count"] = error_count
        
        print(f"Warning: LLM call failed in patient_intake_agent (attempt {error_count}/3): {llm_error}")
        import traceback
        print(traceback.format_exc())
        
        if error_count >= 3:
            state["workflow_error"] = f"LLM service unavailable: {str(llm_error)}"
            state["message"] = "I'm experiencing technical difficulties connecting to the AI service. Please check your API key configuration and try again later."
            state["current_step"] = "error"
            return state
        else:
            state["message"] = f"I'm having trouble processing your information (attempt {error_count}/3). Please try again or check your API key configuration."
            # Don't set current_step back to patient_intake - let routing decide
            # This prevents infinite loops
            state["current_step"] = "patient_intake_retry"
            return state
    
    # Parse JSON response
    try:
        # Extract JSON from response (handles markdown, extra text, etc.)
        patient_data = extract_json_from_text(response_text)
        
        # Update state with extracted patient information
        # Normalize null values (convert string "null" to actual None)
        state["patient_name"] = patient_data.get("name") if patient_data.get("name") not in [None, "null", "None"] else None
        state["patient_age"] = patient_data.get("age") if patient_data.get("age") not in [None, "null", "None"] else None
        state["patient_gender"] = patient_data.get("gender") if patient_data.get("gender") not in [None, "null", "None"] else None
        state["patient_weight"] = patient_data.get("weight") if patient_data.get("weight") not in [None, "null", "None"] else None  # Weight in kg
        state["patient_smoker"] = patient_data.get("smoker") if patient_data.get("smoker") not in [None, "null", "None"] else None
        state["symptoms"] = patient_data.get("symptoms") if patient_data.get("symptoms") not in [None, "null", "None"] else None
        state["symptom_duration"] = patient_data.get("symptom_duration") if patient_data.get("symptom_duration") not in [None, "null", "None"] else None
        state["patient_chronic_conditions"] = patient_data.get("chronic_conditions") if patient_data.get("chronic_conditions") not in [None, "null", "None"] else None
        state["patient_occupation"] = patient_data.get("occupation") if patient_data.get("occupation") not in [None, "null", "None"] else None
        
        # Check if all required fields are collected
        required_fields = ["name", "age", "gender", "symptoms"]
        missing_fields = [field for field in required_fields if not patient_data.get(field)]
        
        if missing_fields:
            state["message"] = f"I still need some information: {', '.join(missing_fields)}. Please provide these details."
            state["current_step"] = "patient_intake"
            return state
        
        # Check if patient has already confirmed
        if state.get("patient_data_confirmed", False):
            # Patient already confirmed, generate doctor note and proceed
            doctor_note = generate_doctor_note(state)
            state["doctor_note"] = doctor_note
            state["message"] = f"Thank you! I've collected your information. Here's the initial assessment:\n\n{doctor_note}"
            state["current_step"] = "patient_intake_complete"
            return state
        
        # Show confirmation summary and wait for patient approval
        confirmation_summary = format_patient_confirmation(state)
        state["message"] = (
            "Please review your information and confirm if it's correct:\n\n"
            f"{confirmation_summary}\n\n"
            "Please reply:\n"
            "- 'Yes' or 'Confirm' if this information is correct\n"
            "- 'No' or tell me what needs to be changed"
        )
        state["current_step"] = "patient_intake_awaiting_confirmation"
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, ask for clarification
        print(f"Warning: Failed to parse patient intake JSON: {e}")
        print(f"Response text was: {response_text[:500]}...")  # Log first 500 chars for debugging
        import traceback
        print(traceback.format_exc())
        error_count = state.get("error_count", 0) + 1
        state["error_count"] = error_count
        if error_count >= 3:
            state["workflow_error"] = "Unable to parse patient information after multiple attempts."
            state["message"] = "I'm experiencing technical difficulties. Please contact support."
            state["current_step"] = "error"
        else:
            state["message"] = "I'm having trouble understanding the information you provided. Could you please provide your name, age, gender, and symptoms clearly?"
            # Don't set current_step back to patient_intake to prevent loops
            state["current_step"] = "patient_intake_retry"
    except Exception as e:
        # Handle any other errors
        print(f"Warning: Patient intake failed: {e}")
        import traceback
        print(traceback.format_exc())
        error_count = state.get("error_count", 0) + 1
        state["error_count"] = error_count
        if error_count >= 3:
            state["workflow_error"] = f"Patient intake failed: {str(e)}"
            state["message"] = "I'm experiencing technical difficulties. Please contact support."
            state["current_step"] = "error"
        else:
            state["message"] = "I encountered an issue processing your information. Please try providing your details again: name, age, gender, and symptoms."
            # Don't set current_step back to patient_intake - let routing decide
            state["current_step"] = "patient_intake_retry"
    
    return state


def format_patient_confirmation(state: AgentState) -> str:
    """
    Format patient information for confirmation.
    
    Args:
        state: Current agent state with patient information
        
    Returns:
        Formatted confirmation text
    """
    weight = state.get('patient_weight')
    weight_str = f"{weight}kg" if weight else "Not provided"
    
    return f"""Name: {state.get('patient_name', 'Not provided')}
Age: {state.get('patient_age', 'Not provided')}
Gender: {state.get('patient_gender', 'Not provided')}
Weight: {weight_str}
Smoker: {"Yes" if state.get('patient_smoker') else "No" if state.get('patient_smoker') is False else "Not provided"}
Symptoms: {state.get('symptoms', 'Not provided')}
Symptom Duration: {state.get('symptom_duration', 'Not provided')}
Medical History: {state.get('patient_chronic_conditions', 'None')}
Occupation: {state.get('patient_occupation', 'Not provided')}"""


def generate_doctor_note(state: AgentState) -> str:
    """
    Generate a 2-3 line doctor note summary from patient information.
    
    Args:
        state: Current agent state with patient information
        
    Returns:
        Doctor note string
    """
    messages = [
        {
            "role": "system",
            "content": """You are a pulmonologist writing a brief clinical note.
Create a concise 2-3 sentence summary in medical language.
Include: age, gender, smoking status, symptoms, duration, and relevant medical history.
Example format: "46-year-old male smoker with 3-day history of cough, fever, and shortness of breath. 
No significant past medical history. Possible lower respiratory infection."""
        },
        {
            "role": "user",
            "content": f"""Patient Information:
- Age: {state.get("patient_age", "Unknown")}
- Gender: {state.get("patient_gender", "Unknown")}
- Smoker: {"Yes" if state.get("patient_smoker") else "No" if state.get("patient_smoker") is False else "Unknown"}
- Symptoms: {state.get("symptoms", "Not specified")}
- Duration: {state.get("symptom_duration", "Not specified")}
- Medical History: {state.get("patient_chronic_conditions", "None")}
- Occupation: {state.get("patient_occupation", "Not specified")}"""
        }
    ]
    
    try:
        response_text = call_groq_llm(messages)
        return response_text.strip()
    except Exception as e:
        # Fallback doctor note if LLM fails
        print(f"Warning: Failed to generate doctor note with LLM: {e}")
        age = state.get("patient_age", "Unknown age")
        gender = state.get("patient_gender", "patient")
        smoker = "smoker" if state.get("patient_smoker") else "non-smoker"
        symptoms = state.get("symptoms", "unspecified symptoms")
        return f"{age}-year-old {gender} {smoker} presenting with {symptoms}. Clinical assessment pending."

