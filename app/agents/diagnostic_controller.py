"""
Diagnostic Controller Agent

Orchestrates ML test API calls (X-ray, Spirometry, CBC) based on symptoms and available data.
"""
import json
import io
from typing import Dict, Any, Optional
from PIL import Image
from .state import AgentState
from .config import call_groq_llm

# Import ML prediction functions
from ..ml_models.xray.preprocessor import predict_xray, predict_xray_proba
from ..ml_models.spirometry.featurizer import predict_spirometry, predict_spirometry_proba
from ..ml_models.bloodcount_report.feature import predict_blood_disease, predict_blood_disease_proba


def diagnostic_controller_agent(state: AgentState) -> AgentState:
    """
    Diagnostic Controller Agent - orchestrates test API calls.
    
    This agent:
    1. Analyzes symptoms to recommend X-ray and/or Spirometry
    2. Always recommends CBC as a general overall body test
    3. Extracts test data from conversation if provided
    4. Calls ML prediction functions for available tests
    5. Stores results in state
    """
    state["current_step"] = "diagnostic_controller"
    
    # Get patient information
    symptoms = state.get("symptoms", "")
    patient_age = state.get("patient_age")
    patient_gender = state.get("patient_gender", "")
    conversation = state.get("conversation_history", [])
    
    # Initialize test results
    state["xray_available"] = False
    state["spirometry_available"] = False
    state["cbc_available"] = False
    state["xray_result"] = None
    state["spirometry_result"] = None
    state["cbc_result"] = None
    state["missing_tests"] = []
    
    # Step 1: Use LLM to recommend tests based on symptoms
    try:
        recommended_tests = recommend_tests(symptoms, patient_age, patient_gender)
    except Exception as e:
        print(f"Warning: Failed to recommend tests: {e}")
        import traceback
        print(traceback.format_exc())
        # Default to recommending all tests if recommendation fails
        recommended_tests = ["xray", "spirometry", "cbc"]
    
    # Step 2: Check if user has indicated which tests they want to provide
    conversation_text = " ".join([msg.get("content", "") for msg in conversation[-5:]])
    conversation_lower = conversation_text.lower()
    
    # Detect user's test selection
    user_wants_xray = any(phrase in conversation_lower for phrase in [
        "xray", "x-ray", "chest xray", "chest x-ray", "i have xray", "i have x-ray",
        "both", "all tests", "all of them"
    ])
    user_wants_cbc = any(phrase in conversation_lower for phrase in [
        "cbc", "blood count", "blood test", "complete blood count", "i have cbc",
        "both", "all tests", "all of them"
    ])
    user_wants_spirometry = any(phrase in conversation_lower for phrase in [
        "spirometry", "spirometry test", "lung function", "i have spirometry",
        "both", "all tests", "all of them"
    ])
    
    # Step 3: Extract test data from conversation
    extracted_data = extract_test_data_from_conversation(conversation, recommended_tests)
    
    # Step 4: Process X-ray if available
    # Check if X-ray result already exists in state (from file upload)
    if state.get("xray_available") and state.get("xray_result"):
        # X-ray already processed in chat endpoint
        pass  # Already set in state
    else:
        # Process X-ray if user wants it and it's recommended
        if ("xray" in recommended_tests and user_wants_xray) or extracted_data.get("xray_image"):
            if extracted_data.get("xray_image"):
                try:
                    xray_result = process_xray_test(extracted_data.get("xray_image"))
                    if xray_result:
                        state["xray_available"] = True
                        state["xray_result"] = xray_result
                except Exception as e:
                    print(f"Warning: Failed to process X-ray test: {e}")
                    import traceback
                    print(traceback.format_exc())
                    state["message"] = "I encountered an issue processing your X-ray image. Please try uploading the image again or proceed without X-ray analysis."
                    return state
            elif user_wants_xray:
                # User wants to provide X-ray but hasn't uploaded yet
                state["message"] = "Please upload your X-ray image file. You can upload it using the file upload button."
                return state
        elif "xray" in recommended_tests and not user_wants_xray:
            # X-ray recommended but user didn't select it
            state["missing_tests"].append("xray")
    
    # Step 5: Process Spirometry if available
    spirometry_missing_values = []
    if ("spirometry" in recommended_tests and user_wants_spirometry) or extracted_data.get("spirometry_data"):
        spirometry_data = extracted_data.get("spirometry_data", {})
        
        # Check if user provided any spirometry data
        has_any_spirometry_data = any(v is not None for v in spirometry_data.values())
        
        if has_any_spirometry_data:
            # Check for missing critical values (FEV1 and FVC are required)
            if not spirometry_data.get("fev1"):
                spirometry_missing_values.append("FEV1")
            if not spirometry_data.get("fvc"):
                spirometry_missing_values.append("FVC")
            
            if spirometry_missing_values:
                # Ask for missing critical values
                state["message"] = f"For Spirometry test, I need the following critical values: {', '.join(spirometry_missing_values)}. Please provide them, or I'll use healthy default values if you skip."
                return state
            else:
                # Process spirometry with available data (missing non-critical values will use defaults)
                try:
                    spirometry_result = process_spirometry_test(
                        spirometry_data,
                        patient_age,
                        patient_gender
                    )
                    if spirometry_result:
                        state["spirometry_available"] = True
                        state["spirometry_result"] = spirometry_result
                except Exception as e:
                    print(f"Warning: Failed to process spirometry test: {e}")
                    import traceback
                    print(traceback.format_exc())
                    state["message"] = "I encountered an issue processing your Spirometry test results. Please verify the values and try again, or proceed without Spirometry."
                    return state
        elif "spirometry" in recommended_tests and user_wants_spirometry:
            # Spirometry recommended but no data provided
            state["missing_tests"].append("spirometry")
    elif "spirometry" in recommended_tests and not user_wants_spirometry:
        state["missing_tests"].append("spirometry")
    
    # Step 6: Process CBC (always recommended as general test)
    # Process CBC if user wants it (or if data provided)
    if (user_wants_cbc or extracted_data.get("cbc_data")):
        cbc_data = extracted_data.get("cbc_data", {})
        
        # Check if user provided any CBC data
        provided_count = sum(1 for v in cbc_data.values() if v is not None)
        
        if provided_count > 0:
            # User provided some CBC values - process with them (missing values will use healthy defaults)
            try:
                cbc_result = process_cbc_test(cbc_data)
                if cbc_result:
                    state["cbc_available"] = True
                    state["cbc_result"] = cbc_result
            except Exception as e:
                print(f"Warning: Failed to process CBC test: {e}")
                import traceback
                print(traceback.format_exc())
                state["message"] = "I encountered an issue processing your CBC test results. Please verify the values and try again, or proceed without CBC."
                return state
        elif user_wants_cbc:
            # User wants to provide CBC but didn't provide values - ask for key ones
            state["message"] = "For CBC test, I need blood count values. Key values: WBC, RBC, HGB, HCT, PLT. Please fill the CBC form with your test results."
            return state
    elif not user_wants_cbc:
        # CBC is always recommended but user didn't select it
        state["missing_tests"].append("cbc")
    
    # Generate summary message
    available_tests = []
    if state["xray_available"]:
        available_tests.append("X-ray")
    if state["spirometry_available"]:
        available_tests.append("Spirometry")
    if state["cbc_available"]:
        available_tests.append("CBC")
    
    if available_tests:
        state["message"] = f"Diagnostic tests completed: {', '.join(available_tests)}"
    else:
        # Check if user has indicated which tests they want to provide
        conversation_text = " ".join([msg.get("content", "") for msg in conversation[-3:]])
        conversation_lower = conversation_text.lower()
        
        # Check if user mentioned which tests they have
        has_xray_mention = any(word in conversation_lower for word in ["xray", "x-ray", "chest xray", "chest x-ray"])
        has_cbc_mention = any(word in conversation_lower for word in ["cbc", "blood count", "blood test", "complete blood count"])
        has_spirometry_mention = any(word in conversation_lower for word in ["spirometry", "spirometry test", "lung function"])
        has_both_mention = any(phrase in conversation_lower for phrase in ["both", "i have both", "all tests", "all of them"])
        
        # Build recommendation message
        recommendations = []
        if "xray" in recommended_tests:
            recommendations.append("X-ray")
        if "spirometry" in recommended_tests:
            recommendations.append("Spirometry")
        recommendations.append("CBC")
        
        # If user hasn't specified which tests they want to provide, ask them
        if not (has_xray_mention or has_cbc_mention or has_spirometry_mention or has_both_mention):
            # Ask which tests they want to provide
            test_options = []
            if "xray" in recommended_tests:
                test_options.append("X-ray")
            if "spirometry" in recommended_tests:
                test_options.append("Spirometry")
            test_options.append("CBC")
            
            state["message"] = (
                f"Based on your symptoms, I recommend: {', '.join(test_options)}.\n\n"
                f"Which test results do you have available to provide?\n"
                f"- X-ray (upload image)\n"
                f"- CBC (blood count values)\n"
                f"- Both X-ray and CBC\n"
                f"- Spirometry (if recommended)\n\n"
                f"Please let me know which tests you can provide, and I'll give you the appropriate forms to fill."
            )
            return state
        else:
            # User has indicated which tests they have - process accordingly
            # The forms will be shown based on what they mentioned
            state["message"] = f"Based on your symptoms, I recommend: {', '.join(recommendations)}. Please provide the test results for the tests you mentioned."
    
    return state


def recommend_tests(symptoms: str, age: Optional[int], gender: Optional[str]) -> list:
    """
    Use LLM to recommend which tests are needed based on symptoms.
    
    Returns:
        List of recommended tests: ["xray", "spirometry"] or ["xray"] or ["spirometry"] or []
    """
    if not symptoms:
        return []
    
    patient_info = f"""Patient Information:
- Age: {age if age else "Unknown"}
- Gender: {gender if gender else "Unknown"}
- Symptoms: {symptoms}"""
    
    messages = [
        {
            "role": "system",
            "content": """You are a pulmonologist recommending diagnostic tests based on symptoms.

Based on the patient's symptoms, recommend which tests are needed:
- X-ray: Recommended for cough, chest pain, fever, suspected pneumonia, lung infection
- Spirometry: Recommended for breathing difficulty, asthma symptoms, COPD symptoms, chronic cough

Return ONLY a valid JSON object with this exact structure:
{
    "xray": true or false,
    "spirometry": true or false
}

Note: CBC is always recommended as a general overall body test, so don't include it in your response."""
        },
        {
            "role": "user",
            "content": patient_info
        }
    ]
    
    try:
        response_text = call_groq_llm(messages, temperature=0.3)
        
        # Parse JSON response
        response_text = response_text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        recommendations = json.loads(response_text)
        
        recommended = []
        if recommendations.get("xray", False):
            recommended.append("xray")
        if recommendations.get("spirometry", False):
            recommended.append("spirometry")
        
        return recommended
    
    except Exception as e:
        print(f"Warning: Failed to get test recommendations: {e}")
        # Default: recommend both if symptoms suggest respiratory issues
        if symptoms:
            return ["xray", "spirometry"]
        return []


def extract_test_data_from_conversation(conversation: list, recommended_tests: list) -> Dict[str, Any]:
    """
    Extract test data from conversation history.
    
    Returns:
        Dictionary with keys: "xray_image", "spirometry_data", "cbc_data"
    """
    extracted = {
        "xray_image": None,
        "spirometry_data": None,
        "cbc_data": None
    }
    
    # Format conversation for LLM analysis
    conv_text = "\n".join([
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
        for msg in conversation[-10:]
    ])
    
    if not conv_text:
        return extracted
    
    # Use LLM to extract test data from conversation
    messages = [
        {
            "role": "system",
            "content": """You are extracting diagnostic test data from a conversation.

The user may have mentioned:
1. X-ray results or uploaded an X-ray image
2. Spirometry values (FEV1, FVC, age, height, weight, BMI, sex, race)
3. CBC/Blood count values (WBC, RBC, HGB, HCT, PLT, etc.)

Extract any mentioned test values and return as JSON.

Return ONLY a valid JSON object with this structure:
{
    "has_xray": true or false,
    "has_spirometry": true or false,
    "has_cbc": true or false,
    "spirometry_values": {
        "sex": "Male" or "Female" or null,
        "race": "string or null",
        "age": number or null,
        "height": number or null,
        "weight": number or null,
        "bmi": number or null,
        "fev1": number or null,
        "fvc": number or null
    },
    "cbc_values": {
        "WBC": number or null,
        "LYMp": number or null,
        "NEUTp": number or null,
        "LYMn": number or null,
        "NEUTn": number or null,
        "RBC": number or null,
        "HGB": number or null,
        "HCT": number or null,
        "MCV": number or null,
        "MCH": number or null,
        "MCHC": number or null,
        "PLT": number or null,
        "PDW": number or null,
        "PCT": number or null
    }
}

Note: For X-ray, we can't extract image from text, so just mark has_xray as true if mentioned."""
        },
        {
            "role": "user",
            "content": f"Conversation:\n{conv_text}"
        }
    ]
    
    try:
        response_text = call_groq_llm(messages, temperature=0.3)
        
        # Parse JSON response
        response_text = response_text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        data = json.loads(response_text)
        
        if data.get("has_spirometry") and data.get("spirometry_values"):
            extracted["spirometry_data"] = data["spirometry_values"]
        
        if data.get("has_cbc") and data.get("cbc_values"):
            extracted["cbc_data"] = data["cbc_values"]
        
        # Note: X-ray image extraction would need file upload handling
        # For now, we'll rely on the user providing it separately
    
    except Exception as e:
        print(f"Warning: Failed to extract test data from conversation: {e}")
    
    return extracted


def process_xray_test(xray_image: Optional[Any]) -> Optional[Dict[str, Any]]:
    """
    Process X-ray test if image is available.
    
    Args:
        xray_image: PIL Image or image data
        
    Returns:
        Dictionary with prediction results or None
    """
    if not xray_image:
        return None
    
    try:
        # Convert to PIL Image if needed
        if isinstance(xray_image, bytes):
            image = Image.open(io.BytesIO(xray_image))
        elif isinstance(xray_image, str):
            # Assume it's a file path
            image = Image.open(xray_image)
        else:
            image = xray_image
        
        # Make prediction
        prediction = predict_xray(image)
        probabilities = predict_xray_proba(image)
        
        # predict_xray returns dict with "class_id", "class_name", and "confidence"
        # predict_xray_proba returns dict with class names as keys
        
        return {
            "prediction": {
                "class_id": int(prediction.get("class_id", 0)),
                "disease_name": prediction.get("class_name", "Unknown"),
                "confidence": float(prediction.get("confidence", 0.0))
            },
            "probabilities": {
                "no_disease": float(probabilities.get("No disease", 0.0)),
                "bacterial_pneumonia": float(probabilities.get("Bacterial pneumonia", 0.0)),
                "viral_pneumonia": float(probabilities.get("Viral pneumonia", 0.0))
            }
        }
    
    except Exception as e:
        print(f"Warning: X-ray prediction failed: {e}")
        return None


def process_spirometry_test(spirometry_data: Optional[Dict[str, Any]], 
                            patient_age: Optional[int],
                            patient_gender: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Process Spirometry test if data is available.
    
    Args:
        spirometry_data: Dictionary with spirometry values
        patient_age: Patient age (to fill if missing)
        patient_gender: Patient gender (to fill if missing)
        
    Returns:
        Dictionary with prediction results or None
    """
    if not spirometry_data:
        return None
    
    try:
        # Fill missing values with patient info or healthy defaults
        input_data = {
            "sex": spirometry_data.get("sex") or patient_gender or "Male",
            "race": spirometry_data.get("race") or "White",
            "age": spirometry_data.get("age") or patient_age or 45.0,
            "height": spirometry_data.get("height") or 170.0,
            "weight": spirometry_data.get("weight") or 70.0,
            "bmi": spirometry_data.get("bmi") or 24.0,
            "fev1": spirometry_data.get("fev1") or 3.5,  # Healthy default if missing
            "fvc": spirometry_data.get("fvc") or 4.5  # Healthy default if missing
        }
        
        # If FEV1 or FVC are still None (shouldn't happen now, but safety check)
        if not input_data.get("fev1") or not input_data.get("fvc"):
            return None
        
        # Make predictions
        predictions = predict_spirometry(input_data)
        probabilities = predict_spirometry_proba(input_data)
        
        # Determine status
        all_zero = all(v == 0 for v in predictions.values())
        status = "healthy" if all_zero else "abnormal"
        
        return {
            "input_values": input_data,
            "prediction": {
                "obstruction": int(predictions["obstruction"]),
                "restriction": int(predictions["restriction"]),
                "prism": int(predictions["prism"]),
                "mixed": int(predictions["mixed"])
            },
            "probabilities": {
                "obstruction": float(probabilities["obstruction"]),
                "restriction": float(probabilities["restriction"]),
                "prism": float(probabilities["prism"]),
                "mixed": float(probabilities["mixed"])
            },
            "status": status
        }
    
    except Exception as e:
        print(f"Warning: Spirometry prediction failed: {e}")
        return None


def process_cbc_test(cbc_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Process CBC/Blood Count test if data is available.
    
    Args:
        cbc_data: Dictionary with blood count values
        
    Returns:
        Dictionary with prediction results or None
    """
    if not cbc_data:
        return None
    
    try:
        # Make prediction
        prediction = predict_blood_disease(cbc_data)
        probabilities = predict_blood_disease_proba(cbc_data)
        
        return {
            "input_values": cbc_data,
            "prediction": {
                "class_id": int(prediction["class_id"]),
                "disease_name": prediction["disease_name"],
                "confidence": prediction.get("confidence")
            },
            "probabilities": probabilities["probabilities"]
        }
    
    except Exception as e:
        print(f"Warning: CBC prediction failed: {e}")
        return None

