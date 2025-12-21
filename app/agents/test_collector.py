"""
Test Collector Agent - Sequential State Machine
Handles X-ray, CBC, and Spirometry collection in a strict sequence.
"""
import json
import io
import re
from typing import Dict, Any, Optional, List
from PIL import Image
from .state import AgentState
from .config import call_groq_llm

# Import ML prediction functions
from ..ml_models.xray.preprocessor import predict_xray, predict_xray_proba
from ..ml_models.spirometry.featurizer import predict_spirometry, predict_spirometry_proba
from ..ml_models.bloodcount_report.feature import predict_blood_disease, predict_blood_disease_proba


def test_collector_agent(state: AgentState) -> AgentState:
    """
    Test Collector Agent - processes test submissions and requests missing ones.
    """
    state["current_step"] = "test_collector"
    
    # Reset modal flags every turn
    state["show_spirometry_form_modal"] = False
    state["show_cbc_form_modal"] = False
    
    # Get conversation context
    conversation = state.get("conversation_history", [])
    last_message = conversation[-1].get("content", "").lower() if conversation else ""
    
    # 1. IDENTIFY RECOMMENDED TESTS
    recommended = state.get("tests_recommended", [])
    if not recommended:
        recommended = ["xray", "spirometry", "cbc"] # Default fallback
        state["tests_recommended"] = recommended

    # 2. PROCESS INCOMING DATA (Manual or Form)
    just_collected = None
    
    # Check for X-ray (usually handled in router, but check here too)
    if "x-ray image uploaded" in last_message or "[x-ray image uploaded" in last_message:
        if state.get("xray_available"):
            just_collected = "X-ray"

    # Check for Spirometry
    if not state.get("spirometry_available"):
        spirom_data = _extract_spirometry_from_conversation(conversation)
        if spirom_data:
            result = _call_spirometry_ml_api_tool(spirom_data, state.get("patient_age"), state.get("patient_gender"))
            if result:
                state["spirometry_available"] = True
                state["spirometry_result"] = result
                just_collected = "Spirometry"
    elif "spirometry test results submitted" in last_message:
        # If submitted via form, values are already in state.spirometry_result
        just_collected = "Spirometry"

    # Check for CBC
    if not state.get("cbc_available"):
        cbc_data = _extract_cbc_from_conversation(conversation)
        if cbc_data:
            result = _call_cbc_ml_api_tool(cbc_data, state.get("patient_age"), state.get("patient_gender"))
            if result:
                state["cbc_available"] = True
                state["cbc_result"] = result
                just_collected = "CBC"
    elif "cbc test results submitted" in last_message:
        just_collected = "CBC"

    # 3. HANDLE SKIP REQUESTS
    if "skip" in last_message:
        target_skipped = None
        
        # Priority 1: Check if user named a specific test to skip
        if "xray" in last_message or "x-ray" in last_message:
            target_skipped = "xray"
        elif "spirometry" in last_message:
            target_skipped = "spirometry"
        elif "cbc" in last_message:
            target_skipped = "cbc"
        
        # Priority 2: Generic "skip" - find first missing recommended test
        if not target_skipped:
            missing_tests_list = state.get("missing_tests", [])
            for t in recommended:
                is_avail = (t == "xray" and state.get("xray_available")) or \
                           (t == "spirometry" and state.get("spirometry_available")) or \
                           (t == "cbc" and state.get("cbc_available"))
                if not is_avail and t not in missing_tests_list:
                    target_skipped = t
                    break
        
        if target_skipped:
            if target_skipped not in state.get("missing_tests", []):
                state.setdefault("missing_tests", []).append(target_skipped)
                test_names = {"xray": "X-ray", "spirometry": "Spirometry", "cbc": "CBC"}
                just_collected = f"{test_names.get(target_skipped, target_skipped)} (skipped)"

    # 4. HANDLE FORM REQUESTS
    if "form" in last_message:
        if "spirometry" in last_message and not state.get("spirometry_available"):
            state["show_spirometry_form_modal"] = True
            state["message"] = "I've opened the **Spirometry Form** for you. Please enter the FEV1 and FVC values from your report and click submit."
            return state
        if "cbc" in last_message and not state.get("cbc_available"):
            state["show_cbc_form_modal"] = True
            state["message"] = "I've opened the **CBC Form** for you. Please enter your blood count values and click submit."
            return state

    # 5. DETERMINE NEXT ACTION
    missing_tests = state.get("missing_tests", [])
    xray_avail = state.get("xray_available", False)
    spirom_avail = state.get("spirometry_available", False)
    cbc_avail = state.get("cbc_available", False)
    
    # Filter recommended to only those not handled
    todo = [t for t in recommended if 
            (t == "xray" and not xray_avail and "xray" not in missing_tests) or
            (t == "spirometry" and not spirom_avail and "spirometry" not in missing_tests) or
            (t == "cbc" and not cbc_avail and "cbc" not in missing_tests)]

    if not todo:
        # All done!
        return _structure_final_output(state)

    # Pick the next test in sequence
    next_test = todo[0]
    
    ack = f"Thank you for providing the {just_collected}. " if just_collected else ""
    
    if next_test == "xray":
        state["message"] = f"{ack}Next, I need your **X-ray image**. Please upload it using the button below, or say 'skip' if you don't have it."
    elif next_test == "cbc":
        state["message"] = f"{ack}Next, I recommend providing your **CBC (Blood Test)** results. You can ask me to show you the form or provide values manually. Say 'skip' to move on."
    elif next_test == "spirometry":
        state["message"] = f"{ack}Next, please provide your **Spirometry (Lung Function)** results. You can ask me for the form or provide values manually. Say 'skip' to move on."

    return state


def _extract_cbc_from_conversation(conversation: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """Extract CBC values from conversation."""
    last_message = conversation[-1].get("content", "") if conversation else ""
    
    def safe_float(match: Optional["re.Match"]) -> Optional[float]:
        if not match: return None
        val = match.group(1)
        try: return float(val)
        except: return None

    # Extraction patterns for all possible CBC fields
    patterns = {
        "WBC": re.search(r'wbc[:\s=]+([\d.]+)', last_message, re.IGNORECASE),
        "RBC": re.search(r'rbc[:\s=]+([\d.]+)', last_message, re.IGNORECASE),
        "HGB": re.search(r'(hgb|hemoglobin)[:\s=]+([\d.]+)', last_message, re.IGNORECASE),
        "HCT": re.search(r'(hct|hematocrit)[:\s=]+([\d.]+)', last_message, re.IGNORECASE),
        "PLT": re.search(r'(plt|platelets)[:\s=]+([\d.]+)', last_message, re.IGNORECASE),
        "MCV": re.search(r'mcv[:\s=]+([\d.]+)', last_message, re.IGNORECASE),
        "MCH": re.search(r'mch[:\s=]+([\d.]+)', last_message, re.IGNORECASE),
        "MCHC": re.search(r'mchc[:\s=]+([\d.]+)', last_message, re.IGNORECASE),
        "LYMp": re.search(r'lymp[:\s=]+([\d.]+)', last_message, re.IGNORECASE),
        "NEUTp": re.search(r'neutp[:\s=]+([\d.]+)', last_message, re.IGNORECASE),
        "LYMn": re.search(r'lymn[:\s=]+([\d.]+)', last_message, re.IGNORECASE),
        "NEUTn": re.search(r'neutn[:\s=]+([\d.]+)', last_message, re.IGNORECASE),
        "PDW": re.search(r'pdw[:\s=]+([\d.]+)', last_message, re.IGNORECASE),
        "PCT": re.search(r'pct[:\s=]+([\d.]+)', last_message, re.IGNORECASE),
    }
    
    if any(patterns.values()):
        return {k: safe_float(v) for k, v in patterns.items()}
    return None


def _extract_spirometry_from_conversation(conversation: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """Extract Spirometry values from conversation."""
    last_message = conversation[-1].get("content", "") if conversation else ""
    fev1_match = re.search(r'fev1[:\s=]+([\d.]+)', last_message, re.IGNORECASE)
    fvc_match = re.search(r'fvc[:\s=]+([\d.]+)', last_message, re.IGNORECASE)
    
    if fev1_match and fvc_match:
        return {
            "fev1": float(fev1_match.group(1)),
            "fvc": float(fvc_match.group(1))
        }
    return None


def _call_cbc_ml_api_tool(cbc_data: Dict[str, Any], age, gender) -> Optional[Dict[str, Any]]:
    """Call CBC ML API with all 14 required columns."""
    try:
        # Prepare input data with ALL 14 columns required by the model
        input_data = {
            "WBC": float(cbc_data.get("WBC") or 7.0),
            "LYMp": float(cbc_data.get("LYMp") or 30.0),
            "NEUTp": float(cbc_data.get("NEUTp") or 60.0),
            "LYMn": float(cbc_data.get("LYMn") or 2.0),
            "NEUTn": float(cbc_data.get("NEUTn") or 4.0),
            "RBC": float(cbc_data.get("RBC") or 4.5),
            "HGB": float(cbc_data.get("HGB") or 14.0),
            "HCT": float(cbc_data.get("HCT") or 42.0),
            "MCV": float(cbc_data.get("MCV") or 90.0),
            "MCH": float(cbc_data.get("MCH") or 30.0),
            "MCHC": float(cbc_data.get("MCHC") or 33.0),
            "PLT": float(cbc_data.get("PLT") or 250.0),
            "PDW": float(cbc_data.get("PDW") or 12.0),
            "PCT": float(cbc_data.get("PCT") or 0.2)
        }
        
        # Call the actual ML model function
        prediction = predict_blood_disease(input_data)
        
        return {
            "input_values": input_data,
            "prediction": {
                "disease_name": prediction.get("disease_name", "Unknown"),
                "confidence": float(prediction.get("confidence", 0.0))
            }
        }
    except Exception as e:
        print(f"ERROR: CBC ML call failed: {e}")
        return None


def _call_spirometry_ml_api_tool(spirometry_data: Dict[str, Any], age, gender) -> Optional[Dict[str, Any]]:
    try:
        fev1 = float(spirometry_data["fev1"])
        fvc = float(spirometry_data["fvc"])
        
        # CORRECT CALCULATION: Baseline_FEV1_FVC_Ratio = FEV1 / FVC
        # Make sure FVC is not zero
        ratio = fev1 / fvc if fvc > 0 else 0.0
        
        input_data = {
            "sex": "Male" if str(gender).lower().startswith('m') else "Female",
            "age": float(age or 45),
            "fev1": fev1,
            "fvc": fvc,
            "fev1_fvc": ratio, # Pass calculated ratio
            "height": 170.0, "weight": 70.0, "bmi": 24.0, "race": "White"
        }
        # Get binary predictions (0 or 1)
        predictions = predict_spirometry(input_data)
        # Get raw probabilities (0.0 to 1.0)
        probabilities = predict_spirometry_proba(input_data)
        
        # Determine pattern and confidence
        pattern = "Normal"
        confidence = 0.0
        
        # Check if any disease model returned positive (1)
        active_diseases = [k for k, v in predictions.items() if v == 1]
        
        if active_diseases:
            # Finding is a disease. Confidence is the probability of the most likely disease found.
            max_p = 0.0
            best_d = "Normal"
            for d in active_diseases:
                if probabilities.get(d, 0.0) > max_p:
                    max_p = probabilities[d]
                    best_d = d.capitalize()
            pattern = best_d
            confidence = max_p
        else:
            # Finding is Normal. Confidence is 1 - highest disease probability.
            # We want to show how sure we are it is NOT a disease.
            max_disease_p = max(probabilities.values()) if probabilities else 0.0
            pattern = "Normal"
            confidence = 1.0 - max_disease_p
            
        # CLIP CONFIDENCE to realistic range (don't always show 100% unless it truly is)
        if confidence > 0.999: confidence = 0.999
        if confidence < 0.01: confidence = 0.01
        
        return {
            "input_values": input_data,
            "pattern": pattern,
            "severity": "Normal" if pattern == "Normal" else "Mild", 
            "confidence": float(confidence),
            "prediction": {"pattern": pattern, "confidence": float(confidence)}
        }
    except Exception as e:
        print(f"ERROR: Spirometry ML call failed: {e}")
        return None


def _structure_final_output(state: AgentState) -> AgentState:
    doctor_note = state.get("doctor_note", "")
    results = []
    if state.get("xray_available") and state.get("xray_result"):
        p = state["xray_result"].get("prediction", {})
        results.append(f"X-ray: {p.get('disease_name')} ({p.get('confidence', 0):.1%})")
    if state.get("spirometry_available") and state.get("spirometry_result"):
        res = state['spirometry_result']
        p = res.get('prediction', {})
        conf = res.get('confidence', p.get('confidence', 0))
        results.append(f"Spirometry: {res.get('pattern')} ({conf:.1%})")
    if state.get("cbc_available") and state.get("cbc_result"):
        p = state["cbc_result"].get("prediction", {})
        results.append(f"CBC: {p.get('disease_name')} ({p.get('confidence', 0):.1%})")
    
    missing = state.get("missing_tests", [])
    if missing: results.append(f"Skipped: {', '.join(missing)}")

    test_summary = "\n".join(results)
    state["message"] = f"{doctor_note}\n\n**Diagnostic Test Results:**\n{test_summary}\n\nProceeding with diagnosis."
    state["test_collection_complete"] = True
    state["test_collector_findings"] = test_summary
    return state
