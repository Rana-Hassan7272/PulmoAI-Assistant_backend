"""
Dosage Calculator Agent

Calculates appropriate medication dosages based on patient age, weight, and condition.
Uses hybrid approach: rule-based for common medications, LLM-based for complex cases.
"""

from typing import Dict, Any, Optional, List
from .state import AgentState
from .config import call_groq_llm
import re


# Rule-based dosing for common medications
COMMON_MEDICATION_DOSING = {
    "amoxicillin": {
        "adult_standard": "500mg",
        "adult_frequency": "twice daily",
        "pediatric_dose_per_kg": 20,  # mg/kg
        "pediatric_max": 500,  # mg
        "pediatric_frequency": "twice daily",
        "elderly_adjustment": "same as adult",
        "with_food": True,
        "notes": "Take with food to reduce stomach upset"
    },
    "azithromycin": {
        "adult_standard": "500mg",
        "adult_frequency": "once daily",
        "pediatric_dose_per_kg": 10,  # mg/kg
        "pediatric_max": 500,  # mg
        "pediatric_frequency": "once daily",
        "elderly_adjustment": "same as adult",
        "with_food": False,
        "notes": "Can be taken with or without food"
    },
    "paracetamol": {
        "adult_standard": "500mg",
        "adult_frequency": "every 4-6 hours",
        "adult_max_daily": "4000mg",
        "pediatric_dose_per_kg": 10,  # mg/kg
        "pediatric_max": 500,  # mg
        "pediatric_frequency": "every 4-6 hours",
        "pediatric_max_daily": "60mg/kg",
        "elderly_adjustment": "reduce to 325mg",
        "with_food": False,
        "notes": "Do not exceed maximum daily dose"
    },
    "ibuprofen": {
        "adult_standard": "400mg",
        "adult_frequency": "every 6-8 hours",
        "adult_max_daily": "1200mg",
        "pediatric_dose_per_kg": 10,  # mg/kg
        "pediatric_max": 400,  # mg
        "pediatric_frequency": "every 6-8 hours",
        "pediatric_max_daily": "40mg/kg",
        "elderly_adjustment": "reduce to 200mg",
        "with_food": True,
        "notes": "Take with food to reduce stomach irritation"
    },
    "prednisone": {
        "adult_standard": "20-40mg",
        "adult_frequency": "once daily",
        "pediatric_dose_per_kg": 1,  # mg/kg
        "pediatric_max": 40,  # mg
        "pediatric_frequency": "once daily",
        "elderly_adjustment": "reduce by 25%",
        "with_food": True,
        "notes": "Take with food, taper dose gradually"
    },
    "albuterol": {
        "adult_standard": "2-4 puffs",
        "adult_frequency": "every 4-6 hours as needed",
        "pediatric_dose_per_kg": None,  # Inhaler, not weight-based
        "pediatric_standard": "1-2 puffs",
        "pediatric_frequency": "every 4-6 hours as needed",
        "elderly_adjustment": "same as adult",
        "with_food": False,
        "notes": "Inhaler - use as needed for breathing difficulty"
    }
}


def extract_medication_name(treatment_text: str) -> Optional[str]:
    """
    Extract medication name from treatment text.
    
    Args:
        treatment_text: Treatment plan item (e.g., "Amoxicillin 500mg twice daily")
        
    Returns:
        Lowercase medication name or None
    """
    treatment_lower = treatment_text.lower()
    
    # Check against known medications
    for med_name in COMMON_MEDICATION_DOSING.keys():
        if med_name in treatment_lower:
            return med_name
    
    # Try to extract first word (medication name)
    words = treatment_lower.split()
    if words:
        # Remove common words
        med_name = words[0]
        if med_name not in ["take", "use", "apply", "inhale"]:
            return med_name
    
    return None


def calculate_pediatric_dose(medication: str, weight_kg: float, age_years: int) -> Optional[Dict[str, Any]]:
    """
    Calculate pediatric dose based on weight.
    
    Args:
        medication: Medication name (lowercase)
        weight_kg: Patient weight in kilograms
        age_years: Patient age in years
        
    Returns:
        Dictionary with calculated dose information or None
    """
    if medication not in COMMON_MEDICATION_DOSING:
        return None
    
    dosing_info = COMMON_MEDICATION_DOSING[medication]
    
    # Check if medication has pediatric dosing
    if "pediatric_dose_per_kg" not in dosing_info or dosing_info["pediatric_dose_per_kg"] is None:
        return None
    
    # Calculate dose
    dose_per_kg = dosing_info["pediatric_dose_per_kg"]
    calculated_dose = weight_kg * dose_per_kg
    
    # Apply maximum limit
    max_dose = dosing_info.get("pediatric_max")
    if max_dose and calculated_dose > max_dose:
        calculated_dose = max_dose
    
    # Round to nearest 50mg for tablets, or appropriate increment
    if calculated_dose >= 100:
        calculated_dose = round(calculated_dose / 50) * 50
    else:
        calculated_dose = round(calculated_dose / 25) * 25
    
    frequency = dosing_info.get("pediatric_frequency", "as directed")
    with_food = dosing_info.get("with_food", False)
    notes = dosing_info.get("notes", "")
    
    return {
        "dose": f"{int(calculated_dose)}mg",
        "frequency": frequency,
        "with_food": with_food,
        "notes": notes,
        "calculation": f"Based on weight: {weight_kg}kg × {dose_per_kg}mg/kg = {int(calculated_dose)}mg"
    }


def calculate_adult_dose(medication: str, age_years: int) -> Optional[Dict[str, Any]]:
    """
    Calculate adult dose.
    
    Args:
        medication: Medication name (lowercase)
        age_years: Patient age in years
        
    Returns:
        Dictionary with calculated dose information or None
    """
    if medication not in COMMON_MEDICATION_DOSING:
        return None
    
    dosing_info = COMMON_MEDICATION_DOSING[medication]
    
    # Check if elderly (age >= 65)
    is_elderly = age_years >= 65
    
    if is_elderly and "elderly_adjustment" in dosing_info:
        adjustment = dosing_info["elderly_adjustment"]
        if "reduce" in adjustment.lower():
            # Extract reduction percentage or amount
            standard_dose = dosing_info.get("adult_standard", "")
            # For now, use standard dose (can be enhanced)
            dose = standard_dose
        else:
            dose = dosing_info.get("adult_standard", "")
    else:
        dose = dosing_info.get("adult_standard", "")
    
    frequency = dosing_info.get("adult_frequency", "as directed")
    with_food = dosing_info.get("with_food", False)
    notes = dosing_info.get("notes", "")
    
    return {
        "dose": dose,
        "frequency": frequency,
        "with_food": with_food,
        "notes": notes,
        "calculation": "Standard adult dose"
    }


def calculate_dose_with_llm(medication: str, age_years: int, weight_kg: Optional[float], 
                            diagnosis: str, treatment_text: str) -> Dict[str, Any]:
    """
    Use LLM to calculate dose for medications not in rule-based database.
    
    Args:
        medication: Medication name
        age_years: Patient age
        weight_kg: Patient weight (optional)
        diagnosis: Patient diagnosis
        treatment_text: Original treatment text
        
    Returns:
        Dictionary with calculated dose information
    """
    messages = [
        {
            "role": "system",
            "content": """You are a clinical pharmacist calculating medication dosages.
Based on patient information, calculate the appropriate dosage for the medication.

Consider:
- Patient age (pediatric vs adult dosing)
- Patient weight (for weight-based dosing)
- Diagnosis/condition being treated
- Standard dosing guidelines

Return ONLY a valid JSON object with these exact keys:
{
    "dose": "specific dose (e.g., '500mg', '2 tablets', '1-2 puffs')",
    "frequency": "how often (e.g., 'twice daily', 'every 6 hours', 'once daily')",
    "with_food": true or false,
    "notes": "any important notes about taking the medication",
    "calculation": "brief explanation of how dose was calculated"
}

Be specific and accurate. Use standard medical dosing guidelines."""
        },
        {
            "role": "user",
            "content": f"""Calculate appropriate dosage for:

Medication: {medication}
Patient Age: {age_years} years
Patient Weight: {weight_kg if weight_kg else 'Not provided'} kg
Diagnosis: {diagnosis}
Original Treatment Text: {treatment_text}

Provide the calculated dosage in the specified JSON format."""
        }
    ]
    
    try:
        response_text = call_groq_llm(messages, temperature=0.3)
        
        # Parse JSON response
        import json
        response_text = response_text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        dose_info = json.loads(response_text)
        return dose_info
    except Exception as e:
        print(f"Warning: LLM dosage calculation failed: {e}")
        # Fallback to original treatment text
        return {
            "dose": "As prescribed",
            "frequency": "As directed",
            "with_food": None,
            "notes": "Please follow your healthcare provider's instructions",
            "calculation": "Unable to calculate specific dose"
        }


def calculate_medication_dosage(medication: str, age_years: int, weight_kg: Optional[float],
                               diagnosis: str, treatment_text: str) -> Dict[str, Any]:
    """
    Calculate medication dosage using hybrid approach.
    
    Args:
        medication: Medication name (lowercase)
        age_years: Patient age in years
        weight_kg: Patient weight in kilograms (optional)
        diagnosis: Patient diagnosis
        treatment_text: Original treatment text
        
    Returns:
        Dictionary with calculated dose information
    """
    # Try rule-based first
    is_pediatric = age_years < 18
    
    if is_pediatric and weight_kg:
        # Pediatric dosing (weight-based)
        dose_info = calculate_pediatric_dose(medication, weight_kg, age_years)
        if dose_info:
            return dose_info
    
    # Adult dosing or pediatric without weight
    dose_info = calculate_adult_dose(medication, age_years)
    if dose_info:
        return dose_info
    
    # If not in rule-based database, use LLM
    return calculate_dose_with_llm(medication, age_years, weight_kg, diagnosis, treatment_text)


def dosage_calculator_agent(state: AgentState) -> AgentState:
    """
    Dosage Calculator Agent - calculates specific dosages for medications.
    
    Now uses calculate_dosage_tool from tools.py.
    """
    state["current_step"] = "dosage_calculator"
    
    from .tools import calculate_dosage_tool
    
    treatment_plan = state.get("treatment_plan", [])
    if not treatment_plan:
        state["message"] = state.get("message") or "No treatment plan to calculate dosages for."
        return state
    
    patient_age = state.get("patient_age")
    if not patient_age:
        state["message"] = "Unable to calculate dosages: Patient age is required."
        return state
    
    # Call tool to calculate dosages
    result = calculate_dosage_tool(state)
    
    if result["status"] == "success" or result["status"] == "partial":
        dosages = result["dosages"]
        state["calculated_dosages"] = dosages
        
        # Update treatment plan with calculated dosages
        updated_treatment_plan = []
        for treatment_item in treatment_plan:
            # Try to match medication from treatment plan with calculated dosages
            medication_found = False
            for med_name, dose_info in dosages.items():
                if med_name.lower() in treatment_item.lower():
                    # Update treatment item with specific dosage
                    updated_item = f"{med_name.capitalize()} {dose_info.get('dose', '')} {dose_info.get('frequency', '')}"
                    if dose_info.get("duration"):
                        updated_item += f" for {dose_info.get('duration')}"
                    if dose_info.get("notes"):
                        updated_item += f" - Note: {dose_info.get('notes')}"
                    updated_treatment_plan.append(updated_item)
                    medication_found = True
                    break
            
            if not medication_found:
                # Keep original treatment item
                updated_treatment_plan.append(treatment_item)
        
        state["treatment_plan"] = updated_treatment_plan
        
        # Generate summary message
        patient_weight = state.get("patient_weight")
        dosage_summary = "Dosages calculated based on your age"
        if patient_weight:
            dosage_summary += f" and weight ({patient_weight}kg)"
        dosage_summary += ":\n\n"
        
        for med, dose_info in dosages.items():
            dosage_summary += f"{med.capitalize()}: {dose_info.get('dose', 'N/A')} {dose_info.get('frequency', 'N/A')}\n"
            if dose_info.get("notes"):
                dosage_summary += f"  Note: {dose_info.get('notes')}\n"
        
        state["message"] = dosage_summary
    else:
        state["message"] = result.get("message", "Dosage calculation encountered an issue.")
    
    return state

