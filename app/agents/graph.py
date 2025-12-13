"""
LangGraph Graph Setup for Pulmonologist Assistant

This sets up the main LangGraph workflow with all agent nodes.
"""
from typing import Literal, Optional, Dict, Any
import json
from langgraph.graph import StateGraph, END
from .state import AgentState

# Always import MemorySaver (used as default/fallback)
from langgraph.checkpoint.memory import MemorySaver

# Try to import SqliteSaver, fallback to MemorySaver if not available
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    print("Warning: langgraph-checkpoint-sqlite not installed. Using MemorySaver instead.")
    print("Install with: pip install langgraph-checkpoint-sqlite")
from .patient_intake import patient_intake_agent
from .emergency_detector import emergency_detector_agent
from .diagnostic_controller import diagnostic_controller_agent
from .dosage_calculator import dosage_calculator_agent
from .config import call_groq_llm
from .rag.rag_agent import get_rag_agent


def create_diagnostic_graph():
    """
    Create the LangGraph workflow for the Pulmonologist Assistant.
    
    Returns:
        Compiled LangGraph graph
    """
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes (agents)
    workflow.add_node("patient_intake", patient_intake_agent)
    workflow.add_node("emergency_detector", emergency_detector_agent)
    workflow.add_node("doctor_note_generator", doctor_note_generator_agent)
    workflow.add_node("diagnostic_controller", diagnostic_controller_agent)
    workflow.add_node("rag_specialist", rag_specialist_agent)
    workflow.add_node("treatment_approval", treatment_approval_agent)
    workflow.add_node("dosage_calculator", dosage_calculator_agent)
    workflow.add_node("report_generator", report_generator_agent)
    workflow.add_node("history_saver", history_saver_agent)
    workflow.add_node("followup_agent", followup_agent)
    
    # Set entry point
    workflow.set_entry_point("patient_intake")
    
    # Add edges (conditional routing)
    # Patient intake: check if confirmation needed
    workflow.add_conditional_edges(
        "patient_intake",
        check_patient_confirmation,
        {
            "awaiting_confirmation": "patient_intake",  # Loop back to wait for confirmation
            "confirmed": "emergency_detector",
            "error": END,  # Exit on error
            "waiting": END  # Exit when waiting for initial user input
        }
    )
    
    # Emergency detector: if emergency -> END, else -> doctor_note_generator
    workflow.add_conditional_edges(
        "emergency_detector",
        check_emergency,
        {
            "emergency": END,
            "continue": "doctor_note_generator"
        }
    )
    workflow.add_edge("doctor_note_generator", "diagnostic_controller")
    workflow.add_edge("diagnostic_controller", "rag_specialist")
    
    # RAG specialist: check if treatment approval needed
    workflow.add_conditional_edges(
        "rag_specialist",
        check_treatment_approval,
        {
            "awaiting_approval": "treatment_approval",  # Go to approval handler
            "approved": "report_generator"  # Proceed to report
        }
    )
    
    # Treatment approval: loop back to rag_specialist if still awaiting, or proceed
    workflow.add_conditional_edges(
        "treatment_approval",
        check_treatment_approval,
        {
            "awaiting_approval": "treatment_approval",  # Still waiting
            "approved": "dosage_calculator"  # Approved, calculate dosages
        }
    )
    
    # Dosage calculator: always proceed to report generator
    workflow.add_edge("dosage_calculator", "report_generator")
    workflow.add_edge("report_generator", "history_saver")
    workflow.add_edge("history_saver", "followup_agent")
    workflow.add_edge("followup_agent", END)
    
    # Compile the graph with checkpointer for state persistence
    # Using MemorySaver for now (SqliteSaver has context manager issues)
    # TODO: Fix SqliteSaver usage when context manager pattern is resolved
    checkpointer = MemorySaver()
    print("✓ Using MemorySaver for checkpointing")
    print("  Note: State persists during server runtime but is lost on restart")
    
    # TODO: Fix SqliteSaver - from_conn_string() returns context manager
    # if SQLITE_AVAILABLE:
    #     import os
    #     from pathlib import Path
    #     checkpoint_dir = Path(__file__).parent.parent.parent.parent / "data" / "checkpoints"
    #     checkpoint_dir.mkdir(parents=True, exist_ok=True)
    #     checkpoint_db = str(checkpoint_dir / "checkpoints.db")
    #     # SqliteSaver.from_conn_string() returns context manager - needs proper handling
    #     checkpointer = SqliteSaver.from_conn_string(checkpoint_db)
    
    app = workflow.compile(checkpointer=checkpointer)
    
    return app


def format_test_results_for_llm(state: AgentState) -> str:
    """
    Format all test results (X-ray, Spirometry, CBC) into a readable string for LLM prompts.
    
    Args:
        state: Current agent state
        
    Returns:
        Formatted string with test results
    """
    test_results = []
    
    # X-ray Results
    xray_result = state.get("xray_result")
    if xray_result and state.get("xray_available"):
        prediction = xray_result.get("prediction", {})
        disease_name = prediction.get("disease_name", "Unknown")
        confidence = prediction.get("confidence", 0.0)
        probabilities = xray_result.get("probabilities", {})
        
        xray_text = f"X-ray Analysis:\n"
        xray_text += f"  - Finding: {disease_name}\n"
        xray_text += f"  - Confidence: {confidence:.1%}\n"
        if probabilities:
            xray_text += f"  - Probabilities:\n"
            for disease, prob in probabilities.items():
                xray_text += f"    * {disease}: {prob:.1%}\n"
        test_results.append(xray_text)
    
    # Spirometry Results
    spirometry_result = state.get("spirometry_result")
    if spirometry_result and state.get("spirometry_available"):
        prediction = spirometry_result.get("prediction", {})
        status = spirometry_result.get("status", "unknown")
        
        spirometry_text = f"Spirometry Analysis:\n"
        spirometry_text += f"  - Status: {status}\n"
        spirometry_text += f"  - Obstruction: {'Yes' if prediction.get('obstruction') else 'No'}\n"
        spirometry_text += f"  - Restriction: {'Yes' if prediction.get('restriction') else 'No'}\n"
        spirometry_text += f"  - PRISm: {'Yes' if prediction.get('prism') else 'No'}\n"
        spirometry_text += f"  - Mixed: {'Yes' if prediction.get('mixed') else 'No'}\n"
        
        probabilities = spirometry_result.get("probabilities", {})
        if probabilities:
            spirometry_text += f"  - Probabilities:\n"
            for condition, prob in probabilities.items():
                spirometry_text += f"    * {condition}: {prob:.1%}\n"
        test_results.append(spirometry_text)
    
    # CBC Results
    cbc_result = state.get("cbc_result")
    if cbc_result and state.get("cbc_available"):
        prediction = cbc_result.get("prediction", {})
        disease_name = prediction.get("disease_name", "Unknown")
        confidence = prediction.get("confidence")
        
        cbc_text = f"CBC (Complete Blood Count) Analysis:\n"
        cbc_text += f"  - Finding: {disease_name}\n"
        if confidence is not None:
            cbc_text += f"  - Confidence: {confidence:.1%}\n"
        
        probabilities = cbc_result.get("probabilities", {}).get("probabilities", {})
        if probabilities:
            cbc_text += f"  - Probabilities:\n"
            for disease, prob in probabilities.items():
                cbc_text += f"    * {disease}: {prob:.1%}\n"
        test_results.append(cbc_text)
    
    if not test_results:
        return "No diagnostic test results available yet."
    
    return "\n\n".join(test_results)


def doctor_note_generator_agent(state: AgentState) -> AgentState:
    """
    Doctor Note Generator - creates comprehensive doctor note including test results.
    """
    state["current_step"] = "doctor_note_generator"
    
    # Get test results formatted for LLM
    test_results_text = format_test_results_for_llm(state)
    
    # Build comprehensive doctor note with test results
    messages = [
        {
            "role": "system",
            "content": """You are a pulmonologist writing a comprehensive clinical note.
Create a detailed 3-5 sentence summary in medical language.
Include: patient demographics, symptoms, duration, medical history, and ALL available test results.
If test results are available (X-ray, Spirometry, CBC), incorporate them into the note.
Use professional medical terminology."""
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
- Occupation: {state.get("patient_occupation", "Not specified")}

Diagnostic Test Results:
{test_results_text}

Generate a comprehensive doctor's note that incorporates all available information, especially the test results."""
        }
    ]
    
    try:
        doctor_note = call_groq_llm(messages, temperature=0.5)
        if doctor_note and doctor_note.strip():
            state["doctor_note"] = doctor_note.strip()
        else:
            raise ValueError("Empty response from LLM")
    except Exception as e:
        print(f"Warning: Doctor note generation failed: {e}")
        import traceback
        print(traceback.format_exc())
        # Fallback to basic note
        state["doctor_note"] = state.get("doctor_note") or (
            f"Patient: {state.get('patient_name', 'Unknown')}, "
            f"Age: {state.get('patient_age', 'Unknown')}, "
            f"Symptoms: {state.get('symptoms', 'Not specified')}"
        )
        if not state.get("message"):
            state["message"] = "Initial assessment completed. Proceeding with diagnostic tests."
    
    return state


def rag_specialist_agent(state: AgentState) -> AgentState:
    """
    RAG Specialist - generates diagnosis and treatment using RAG and test results.
    
    This agent:
    1. Retrieves relevant medical knowledge using RAG
    2. Uses retrieved context to generate accurate diagnosis
    3. Provides evidence-based treatment recommendations
    """
    state["current_step"] = "rag_specialist"
    
    # Get test results formatted for LLM
    test_results_text = format_test_results_for_llm(state)
    
    # Build query for RAG retrieval
    symptoms = state.get("symptoms", "")
    patient_age = state.get("patient_age")
    patient_gender = state.get("patient_gender", "")
    chronic_conditions = state.get("patient_chronic_conditions", "")
    
    # Create comprehensive query for RAG
    rag_query = f"""
    Patient symptoms: {symptoms}
    Age: {patient_age if patient_age else "Unknown"}
    Gender: {patient_gender if patient_gender else "Unknown"}
    Medical history: {chronic_conditions if chronic_conditions else "None"}
    Test results: {test_results_text}
    """
    
    # Retrieve relevant medical knowledge using RAG
    rag_context = ""
    try:
        rag_agent = get_rag_agent()
        rag_context = rag_agent.retrieve_context(
            query=rag_query,
            k=5,  # Retrieve top 5 relevant documents
            min_similarity=0.3  # Minimum similarity threshold
        )
        state["rag_context"] = rag_context
    except Exception as e:
        print(f"Warning: RAG retrieval failed: {e}")
        rag_context = "No additional medical knowledge retrieved. Using general medical knowledge."
        state["rag_context"] = None
    
    # Build diagnosis prompt with RAG context and test results
    messages = [
        {
            "role": "system",
            "content": """You are a pulmonologist specialist providing diagnosis and treatment recommendations.
Analyze the patient information, test results, and retrieved medical knowledge to provide:
1. Primary diagnosis (be specific and evidence-based)
2. Treatment plan (list of medications, procedures, lifestyle changes)
3. Home remedies (if applicable)
4. Follow-up instructions

Base your diagnosis on:
- Test results (especially X-ray findings for pneumonia detection)
- Patient symptoms and medical history
- Retrieved medical knowledge and guidelines

Use the retrieved medical knowledge to ensure your recommendations are evidence-based and current."""
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

Diagnostic Test Results:
{test_results_text}

Doctor's Note:
{state.get("doctor_note", "Not available")}

Retrieved Medical Knowledge:
{rag_context}

Provide:
1. Primary diagnosis (based on test results, symptoms, and medical knowledge)
2. Treatment plan (as a JSON array of strings) - Include medication names, dosages, timing (e.g., "Amoxicillin 500mg twice daily with meals")
3. Home remedies (as a JSON array of strings, if applicable)
4. Follow-up instructions - Specify when patient should return (e.g., "Return in 7 days" or "Follow up in 2 weeks if symptoms persist")

IMPORTANT: 
- For treatment plan, be specific about medication timing and dosage
- For follow-up, specify exact number of days/weeks when patient should return

Return your response as a JSON object with keys: diagnosis, treatment_plan, home_remedies, followup_instruction"""
        }
    ]
    
    try:
        response_text = call_groq_llm(messages, temperature=0.6)
        
        # Parse JSON response
        response_text = response_text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        diagnosis_data = json.loads(response_text)
        
        state["diagnosis"] = diagnosis_data.get("diagnosis", "Diagnosis pending")
        
        # Deduplicate treatment plan (remove exact duplicates)
        treatment_plan = diagnosis_data.get("treatment_plan", [])
        if isinstance(treatment_plan, list):
            # Remove duplicates while preserving order
            seen = set()
            deduplicated_plan = []
            for item in treatment_plan:
                # Normalize the item for comparison (lowercase, strip whitespace)
                normalized = item.strip().lower() if isinstance(item, str) else str(item).strip().lower()
                if normalized not in seen:
                    seen.add(normalized)
                    deduplicated_plan.append(item)
            state["treatment_plan"] = deduplicated_plan
        else:
            state["treatment_plan"] = treatment_plan
        
        state["home_remedies"] = diagnosis_data.get("home_remedies", [])
        state["followup_instruction"] = diagnosis_data.get("followup_instruction", "Follow up as needed")
        
        # Check if treatment is already approved
        if state.get("treatment_approved", False):
            # Treatment already approved, proceed to report generator
            state["current_step"] = "rag_specialist_complete"
            return state
        
        # Show treatment plan for patient approval
        treatment_summary = format_treatment_for_approval(state)
        state["message"] = (
            "Based on your test results and medical knowledge, here's my diagnosis and treatment plan:\n\n"
            f"{treatment_summary}\n\n"
            "Please review and let me know:\n"
            "- 'Yes' or 'Approve' if you're okay with this treatment plan\n"
            "- 'No' or tell me if any medication doesn't suit you or if you have questions\n"
            "- Ask questions about medication timing, dosage, or how to take them"
        )
        state["current_step"] = "rag_specialist_awaiting_approval"
        
    except json.JSONDecodeError as e:
        print(f"Warning: RAG specialist JSON parsing failed: {e}")
        import traceback
        print(traceback.format_exc())
        # Fallback diagnosis
        xray_result = state.get("xray_result")
        if xray_result:
            disease_name = xray_result.get("prediction", {}).get("disease_name", "Unknown")
            state["diagnosis"] = f"Based on X-ray analysis: {disease_name}"
        else:
            state["diagnosis"] = "Diagnosis pending further evaluation"
        state["treatment_plan"] = ["Consult with healthcare provider"]
        state["home_remedies"] = []
        state["followup_instruction"] = "Follow up as needed"
        
        # Even with fallback, show for approval
        treatment_summary = format_treatment_for_approval(state)
        state["message"] = (
            "Here's the treatment plan:\n\n"
            f"{treatment_summary}\n\n"
            "Please confirm if you're okay with this plan."
        )
        state["current_step"] = "rag_specialist_awaiting_approval"
    except Exception as e:
        print(f"Warning: RAG specialist failed: {e}")
        import traceback
        print(traceback.format_exc())
        # Fallback diagnosis
        xray_result = state.get("xray_result")
        if xray_result:
            disease_name = xray_result.get("prediction", {}).get("disease_name", "Unknown")
            state["diagnosis"] = f"Based on X-ray analysis: {disease_name}"
        else:
            state["diagnosis"] = "Diagnosis pending further evaluation"
        state["treatment_plan"] = ["Consult with healthcare provider"]
        state["home_remedies"] = []
        state["followup_instruction"] = "Follow up as needed"
        
        # Even with fallback, show for approval
        treatment_summary = format_treatment_for_approval(state)
        state["message"] = (
            "Here's the treatment plan:\n\n"
            f"{treatment_summary}\n\n"
            "Please confirm if you're okay with this plan."
        )
        state["current_step"] = "rag_specialist_awaiting_approval"
    
    return state


def format_treatment_for_approval(state: AgentState) -> str:
    """
    Format treatment plan for patient approval.
    
    Args:
        state: Current agent state with diagnosis and treatment
        
    Returns:
        Formatted treatment summary
    """
    diagnosis = state.get("diagnosis", "Pending diagnosis")
    treatment_plan = state.get("treatment_plan", [])
    home_remedies = state.get("home_remedies", [])
    followup = state.get("followup_instruction", "Follow up as needed")
    
    summary = f"Diagnosis: {diagnosis}\n\n"
    
    if treatment_plan:
        summary += "Treatment Plan:\n"
        for i, treatment in enumerate(treatment_plan, 1):
            summary += f"{i}. {treatment}\n"
    else:
        summary += "Treatment Plan: To be determined\n"
    
    if home_remedies:
        summary += "\nHome Remedies:\n"
        for i, remedy in enumerate(home_remedies, 1):
            summary += f"{i}. {remedy}\n"
    
    summary += f"\nFollow-up: {followup}"
    
    return summary


def treatment_approval_agent(state: AgentState) -> AgentState:
    """
    Treatment Approval Agent - handles patient approval/modification of treatment plan.
    
    This agent:
    1. Checks if patient approved treatment
    2. Handles questions about medications
    3. Processes modifications if patient has concerns
    4. Updates treatment plan if needed
    """
    state["current_step"] = "treatment_approval"
    
    conversation = state.get("conversation_history", [])
    last_message = conversation[-1].get("content", "").lower() if conversation else ""
    
    # Check if patient approved
    if any(word in last_message for word in ["yes", "approve", "ok", "okay", "fine", "good", "proceed"]):
        state["treatment_approved"] = True
        state["message"] = "Thank you for approving the treatment plan. Generating your final report..."
        state["current_step"] = "treatment_approval_complete"
        return state
    
    # Check if patient has questions or concerns
    question_keywords = ["how", "when", "what", "why", "time", "dose", "dosage", "take", "medicine", "medication", "confused", "understand"]
    concern_keywords = ["allergy", "allergic", "suit", "suitable", "problem", "issue", "can't", "cannot", "don't like", "not good"]
    
    has_question = any(keyword in last_message for keyword in question_keywords)
    has_concern = any(keyword in last_message for keyword in concern_keywords)
    
    if has_question:
        # Patient has questions, use LLM to answer
        treatment_plan_text = "\n".join([f"- {t}" for t in state.get("treatment_plan", [])])
        
        messages = [
            {
                "role": "system",
                "content": """You are a helpful pulmonologist assistant answering patient questions about their treatment plan.
Provide clear, simple explanations about:
- How to take medications (timing, with/without food)
- Dosage and frequency
- What to expect
- When to take each medication
- Any precautions

Be patient-friendly and clear."""
            },
            {
                "role": "user",
                "content": f"""Patient's Question: {conversation[-1].get('content', '')}

Treatment Plan:
{treatment_plan_text}

Please answer the patient's question clearly and helpfully."""
            }
        ]
        
        try:
            answer = call_groq_llm(messages, temperature=0.5)
            if answer and answer.strip():
                state["message"] = answer.strip()
            else:
                state["message"] = "I apologize, but I couldn't generate a response to your question. Please try rephrasing your question or consult with your healthcare provider."
            state["current_step"] = "rag_specialist_awaiting_approval"  # Still waiting for approval
        except Exception as e:
            print(f"Warning: Failed to answer patient question: {e}")
            import traceback
            print(traceback.format_exc())
            state["message"] = "I apologize, but I'm having trouble processing your question right now. Please try rephrasing it or ask about a specific medication from your treatment plan."
            state["current_step"] = "rag_specialist_awaiting_approval"
        return state
    
    elif has_concern or any(word in last_message for word in ["no", "change", "modify", "different", "alternative"]):
        # Patient has concerns or wants modification, use LLM to suggest alternatives
        treatment_plan_text = "\n".join([f"- {t}" for t in state.get("treatment_plan", [])])
        diagnosis = state.get("diagnosis", "Unknown")
        
        messages = [
            {
                "role": "system",
                "content": """You are a pulmonologist adjusting treatment plan based on patient concerns.
If patient has allergies or medications don't suit them, suggest appropriate alternatives.
Maintain treatment effectiveness while addressing patient concerns."""
            },
            {
                "role": "user",
                "content": f"""Patient's Concern: {conversation[-1].get('content', '')}

Current Diagnosis: {diagnosis}

Current Treatment Plan:
{treatment_plan_text}

Please suggest a modified treatment plan that addresses the patient's concern while maintaining effectiveness.
Return a JSON object with keys: treatment_plan (array), explanation (string explaining the change)"""
            }
        ]
        
        try:
            response_text = call_groq_llm(messages, temperature=0.5)
            response_text = response_text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            import json
            modification = json.loads(response_text)
            
            # Update treatment plan
            state["treatment_plan"] = modification.get("treatment_plan", state.get("treatment_plan", []))
            state["treatment_modifications"] = modification.get("explanation", "Treatment plan modified based on patient concern")
            
            # Show updated plan for approval
            treatment_summary = format_treatment_for_approval(state)
            explanation = modification.get("explanation", "")
            state["message"] = (
                f"I understand your concern. {explanation}\n\n"
                f"Here's the updated treatment plan:\n\n"
                f"{treatment_summary}\n\n"
                "Is this updated plan acceptable? Please reply 'Yes' to approve or let me know if you need further adjustments."
            )
            state["current_step"] = "rag_specialist_awaiting_approval"
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse treatment modification response: {e}")
            import traceback
            print(traceback.format_exc())
            state["message"] = "I understand your concern. I'm having difficulty processing the modification request. Please consult with your healthcare provider for alternative treatment options, or you can proceed with the original plan."
            state["current_step"] = "rag_specialist_awaiting_approval"
        except Exception as e:
            print(f"Warning: Failed to modify treatment: {e}")
            import traceback
            print(traceback.format_exc())
            state["message"] = "I understand your concern. I'm having trouble processing your request right now. Please consult with your healthcare provider for alternative treatment options, or you can proceed with the original plan."
            state["current_step"] = "rag_specialist_awaiting_approval"
        
        return state
    
    else:
        # Unclear response, ask again
        treatment_summary = format_treatment_for_approval(state)
        state["message"] = (
            "I need your confirmation on the treatment plan:\n\n"
            f"{treatment_summary}\n\n"
            "Please reply:\n"
            "- 'Yes' to approve the treatment\n"
            "- Ask questions if you're unsure about anything\n"
            "- Tell me if any medication doesn't suit you"
        )
        state["current_step"] = "rag_specialist_awaiting_approval"
        return state


def report_generator_agent(state: AgentState) -> AgentState:
    """
    Report Generator - combines all data into final comprehensive report.
    """
    state["current_step"] = "report_generator"
    
    # Get test results formatted for LLM
    test_results_text = format_test_results_for_llm(state)
    
    # Get dosage information if available
    calculated_dosages = state.get("calculated_dosages", {})
    dosage_info_text = ""
    if calculated_dosages:
        dosage_info_text = "\n\nCalculated Medication Dosages:\n"
        for med, dose_info in calculated_dosages.items():
            dosage_info_text += f"- {med.capitalize()}: {dose_info.get('dose', 'N/A')} {dose_info.get('frequency', 'N/A')}\n"
            if dose_info.get('notes'):
                dosage_info_text += f"  Note: {dose_info.get('notes')}\n"
    
    # Build final report
    messages = [
        {
            "role": "system",
            "content": """You are a medical report generator creating a comprehensive patient report.
Create a well-structured, professional medical report that includes:
- Patient information summary
- Chief complaints and symptoms
- Diagnostic test results (with emphasis on X-ray findings)
- Diagnosis
- Treatment plan with specific dosages
- Recommendations

Format it as a clear, readable report suitable for patient review."""
        },
        {
            "role": "user",
            "content": f"""Patient Information:
- Name: {state.get("patient_name", "Unknown")}
- Age: {state.get("patient_age", "Unknown")}
- Gender: {state.get("patient_gender", "Unknown")}
- Weight: {state.get("patient_weight", "Not provided")} kg
- Smoker: {"Yes" if state.get("patient_smoker") else "No" if state.get("patient_smoker") is False else "Unknown"}
- Symptoms: {state.get("symptoms", "Not specified")}
- Duration: {state.get("symptom_duration", "Not specified")}
- Medical History: {state.get("patient_chronic_conditions", "None")}
- Occupation: {state.get("patient_occupation", "Not specified")}

Diagnostic Test Results:
{test_results_text}

Doctor's Note:
{state.get("doctor_note", "Not available")}

Diagnosis:
{state.get("diagnosis", "Pending")}

Treatment Plan:
{', '.join(state.get("treatment_plan", []))}
{dosage_info_text}

Home Remedies:
{', '.join(state.get("home_remedies", [])) if state.get("home_remedies") else "None"}

Follow-up Instructions:
{state.get("followup_instruction", "Not specified")}

Generate a comprehensive, well-formatted medical report that includes the calculated dosages."""
        }
    ]
    
    try:
        final_report = call_groq_llm(messages, temperature=0.5)
        if final_report and final_report.strip():
            # Store report in state (you can add a "final_report" field to state if needed)
            state["message"] = final_report.strip()
        else:
            raise ValueError("Empty response from LLM")
    except Exception as e:
        print(f"Warning: Report generation failed: {e}")
        import traceback
        print(traceback.format_exc())
        # Fallback report with available information
        diagnosis = state.get('diagnosis', 'Pending')
        treatment = ', '.join(state.get('treatment_plan', [])) if state.get('treatment_plan') else 'To be determined'
        state["message"] = (
            f"Patient Visit Report\n\n"
            f"Diagnosis: {diagnosis}\n"
            f"Treatment Plan: {treatment}\n"
            f"Follow-up: {state.get('followup_instruction', 'As needed')}\n\n"
            f"Note: Full report generation encountered an issue. Please review the diagnosis and treatment plan above."
        )
    
    return state


def history_saver_agent(state: AgentState) -> AgentState:
    """
    History Saver - saves visit data to database.
    
    Saves:
    1. Patient information (create or update)
    2. Visit record with all visit data
    3. Diagnosis record with treatment plan, home remedies, etc.
    """
    state["current_step"] = "history_saver"
    
    try:
        from ..core.database import SessionLocal
        from ..db_models.patient import Patient
        from ..db_models.visit import Visit
        from ..db_models.diagnosis import Diagnosis
        import json
        
        db = SessionLocal()
        
        try:
            # Step 1: Create or update Patient
            patient_id = state.get("patient_id")
            patient_name = state.get("patient_name")
            patient_age = state.get("patient_age")
            patient_gender = state.get("patient_gender")
            patient_smoker = state.get("patient_smoker")
            patient_chronic_conditions = state.get("patient_chronic_conditions")
            patient_occupation = state.get("patient_occupation")
            
            if patient_id:
                # Update existing patient
                patient = db.query(Patient).filter(Patient.id == patient_id).first()
                if patient:
                    if patient_name: patient.name = patient_name
                    if patient_age: patient.age = patient_age
                    if patient_gender: patient.gender = patient_gender
                    if patient_smoker is not None: patient.smoker = patient_smoker
                    if patient_chronic_conditions: patient.chronic_conditions = patient_chronic_conditions
                    if patient_occupation: patient.occupation = patient_occupation
                else:
                    # Patient ID provided but not found, create new
                    patient = Patient(
                        name=patient_name or "Unknown",
                        age=patient_age,
                        gender=patient_gender,
                        smoker=patient_smoker or False,
                        chronic_conditions=patient_chronic_conditions,
                        occupation=patient_occupation
                    )
                    db.add(patient)
                    db.flush()  # Get the ID
                    patient_id = patient.id
            else:
                # Create new patient
                if patient_name:  # Only create if we have at least a name
                    patient = Patient(
                        name=patient_name,
                        age=patient_age,
                        gender=patient_gender,
                        smoker=patient_smoker or False,
                        chronic_conditions=patient_chronic_conditions,
                        occupation=patient_occupation
                    )
                    db.add(patient)
                    db.flush()
                    patient_id = patient.id
                else:
                    patient_id = None
            
            # Step 2: Create Visit record
            visit_id_str = state.get("visit_id")
            visit = Visit(
                visit_id=visit_id_str,
                patient_id=patient_id,
                symptoms=state.get("symptoms"),
                doctor_notes=state.get("doctor_note"),
                diagnosis=state.get("diagnosis"),
                vitals=json.dumps(state.get("vitals")) if state.get("vitals") else None,
                emergency_flag=state.get("emergency_flag", False),
                xray_result=json.dumps(state.get("xray_result")) if state.get("xray_result") else None,
                spirometry_result=json.dumps(state.get("spirometry_result")) if state.get("spirometry_result") else None,
                cbc_result=json.dumps(state.get("cbc_result")) if state.get("cbc_result") else None
            )
            db.add(visit)
            db.flush()  # Get the visit.id for Diagnosis foreign key
            
            # Step 3: Create Diagnosis record
            if state.get("diagnosis"):
                diagnosis = Diagnosis(
                    visit_id=visit.id,  # Use Visit.id (Integer), not visit_id (String)
                    diagnosis=state.get("diagnosis"),
                    treatment_plan=json.dumps(state.get("treatment_plan", [])),
                    tests_recommended=json.dumps(state.get("tests_recommended", [])),
                    home_remedies=json.dumps(state.get("home_remedies", [])),
                    followup_instruction=state.get("followup_instruction")
                )
                db.add(diagnosis)
            
            # Commit all changes
            db.commit()
            
            # Update state with saved IDs
            state["patient_id"] = patient_id
            state["visit_id"] = visit_id_str or str(visit.id)
            
            state["message"] = "Visit data saved successfully to database."
            print(f"✓ Visit saved: Patient ID={patient_id}, Visit ID={visit.id}")
            
        except Exception as e:
            db.rollback()
            print(f"Error saving to database: {e}")
            import traceback
            print(traceback.format_exc())
            # Don't fail the workflow, just warn
            state["message"] = "Visit processing completed, but there was an issue saving to database. Your session data is still available."
        finally:
            db.close()
    
    except ImportError as e:
        print(f"Error importing database modules: {e}")
        import traceback
        print(traceback.format_exc())
        state["message"] = "Visit processing completed, but database save is unavailable."
    except Exception as e:
        print(f"Error in history_saver_agent: {e}")
        import traceback
        print(traceback.format_exc())
        state["message"] = "Visit processing completed, but database save encountered an issue."
    
    return state


def followup_agent(state: AgentState) -> AgentState:
    """
    Follow-up Agent - compares current visit with previous visits.
    
    This agent:
    1. Retrieves previous visits from database (last 3 visits)
    2. Compares current vs previous: symptoms, test results, diagnosis, treatment
    3. Generates progress summary using LLM
    4. Suggests treatment modifications based on history
    5. Stores results in state
    """
    state["current_step"] = "followup_agent"
    
    # Only run if patient_id exists
    patient_id = state.get("patient_id")
    if not patient_id:
        state["previous_visits"] = None
        state["progress_summary"] = None
        state["message"] = "No patient ID available. Skipping follow-up analysis."
        return state
    
    try:
        from ..core.database import SessionLocal
        from ..db_models.visit import Visit
        from ..db_models.diagnosis import Diagnosis
        import json
        from datetime import datetime
        
        db = SessionLocal()
        
        try:
            # Step 1: Retrieve previous visits (last 3, excluding current visit)
            current_visit_id_str = state.get("visit_id")
            
            previous_visits_query = db.query(Visit).filter(
                Visit.patient_id == patient_id
            ).order_by(Visit.created_at.desc())
            
            # Exclude current visit if visit_id matches
            if current_visit_id_str:
                previous_visits_query = previous_visits_query.filter(
                    Visit.visit_id != current_visit_id_str
                )
            
            previous_visits_db = previous_visits_query.limit(3).all()
            
            if not previous_visits_db:
                state["previous_visits"] = None
                state["progress_summary"] = "This is the patient's first visit. No previous history available for comparison."
                state["message"] = "No previous visits found. This appears to be the first visit."
                return state
            
            # Step 2: Format previous visits data
            previous_visits_data = []
            for visit in previous_visits_db:
                # Get diagnosis for this visit
                diagnosis = db.query(Diagnosis).filter(Diagnosis.visit_id == visit.id).first()
                
                visit_data = {
                    "visit_id": visit.visit_id,
                    "date": visit.created_at.isoformat() if visit.created_at else None,
                    "symptoms": visit.symptoms,
                    "diagnosis": visit.diagnosis,
                    "doctor_notes": visit.doctor_notes,
                    "emergency_flag": visit.emergency_flag,
                    "xray_result": json.loads(visit.xray_result) if visit.xray_result else None,
                    "spirometry_result": json.loads(visit.spirometry_result) if visit.spirometry_result else None,
                    "cbc_result": json.loads(visit.cbc_result) if visit.cbc_result else None,
                }
                
                if diagnosis:
                    visit_data["treatment_plan"] = json.loads(diagnosis.treatment_plan) if diagnosis.treatment_plan else []
                    visit_data["home_remedies"] = json.loads(diagnosis.home_remedies) if diagnosis.home_remedies else []
                    visit_data["followup_instruction"] = diagnosis.followup_instruction
                else:
                    visit_data["treatment_plan"] = []
                    visit_data["home_remedies"] = []
                    visit_data["followup_instruction"] = None
                
                previous_visits_data.append(visit_data)
            
            state["previous_visits"] = previous_visits_data
            
            # Step 3: Prepare comparison data for LLM
            current_data = {
                "symptoms": state.get("symptoms"),
                "symptom_duration": state.get("symptom_duration"),
                "diagnosis": state.get("diagnosis"),
                "doctor_note": state.get("doctor_note"),
                "xray_result": state.get("xray_result"),
                "spirometry_result": state.get("spirometry_result"),
                "cbc_result": state.get("cbc_result"),
                "treatment_plan": state.get("treatment_plan", []),
                "home_remedies": state.get("home_remedies", []),
                "emergency_flag": state.get("emergency_flag", False)
            }
            
            # Format previous visits for LLM
            previous_visits_text = ""
            for i, prev_visit in enumerate(previous_visits_data, 1):
                days_ago = ""
                if prev_visit.get("date"):
                    try:
                        prev_date = datetime.fromisoformat(prev_visit["date"].replace('Z', '+00:00'))
                        days_diff = (datetime.now() - prev_date.replace(tzinfo=None)).days
                        days_ago = f" ({days_diff} days ago)"
                    except:
                        pass
                
                previous_visits_text += f"\n\nVisit {i}{days_ago}:\n"
                previous_visits_text += f"- Date: {prev_visit.get('date', 'Unknown')}\n"
                previous_visits_text += f"- Symptoms: {prev_visit.get('symptoms', 'Not recorded')}\n"
                previous_visits_text += f"- Diagnosis: {prev_visit.get('diagnosis', 'Not recorded')}\n"
                
                if prev_visit.get("xray_result"):
                    xray_pred = prev_visit["xray_result"].get("prediction", {})
                    previous_visits_text += f"- X-ray: {xray_pred.get('disease_name', 'Unknown')} (confidence: {xray_pred.get('confidence', 0):.1%})\n"
                
                if prev_visit.get("spirometry_result"):
                    spirometry_status = prev_visit["spirometry_result"].get("status", "Unknown")
                    previous_visits_text += f"- Spirometry: {spirometry_status}\n"
                
                if prev_visit.get("cbc_result"):
                    cbc_pred = prev_visit["cbc_result"].get("prediction", {})
                    previous_visits_text += f"- CBC: {cbc_pred.get('disease_name', 'Unknown')}\n"
                
                if prev_visit.get("treatment_plan"):
                    previous_visits_text += f"- Treatment: {', '.join(prev_visit['treatment_plan'])}\n"
            
            # Step 4: Generate progress summary using LLM
            messages = [
                {
                    "role": "system",
                    "content": """You are a pulmonologist analyzing patient visit history to track progress and provide continuity of care.

Analyze the current visit compared to previous visits and provide:
1. Overall progress assessment (improving, stable, worsening, or new condition)
2. Key changes since last visit (symptoms, test results, diagnosis)
3. Recurring issues or patterns
4. Treatment effectiveness analysis (if same condition)
5. Time between visits
6. Alerts for concerning patterns (e.g., frequent recurrences, worsening trends)
7. Treatment modification suggestions based on history

Be specific and evidence-based. Use medical terminology appropriately."""
                },
                {
                    "role": "user",
                    "content": f"""Patient ID: {patient_id}
Patient Age: {state.get('patient_age', 'Unknown')}
Patient Gender: {state.get('patient_gender', 'Unknown')}

CURRENT VISIT:
- Symptoms: {current_data.get('symptoms', 'Not specified')}
- Symptom Duration: {current_data.get('symptom_duration', 'Not specified')}
- Diagnosis: {current_data.get('diagnosis', 'Pending')}
- Doctor's Note: {current_data.get('doctor_note', 'Not available')}
- Emergency Flag: {'Yes' if current_data.get('emergency_flag') else 'No'}

Current Test Results:
- X-ray: {format_test_result_for_comparison(current_data.get('xray_result'), 'xray')}
- Spirometry: {format_test_result_for_comparison(current_data.get('spirometry_result'), 'spirometry')}
- CBC: {format_test_result_for_comparison(current_data.get('cbc_result'), 'cbc')}

Current Treatment Plan: {', '.join(current_data.get('treatment_plan', [])) if current_data.get('treatment_plan') else 'None'}

PREVIOUS VISITS (Last 3):{previous_visits_text}

Please provide:
1. Progress Summary (2-3 sentences)
2. Key Changes (bullet points)
3. Recurring Issues (if any)
4. Treatment Effectiveness Analysis (if applicable)
5. Treatment Modification Suggestions (based on history)
6. Alerts/Concerns (if any)

Format your response as a clear, structured summary suitable for medical records."""
                }
            ]
            
            try:
                progress_summary = call_groq_llm(messages, temperature=0.5)
                if progress_summary and progress_summary.strip():
                    state["progress_summary"] = progress_summary.strip()
                    state["message"] = "Follow-up analysis completed. Progress summary generated."
                else:
                    raise ValueError("Empty response from LLM")
            except Exception as llm_error:
                print(f"Warning: LLM failed in followup_agent: {llm_error}")
                import traceback
                print(traceback.format_exc())
                # Fallback summary
                state["progress_summary"] = "Progress analysis completed. Previous visits compared successfully."
                state["message"] = "Follow-up analysis completed."
            
        except Exception as e:
            print(f"Error in followup_agent database query: {e}")
            import traceback
            print(traceback.format_exc())
            state["previous_visits"] = None
            state["progress_summary"] = "Unable to retrieve previous visit history for comparison."
            state["message"] = "Current visit processed successfully."
        finally:
            db.close()
    
    except ImportError as e:
        print(f"Error importing database modules in followup_agent: {e}")
        import traceback
        print(traceback.format_exc())
        state["previous_visits"] = None
        state["progress_summary"] = None
        state["message"] = "Current visit processed successfully."
    except Exception as e:
        print(f"Error in followup_agent: {e}")
        import traceback
        print(traceback.format_exc())
        state["previous_visits"] = None
        state["progress_summary"] = None
        state["message"] = "Current visit processed successfully."
    
    return state


def format_test_result_for_comparison(test_result: Optional[Dict[str, Any]], test_type: str) -> str:
    """
    Format test result for comparison in LLM prompt.
    
    Args:
        test_result: Test result dictionary
        test_type: Type of test ('xray', 'spirometry', 'cbc')
        
    Returns:
        Formatted string
    """
    if not test_result:
        return "Not performed"
    
    if test_type == 'xray':
        prediction = test_result.get("prediction", {})
        disease = prediction.get("disease_name", "Unknown")
        confidence = prediction.get("confidence", 0)
        return f"{disease} (confidence: {confidence:.1%})"
    
    elif test_type == 'spirometry':
        status = test_result.get("status", "Unknown")
        prediction = test_result.get("prediction", {})
        details = []
        if prediction.get("obstruction"): details.append("Obstruction")
        if prediction.get("restriction"): details.append("Restriction")
        if prediction.get("prism"): details.append("PRISm")
        if prediction.get("mixed"): details.append("Mixed")
        details_str = f" ({', '.join(details)})" if details else ""
        return f"{status}{details_str}"
    
    elif test_type == 'cbc':
        prediction = test_result.get("prediction", {})
        disease = prediction.get("disease_name", "Unknown")
        confidence = prediction.get("confidence")
        if confidence is not None:
            return f"{disease} (confidence: {confidence:.1%})"
        return disease
    
    return "Available"


def check_patient_confirmation(state: AgentState) -> Literal["awaiting_confirmation", "confirmed", "error", "waiting"]:
    """Check if patient data is confirmed."""
    # Check for errors first
    if state.get("workflow_error") or state.get("current_step") == "error":
        return "error"
    
    # Check error count - if too high, route to error
    if state.get("error_count", 0) >= 3:
        return "error"
    
    # If we're waiting for initial user input (greeting shown, no conversation yet), route to END
    # This stops the workflow until user sends a message via /chat endpoint
    if state.get("current_step") == "patient_intake_waiting_input":
        return "waiting"  # Route to END to pause workflow
    
    if state.get("patient_data_confirmed", False):
        return "confirmed"
    if state.get("current_step") == "patient_intake_awaiting_confirmation":
        return "awaiting_confirmation"
    
    # If we're in retry state, allow one more attempt but limit it
    if state.get("current_step") == "patient_intake_retry":
        # Only allow retry if error_count is still low
        if state.get("error_count", 0) < 3:
            return "awaiting_confirmation"  # Route back to try again
        else:
            return "error"  # Too many errors, stop
    
    # If we have required data but not confirmed, check if we should ask for confirmation
    if state.get("patient_name") and state.get("patient_age") and state.get("symptoms"):
        return "awaiting_confirmation"
    
    # If we have conversation but no extracted data yet, continue to process
    conversation = state.get("conversation_history", [])
    if conversation and len(conversation) > 0:
        return "awaiting_confirmation"  # Process the conversation
    
    # Default: if no conversation and no data, we've shown greeting, proceed (will be handled by chat endpoint)
    return "confirmed"


def check_treatment_approval(state: AgentState) -> Literal["awaiting_approval", "approved"]:
    """Check if treatment plan is approved."""
    if state.get("treatment_approved", False):
        return "approved"
    if state.get("current_step") == "rag_specialist_awaiting_approval":
        return "awaiting_approval"
    # If we have treatment plan but not approved, need approval
    if state.get("treatment_plan") and len(state.get("treatment_plan", [])) > 0:
        return "awaiting_approval"
    return "approved"  # If no treatment yet, proceed (will be generated)


def check_emergency(state: AgentState) -> Literal["emergency", "continue"]:
    """Check if emergency flag is set."""
    if state.get("emergency_flag", False):
        return "emergency"
    return "continue"

