"""
FastAPI router for diagnostic workflow (LangGraph integration).
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
from ..schemas.diagnostic import DiagnosticMessage, DiagnosticResponse
from ..agents.graph import create_diagnostic_graph
from ..agents.state import AgentState
import uuid
from PIL import Image
import io

router = APIRouter(prefix="/diagnostic", tags=["Diagnostic Workflow"])

# Store graph instances
_graph_instance = None


def get_graph():
    """Get or create the diagnostic graph instance."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = create_diagnostic_graph()
    return _graph_instance


@router.post("/start", response_model=DiagnosticResponse)
def start_diagnostic(patient_id: Optional[int] = None):
    """
    Start a new diagnostic workflow session.
    
    Returns:
        Initial message from Patient Intake Agent
    """
    try:
        # Test graph creation first
        graph = get_graph()
        if graph is None:
            raise ValueError("Graph instance is None")
        
        # Initialize state
        initial_state: AgentState = {
            "patient_id": patient_id,
            "patient_name": None,
            "patient_age": None,
            "patient_gender": None,
            "patient_smoker": None,
            "patient_chronic_conditions": None,
            "patient_occupation": None,
            "visit_id": str(uuid.uuid4()),
            "symptoms": None,
            "symptom_duration": None,
            "patient_weight": None,
            "vitals": None,
            "emergency_flag": False,
            "emergency_reason": None,
            "doctor_note": None,
            "xray_result": None,
            "xray_available": False,
            "spirometry_result": None,
            "spirometry_available": False,
            "cbc_result": None,
            "cbc_available": False,
            "missing_tests": [],
            "rag_context": None,
            "diagnosis": None,
            "treatment_plan": None,
            "tests_recommended": None,
            "home_remedies": None,
            "followup_instruction": None,
            "previous_visits": None,
            "progress_summary": None,
            "conversation_history": [],
            "current_step": None,
            "message": None,
            "patient_data_confirmed": False,
            "treatment_approved": False,
            "treatment_modifications": None,
            "calculated_dosages": None,
            "error_count": 0,
            "workflow_error": None
        }
        
        # Use visit_id as thread_id for LangGraph checkpointing
        visit_id = initial_state.get("visit_id")
        if not visit_id:
            raise ValueError("Failed to generate visit_id")
        
        config = {
            "configurable": {"thread_id": visit_id},
            "recursion_limit": 50  # Increase recursion limit
        }
        
        # Run the graph (first step: patient_intake)
        # LangGraph automatically saves state at each step via checkpointer
        try:
            result = graph.invoke(initial_state, config=config)
        except Exception as invoke_error:
            import traceback
            print(f"Error in graph.invoke: {traceback.format_exc()}")
            raise invoke_error
        
        # Check for workflow errors
        if result.get("workflow_error"):
            message = result.get("message") or result.get("workflow_error")
            return DiagnosticResponse(
                message=message,
                current_step="error",
                patient_id=result.get("patient_id"),
                visit_id=result.get("visit_id"),
                emergency_flag=False,
                emergency_reason=None,
                state=result
            )
        
        # Ensure message is never None
        message = result.get("message") or "Welcome to Pulmonologist Assistant"
        if message is None:
            message = "Welcome to Pulmonologist Assistant"
        
        return DiagnosticResponse(
            message=message,
            current_step=result.get("current_step"),
            patient_id=result.get("patient_id"),
            visit_id=result.get("visit_id"),
            emergency_flag=result.get("emergency_flag", False),
            emergency_reason=result.get("emergency_reason"),
            state=result
        )
    
    except ValueError as e:
        # Validation errors
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except KeyError as e:
        # Missing required fields
        raise HTTPException(
            status_code=400,
            detail=f"Missing required field: {str(e)}"
        )
    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error starting diagnostic workflow: {error_trace}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        # Return more detailed error in development (you can remove detail in production)
        raise HTTPException(
            status_code=500,
            detail=f"We encountered an issue starting your diagnostic session: {str(e)}. Please check server logs for details."
        )


@router.post("/chat", response_model=DiagnosticResponse)
async def chat_with_agent(
    message: str = Form(...),
    patient_id: Optional[int] = Form(None),
    visit_id: Optional[str] = Form(None),
    xray_image: Optional[UploadFile] = File(None)
):
    """
    Continue the diagnostic workflow with a user message.
    
    This endpoint:
    1. Accepts text message and optional X-ray image upload
    2. Adds user message to conversation history
    3. Continues the LangGraph workflow from current step
    4. Returns agent response
    
    Parameters:
    - message: User's text message
    - patient_id: Optional patient ID
    - visit_id: Optional visit ID (for continuing conversation)
    - xray_image: Optional X-ray image file upload
    """
    try:
        graph = get_graph()
        
        # Process X-ray image if uploaded
        xray_image_data = None
        if xray_image:
            try:
                contents = await xray_image.read()
                
                # Validate file size (max 10MB)
                max_size = 10 * 1024 * 1024  # 10MB
                if len(contents) > max_size:
                    raise HTTPException(
                        status_code=400,
                        detail="X-ray image is too large. Maximum size is 10MB. Please upload a smaller image."
                    )
                
                # Validate it's an image
                if not xray_image.content_type or not xray_image.content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid file type. Please upload an image file (JPEG, PNG, etc.)."
                    )
                
                try:
                    image = Image.open(io.BytesIO(contents))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    xray_image_data = image
                    # Add to message that X-ray was uploaded
                    message += " [X-ray image uploaded and ready for analysis]"
                except Exception as img_error:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Could not process the image file. Please ensure it's a valid image format. Error: {str(img_error)}"
                    )
            except HTTPException:
                raise
            except Exception as e:
                print(f"Warning: Failed to process X-ray image: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process X-ray image. Please try uploading again or contact support if the problem persists."
                )
        
        # Get or create visit_id (used as thread_id for LangGraph)
        if not visit_id:
            visit_id = str(uuid.uuid4())
        
        # Use visit_id as thread_id for LangGraph checkpointing
        config = {
            "configurable": {"thread_id": visit_id},
            "recursion_limit": 50  # Increase recursion limit for chat interactions
        }
        
        # Get current state from LangGraph checkpointer
        try:
            # Try to get existing state from checkpointer
            state_snapshot = graph.get_state(config)
            if state_snapshot and state_snapshot.values:
                # Load existing state
                state = state_snapshot.values
            else:
                # No existing state, create new one
                state = None
        except Exception as e:
            # Log error but don't fail - we'll create new state
            print(f"Warning: Could not load state from checkpointer: {e}")
            state = None
        
        # If no existing state, create new one
        if state is None:
            state: AgentState = {
                "patient_id": patient_id,
                "visit_id": visit_id,
                "conversation_history": [],
                "patient_name": None,
                "patient_age": None,
                "patient_gender": None,
                "patient_smoker": None,
                "patient_chronic_conditions": None,
                "patient_occupation": None,
                "symptoms": None,
                "symptom_duration": None,
                "patient_weight": None,
                "vitals": None,
                "emergency_flag": False,
                "emergency_reason": None,
                "doctor_note": None,
                "xray_result": None,
                "xray_available": False,
                "spirometry_result": None,
                "spirometry_available": False,
                "cbc_result": None,
                "cbc_available": False,
                "missing_tests": [],
                "rag_context": None,
                "diagnosis": None,
                "treatment_plan": None,
                "tests_recommended": None,
                "home_remedies": None,
                "followup_instruction": None,
                "previous_visits": None,
                "progress_summary": None,
                "current_step": None,
                "message": None,
                "patient_data_confirmed": False,
                "treatment_approved": False,
                "treatment_modifications": None,
                "calculated_dosages": None,
                "error_count": 0,
                "workflow_error": None
            }
        else:
            # Update visit_id in existing state if needed
            state["visit_id"] = visit_id
        
        # Add user message to conversation history
        if "conversation_history" not in state:
            state["conversation_history"] = []
        state["conversation_history"].append({"role": "user", "content": message})
        
        # Update patient_id if provided
        if patient_id is not None:
            state["patient_id"] = patient_id
        
        # If X-ray image was uploaded, process it immediately and add result to conversation
        if xray_image_data:
            try:
                from ..ml_models.xray.preprocessor import predict_xray, predict_xray_proba
                
                try:
                    prediction = predict_xray(xray_image_data)
                    probabilities = predict_xray_proba(xray_image_data)
                except Exception as model_error:
                    print(f"Error in X-ray model prediction: {model_error}")
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to analyze X-ray image. The image may be corrupted or the model is unavailable. Please try uploading again."
                    )
                
                # Validate prediction results
                if not prediction or "class_name" not in prediction:
                    raise HTTPException(
                        status_code=500,
                        detail="X-ray analysis completed but results are invalid. Please try uploading the image again."
                    )
                
                # Add X-ray result to conversation so diagnostic controller can use it
                xray_result_text = f"X-ray analysis result: {prediction.get('class_name')} (confidence: {prediction.get('confidence', 0):.2%})"
                state["conversation_history"].append({
                    "role": "assistant",
                    "content": xray_result_text
                })
                
                # Also store in state for diagnostic controller to pick up
                state["xray_result"] = {
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
                state["xray_available"] = True
            except HTTPException:
                raise
            except Exception as e:
                print(f"Warning: Failed to process X-ray in chat endpoint: {e}")
                # Don't fail the entire request, just log the error
                # The diagnostic controller can handle missing X-ray data
        
        # Validate message is not empty
        if not message or not message.strip():
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty. Please provide your input."
            )
        
        # Run the graph with config (LangGraph automatically saves state via checkpointer)
        try:
            result = graph.invoke(state, config=config)
        except Exception as graph_error:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in graph execution: {error_trace}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your request. Please try again or contact support if the problem persists."
            )
        
        # Validate result
        if not result:
            raise HTTPException(
                status_code=500,
                detail="The diagnostic workflow did not return a result. Please try again."
            )
        
        # Ensure message is never None
        message = result.get("message") or "Processing..."
        if message is None:
            message = "Processing..."
        
        return DiagnosticResponse(
            message=message,
            current_step=result.get("current_step"),
            patient_id=result.get("patient_id"),
            visit_id=result.get("visit_id") or visit_id,
            emergency_flag=result.get("emergency_flag", False),
            emergency_reason=result.get("emergency_reason"),
            state=result
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        # Validation errors
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in chat endpoint: {error_trace}")
        
        # Return user-friendly message
        raise HTTPException(
            status_code=500,
            detail="We encountered an issue processing your message. Please try again or contact support if the problem persists."
        )


@router.get("/state/{visit_id}")
async def get_state(visit_id: str):
    """
    Get current state for a visit session using LangGraph checkpointer.
    
    Useful for debugging or frontend state management.
    """
    try:
        # Validate visit_id
        if not visit_id or not visit_id.strip():
            raise HTTPException(
                status_code=400,
                detail="Invalid visit_id. Please provide a valid visit ID."
            )
        
        graph = get_graph()
        config = {"configurable": {"thread_id": visit_id}}
        
        # Get state from LangGraph checkpointer
        try:
            state_snapshot = graph.get_state(config)
        except Exception as checkpoint_error:
            print(f"Error accessing checkpointer: {checkpoint_error}")
            raise HTTPException(
                status_code=500,
                detail="Unable to access session storage. Please try again later."
            )
        
        if not state_snapshot or not state_snapshot.values:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found for visit_id: {visit_id}. The session may have expired or never existed."
            )
        
        state = state_snapshot.values
        
        return {
            "visit_id": visit_id,
            "state": state,
            "current_step": state.get("current_step"),
            "patient_id": state.get("patient_id"),
            "checkpoint_id": state_snapshot.config.get("configurable", {}).get("checkpoint_id")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error getting state: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving session state. Please try again or contact support."
        )


@router.delete("/state/{visit_id}")
async def clear_session(visit_id: str):
    """
    Clear/delete a session from LangGraph checkpointer.
    
    Useful when a visit is completed or needs to be reset.
    Note: With MemorySaver, this clears the thread. For persistent storage,
    consider using SqliteSaver which supports deletion.
    """
    try:
        # Validate visit_id
        if not visit_id or not visit_id.strip():
            raise HTTPException(
                status_code=400,
                detail="Invalid visit_id. Please provide a valid visit ID."
            )
        
        graph = get_graph()
        config = {"configurable": {"thread_id": visit_id}}
        
        # Check if thread exists
        try:
            state_snapshot = graph.get_state(config)
        except Exception as checkpoint_error:
            print(f"Error accessing checkpointer: {checkpoint_error}")
            raise HTTPException(
                status_code=500,
                detail="Unable to access session storage. Please try again later."
            )
        
        if not state_snapshot or not state_snapshot.values:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found for visit_id: {visit_id}. The session may have already been deleted or never existed."
            )
        
        # Note: With SqliteSaver, we can implement proper deletion
        # For now, the session will remain in the database but can be cleaned up manually
        # In production, you might want to implement a cleanup job
        
        return {
            "message": f"Session {visit_id} found. Note: Session data is stored persistently. For complete deletion, database cleanup may be required.",
            "visit_id": visit_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error clearing session: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while clearing the session. Please try again or contact support."
        )

