# Backend Workflow Documentation

## Overview
The Doctor Assistant backend uses a **LangGraph-based agentic workflow** with a **Supervisor Agent** orchestrating specialized agents and tools. The system follows a modular, tool-based architecture.

## Architecture

### Core Components

1. **FastAPI Application** (`main.py`)
   - Entry point for the backend
   - Registers all routers
   - Handles CORS and database initialization

2. **LangGraph Workflow** (`agents/graph.py`)
   - Defines the agent workflow as a state machine
   - Manages routing between agents
   - Uses checkpointing for state persistence

3. **Supervisor Agent** (`agents/supervisor.py`)
   - Master orchestrator that decides which agent to call next
   - Uses LLM-based routing with rule-based fallback
   - Routes workflow based on current state

4. **Tool Functions** (`agents/tools.py`)
   - Modular, reusable functions for specific tasks
   - Can be called by any agent
   - Includes: dosage calculation, report generation, history retrieval, RAG treatment planning

## Workflow Flow

### 1. **Patient Intake** (`patient_intake_agent`)
   - **Purpose**: Collect patient information
   - **Input**: User provides name, age, gender, symptoms, medical history, etc.
   - **Process**:
     - Extracts information from conversation using LLM
     - Shows confirmation summary
     - Waits for patient confirmation
   - **Output**: Confirmed patient data in state
   - **Next**: Routes to Emergency Detector

### 2. **Emergency Detector** (`emergency_detector_agent`)
   - **Purpose**: Detect life-threatening conditions
   - **Input**: Patient symptoms and data
   - **Process**: Analyzes symptoms for emergency indicators
   - **Output**: Sets `emergency_flag` if emergency detected
   - **Next**: 
     - If emergency → END (stop workflow, alert user)
     - If no emergency → Supervisor

### 3. **Supervisor Agent** (`supervisor_agent`)
   - **Purpose**: Orchestrate workflow by deciding next step
   - **Process**: 
     - Analyzes current state
     - Uses LLM to decide next agent/tool to call
     - Falls back to rule-based routing if LLM fails
   - **Routes to**:
     - `doctor_note_generator` - If doctor note not generated
     - `test_collector` - If tests recommended but not collected
     - `rag_treatment_planner` - If tests complete but no treatment plan
     - `dosage_calculator` - If treatment approved but dosages not calculated
     - `report_generator` - If dosages calculated but no report
     - `history_saver` - If report ready but not saved
     - `end` - If everything complete

### 4. **Doctor Note Generator** (`doctor_note_generator_agent`)
   - **Purpose**: Generate clinical assessment and recommend tests
   - **Input**: Patient data, symptoms, medical history
   - **Process**:
     - Generates 2-3 sentence clinical assessment
     - Calls `recommend_tests()` to determine which tests are needed
     - Formats message with test recommendations
   - **Output**: 
     - `doctor_note`: Clinical assessment
     - `tests_recommended`: List of recommended tests (e.g., ["xray", "spirometry", "cbc"])
   - **Next**: Routes to Supervisor

### 5. **Test Collector** (`test_collector_agent`) - **NEW ReAct Agent**
   - **Purpose**: Collect diagnostic tests sequentially and interruptibly
   - **Input**: Recommended tests from state
   - **Process**:
     - Uses ReAct pattern (Reasoning + Acting)
     - Collects tests in order: X-ray → CBC → Spirometry
     - For each test:
       - Prompts user to provide data or skip
       - If data provided → Calls ML API tool
       - If skipped → Marks as skipped
     - Interruptible: Can pause and resume
   - **Tools Used**:
     - `_collect_xray_tool` - Handles X-ray upload or skip
     - `_call_xray_ml_api_tool` - Processes X-ray image with ML model
     - `_collect_cbc_tool` - Handles CBC form/data or skip
     - `_call_cbc_ml_api_tool` - Processes CBC data with ML model
     - `_collect_spirometry_tool` - Handles Spirometry form/data or skip
     - `_call_spirometry_ml_api_tool` - Processes Spirometry data with ML model
   - **Output**: 
     - Test results stored in state (`xray_result`, `cbc_result`, `spirometry_result`)
     - `test_collection_complete`: True when all tests collected/skipped
   - **Next**: 
     - If tests complete → Supervisor
     - If awaiting tests → END (wait for user input)

### 6. **RAG Treatment Planner** (`rag_specialist_agent` using `rag_treatment_planner_tool`)
   - **Purpose**: Generate diagnosis and treatment plan using RAG
   - **Input**: Clinical note, test results, patient data
   - **Process**:
     - Calls `rag_treatment_planner_tool()`:
       1. Formats test results for LLM
       2. Retrieves relevant medical knowledge using RAG
       3. Generates diagnosis and treatment plan using LLM
   - **Output**:
     - `diagnosis`: Primary diagnosis
     - `treatment_plan`: List of medications/treatments
     - `home_remedies`: List of home remedies
     - `followup_instruction`: Follow-up instructions
   - **Next**: Routes to Treatment Approval

### 7. **Treatment Approval** (`treatment_approval_agent`)
   - **Purpose**: Get patient approval for treatment plan
   - **Input**: Treatment plan from RAG specialist
   - **Process**:
     - Shows treatment plan to patient
     - Handles patient questions using RAG (for medication info)
     - Waits for approval
   - **Output**: 
     - `treatment_approved`: True when patient approves
   - **Next**: Routes to Dosage Calculator

### 8. **Dosage Calculator** (`dosage_calculator_agent` using `calculate_dosage_tool`)
   - **Purpose**: Calculate specific medication dosages
   - **Input**: Treatment plan, patient age, weight
   - **Process**:
     - Calls `calculate_dosage_tool()`:
       1. Extracts medications from treatment plan
       2. Uses LLM to calculate appropriate dosages
       3. Updates treatment plan with specific dosages
   - **Output**:
     - `calculated_dosages`: Dict with dose, frequency, duration for each medication
     - Updated `treatment_plan` with specific dosages
   - **Next**: Routes to Report Generator

### 9. **Report Generator** (`report_generator_agent` using `generate_final_report_tool`)
   - **Purpose**: Generate comprehensive final report
   - **Input**: All collected data (patient info, tests, diagnosis, treatment, dosages)
   - **Process**:
     - Calls `generate_final_report_tool()`:
       1. Formats all test results
       2. Uses LLM to generate comprehensive report
   - **Output**:
     - `final_report`: Complete medical report as string
   - **Next**: Routes to History Saver

### 10. **History Saver** (`history_saver_agent` using multiple tools)
   - **Purpose**: Save visit data to database
   - **Input**: Complete visit data
   - **Process**:
     - Calls `summarize_report_tool()` - Creates 2-3 line summary
     - Calls `generate_visit_id_tool()` - Generates unique visit ID
     - Calls `save_to_db_tool()` - Saves to database:
       1. Creates/updates Patient record
       2. Creates Visit record
       3. Creates Diagnosis record
   - **Output**:
     - `visit_id`: Unique visit identifier
     - `patient_id`: Patient database ID
     - `visit_summary`: 2-3 line summary for future reference
   - **Next**: Routes to END

### 11. **Follow-up Agent** (`followup_agent`)
   - **Purpose**: Compare current visit with previous visits
   - **Input**: Current visit data, previous visits from database
   - **Process**:
     - Retrieves last 3 visits from database
     - Compares current vs previous: symptoms, test results, diagnosis, treatment
     - Generates progress summary using LLM
   - **Output**:
     - `progress_summary`: Analysis of patient progress
   - **Next**: Routes to END

## State Management

### AgentState (`agents/state.py`)
- TypedDict that flows through the workflow
- Contains all patient data, test results, diagnosis, treatment, etc.
- Persisted using LangGraph checkpointing (MemorySaver or SqliteSaver)

### Key State Fields
- Patient info: `patient_id`, `patient_name`, `patient_age`, etc.
- Test results: `xray_result`, `spirometry_result`, `cbc_result`
- Workflow: `current_step`, `next_step`, `message`
- Flags: `patient_data_confirmed`, `treatment_approved`, `test_collection_complete`

## API Endpoints

### `/diagnostic/start` (POST)
- Starts new diagnostic session
- Returns initial message from Patient Intake Agent
- Creates new LangGraph thread with unique `visit_id`

### `/diagnostic/chat` (POST)
- Main endpoint for user interactions
- Accepts: `message` (user message), `visit_id` (session identifier)
- Optional: `xray_image` (file upload), `cbc_data`, `spirometry_data`
- Processes message through LangGraph workflow
- Returns: Updated state and assistant message

### `/diagnostic/upload-xray` (POST)
- Dedicated endpoint for X-ray image upload
- Accepts: `file` (image), `visit_id`
- Stores image in state for processing

## Tools Reference

All tools are in `agents/tools.py`:

1. **`format_test_results_for_llm(state)`**
   - Formats test results for LLM prompts
   - Returns formatted string

2. **`calculate_dosage_tool(state)`**
   - Calculates medication dosages
   - Returns: `{status, dosages, message}`

3. **`generate_final_report_tool(state)`**
   - Generates comprehensive final report
   - Returns: `{status, report, message}`

4. **`rag_treatment_planner_tool(state)`**
   - Generates treatment plan using RAG
   - Returns: `{status, diagnosis, treatment_plan, home_remedies, followup_instruction}`

5. **`summarize_report_tool(state)`**
   - Creates 2-3 line summary of visit
   - Returns: `{status, summary}`

6. **`generate_visit_id_tool(state)`**
   - Generates unique visit ID
   - Returns: `{status, visit_id}`

7. **`save_to_db_tool(state, visit_summary)`**
   - Saves visit to database
   - Returns: `{status, patient_id, visit_id, message}`

8. **`fetch_patient_history_tool(state)`**
   - Fetches last 3 visits for patient
   - Returns: `{status, visits, formatted_history}`

## ML Models

1. **X-ray Model** (`ml_models/xray/preprocessor.py`)
   - Predicts: No disease, Bacterial pneumonia, Viral pneumonia
   - Input: Chest X-ray image
   - Output: Disease classification with confidence

2. **Spirometry Model** (`ml_models/spirometry/featurizer.py`)
   - Predicts: Obstruction, Restriction, PRISm, Mixed patterns
   - Input: FEV1, FVC, age, gender, height, weight, BMI
   - Output: Pattern and severity classification

3. **CBC Model** (`ml_models/bloodcount_report/feature.py`)
   - Predicts: Blood disease classification
   - Input: WBC, RBC, HGB, HCT, PLT
   - Output: Disease classification with confidence

## RAG System

- **Location**: `agents/rag/`
- **Purpose**: Retrieve relevant medical knowledge for treatment planning
- **Components**:
  - `rag_agent.py`: Main RAG agent class
  - `retriever.py`: Vector store retrieval
  - `embeddings.py`: Text embeddings
  - `vector_store.py`: FAISS vector store
  - `document_loader.py`: PDF/text document loading
- **Knowledge Base**: Medical documents in `agents/rag/documents/`

## Database Models

- **Patient** (`db_models/patient.py`): Patient information
- **Visit** (`db_models/visit.py`): Visit records with test results
- **Diagnosis** (`db_models/diagnosis.py`): Diagnosis and treatment plans
- **User** (`db_models/user.py`): Authentication users

## Notes

- **`diagnostic_controller.py`**: Kept for backward compatibility but not actively used (replaced by `test_collector`)
- **`session_manager.py`**: DELETED - Not used (LangGraph handles checkpointing)
- All agents use tools from `tools.py` for modularity
- Supervisor uses LLM-based routing with rule-based fallback
- Test Collector uses ReAct pattern for interruptible test collection

