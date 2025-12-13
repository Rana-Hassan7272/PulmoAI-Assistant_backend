# Doctor Assistant - Backend System

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Workflow](#workflow)
- [ML Models](#ml-models)
- [Database Schema](#database-schema)
- [RAG System](#rag-system)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## 🎯 Overview

The **Doctor Assistant** is an AI-powered diagnostic system designed to assist pulmonologists in patient diagnosis and treatment planning. The system uses a multi-agent LangGraph workflow to collect patient information, analyze medical tests (X-ray, Spirometry, CBC), generate diagnoses, and create treatment plans with medication dosage calculations.

### Key Capabilities

- **User Authentication**: Secure signup/login system with JWT tokens and automatic patient_id assignment
- **Intelligent Patient Intake**: Natural language processing to extract patient information
- **Emergency Detection**: Automatic detection of critical symptoms requiring immediate attention
- **ML-Powered Diagnostics**: Integration of deep learning models for X-ray, Spirometry, and CBC analysis
- **RAG-Enhanced Diagnosis**: Retrieval-Augmented Generation using medical knowledge base
- **Dosage Calculation**: Age and weight-based medication dosage calculations
- **Follow-up Analysis**: Comparison of current visit with previous visits for returning patients
- **Human-in-the-Loop**: Confirmation checkpoints for patient data and treatment approval
- **State Persistence**: Session management using LangGraph checkpointing
- **Multi-LLM Support**: Automatic provider selection (OpenAI preferred, Groq fallback)

---

## ✨ Features

### 1. **Multi-Agent Workflow**
   - Patient Intake Agent
   - Emergency Detector Agent
   - Doctor Note Generator Agent
   - Diagnostic Controller Agent
   - RAG Specialist Agent
   - Treatment Approval Agent (HITL)
   - Dosage Calculator Agent
   - Report Generator Agent
   - History Saver Agent
   - Follow-up Agent

### 2. **Human-in-the-Loop (HITL)**
   - **Patient Data Confirmation**: After initial intake, patient confirms their information
   - **Treatment Approval**: Patient reviews and approves/modifies treatment plan before finalization

### 3. **ML Model Integration**
   - **X-ray Analysis**: ResNet50-based pneumonia detection
   - **Spirometry Analysis**: XGBoost models for lung function patterns (obstruction, restriction, mixed)
   - **CBC Analysis**: Blood disease classification using XGBoost

### 4. **RAG System**
   - Vector-based retrieval from medical knowledge base
   - FAISS vector store with sentence-transformers embeddings
   - Medical document collection (PDFs, text files)
   - Context-aware diagnosis generation

### 5. **Dosage Calculator**
   - Hybrid approach: Rule-based + LLM-based calculations
   - Age and weight-based dosing
   - Supports multiple medication types
   - Automatic unit conversion (lbs to kg)

### 6. **Follow-up Analysis**
   - Retrieves last 3 previous visits for returning patients
   - Compares symptoms, test results, diagnoses, and treatments
   - Generates progress summary using LLM
   - Suggests treatment modifications based on history

### 7. **State Management**
   - LangGraph checkpointing with SQLite backend
   - Thread-based session persistence using `visit_id`
   - Automatic state recovery across API calls
   - Conversation history tracking

### 8. **Error Handling**
   - Retry mechanisms for LLM calls (up to 3 attempts)
   - Robust JSON parsing with fallback extraction
   - Graceful error handling to prevent infinite loops
   - Error tracking and workflow termination on critical failures

### 9. **Authentication & Security**
   - JWT-based authentication system
   - Secure password hashing with bcrypt
   - Automatic patient_id assignment on signup
   - Protected API endpoints
   - Token-based session management

### 10. **Multi-LLM Provider Support**
   - **OpenAI Integration**: GPT-3.5/GPT-4 support with automatic preference
   - **Groq Integration**: Fast inference as fallback option
   - **Automatic Selection**: Uses OpenAI if available, falls back to Groq
   - **Fallback Mechanism**: Auto-switches to Groq if OpenAI fails
   - **Flexible Configuration**: Easy to switch between providers via environment variables

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Patients   │  │    Visits    │  │  Diagnostic  │     │
│  │   Router     │  │    Router    │  │   Router     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              LangGraph Workflow (StateGraph)                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Patient Intake → Emergency Detector → Doctor Note   │   │
│  │  → Diagnostic Controller → RAG Specialist →         │   │
│  │  Treatment Approval → Dosage Calculator →           │   │
│  │  Report Generator → History Saver → Follow-up       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  ML Models   │  │  RAG System  │  │  Database    │
│  (X-ray,     │  │  (FAISS +    │  │  (SQLite)    │
│  Spirometry, │  │  Embeddings) │  │              │
│  CBC)        │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Agent State Flow

The `AgentState` (TypedDict) flows through all agents, accumulating information:

```python
AgentState {
    # Patient Information
    patient_id, patient_name, patient_age, patient_gender, 
    patient_smoker, patient_chronic_conditions, patient_occupation,
    patient_weight
    
    # Visit Information
    visit_id, symptoms, symptom_duration, vitals
    
    # Test Results
    xray_result, spirometry_result, cbc_result
    
    # Diagnosis & Treatment
    diagnosis, treatment_plan, calculated_dosages
    
    # Conversation & State
    conversation_history, current_step, message
    
    # HITL Flags
    patient_data_confirmed, treatment_approved
    
    # Error Handling
    error_count, workflow_error
}
```

---

## 🛠️ Technology Stack

### Core Framework
- **FastAPI**: Modern, fast web framework for building APIs
- **LangGraph**: Framework for building stateful, multi-agent workflows
- **LangChain**: LLM application framework
- **SQLAlchemy**: Python SQL toolkit and ORM
- **Pydantic**: Data validation using Python type annotations

### Machine Learning
- **PyTorch**: Deep learning framework (X-ray model)
- **XGBoost**: Gradient boosting framework (Spirometry, CBC)
- **scikit-learn**: Machine learning utilities
- **TensorFlow/Keras**: Keras 3 compatibility (sentence-transformers)

### RAG & Embeddings
- **sentence-transformers**: For generating embeddings
- **FAISS**: Vector similarity search library
- **PyPDF2**: PDF document processing

### LLM Providers
- **OpenAI API**: GPT-3.5/GPT-4 for agent intelligence (preferred)
- **Groq API**: Fast LLM inference as fallback option
- **Automatic Selection**: System automatically uses OpenAI if available, otherwise Groq

### Database
- **SQLite**: Lightweight database for development
- **langgraph-checkpoint-sqlite**: Persistent checkpointing for LangGraph

### Authentication & Security
- **python-jose**: JWT token generation and validation
- **passlib**: Secure password hashing (bcrypt)
- **OAuth2**: Token-based authentication

### Other Libraries
- **Pillow**: Image processing
- **OpenCV**: Computer vision
- **pytesseract**: OCR for text extraction
- **python-dotenv**: Environment variable management

---

## 📦 Setup & Installation

### Prerequisites

- Python 3.8+
- pip or conda
- Virtual environment (recommended)

### Installation Steps

1. **Clone the repository** (if not already done)
   ```bash
   cd Doctor-Assistant
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the `backend` directory:
   ```env
   # LLM Provider (choose one or both - OpenAI is preferred if both exist)
   OPENAI_API_KEY=sk-your_openai_api_key_here
   OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4, gpt-4-turbo, etc.
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   
   # Authentication
   JWT_SECRET_KEY=your-very-secure-random-secret-key-here
   
   # Database (optional)
   DATABASE_URL=sqlite:///./doctor_assistant.db
   ```
   
   **Generate a secure JWT secret key:**
   ```python
   import secrets
   print(secrets.token_urlsafe(32))
   ```

5. **Initialize the database**
   ```bash
   python -m app.core.init_db
   ```

6. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

The API will be available at `http://127.0.0.1:8000`

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | API key for OpenAI (preferred) | No* | - |
| `OPENAI_MODEL` | OpenAI model to use | No | `gpt-3.5-turbo` |
| `GROQ_API_KEY` | API key for Groq (fallback) | No* | - |
| `GROQ_MODEL` | Groq model to use | No | `llama-3.1-8b-instant` |
| `JWT_SECRET_KEY` | Secret key for JWT tokens | Yes | - |
| `LLM_TEMPERATURE` | Temperature for LLM generation | No | `0.7` |
| `DATABASE_URL` | Database connection string | No | SQLite default |

\* **At least one LLM API key is required** (OpenAI or Groq). If both are provided, OpenAI will be preferred.

### LLM Provider Selection

The system automatically selects the LLM provider based on available API keys:

1. **If `OPENAI_API_KEY` is set**: Uses OpenAI (preferred)
2. **If only `GROQ_API_KEY` is set**: Uses Groq
3. **If both are set**: Uses OpenAI (with Groq as fallback if OpenAI fails)
4. **If neither is set**: System will fail to start

### Obtaining API Keys

#### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and add it to your `.env` file

#### Groq API Key
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and add it to your `.env` file

### JWT Secret Key

Generate a secure secret key for JWT tokens:

```python
import secrets
print(secrets.token_urlsafe(32))
```

Add the generated key to your `.env` file as `JWT_SECRET_KEY`.

---

## 🔌 API Endpoints

### Authentication Endpoints

#### 1. Sign Up
```http
POST /auth/signup
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password",
  "name": "John Doe",        // Optional
  "age": 35,                 // Optional
  "gender": "male"           // Optional
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "patient_id": 1,
  "user_id": 1
}
```

**Note**: On signup, the system automatically:
- Creates a Patient record
- Assigns a unique `patient_id` to the user
- Returns JWT token for authentication

#### 2. Login
```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "patient_id": 1,
  "user_id": 1
}
```

#### 3. Get Current User Info
```http
GET /auth/me
Authorization: Bearer {token}
```

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "patient_id": 1
}
```

---

### Diagnostic Workflow (Protected - Requires Authentication)

**All diagnostic endpoints require JWT authentication. Include the token in the Authorization header:**

```http
Authorization: Bearer {your_jwt_token}
```

#### 1. Start Diagnostic Session
```http
POST /diagnostic/start
Authorization: Bearer {token}
```

**Note**: `patient_id` is automatically extracted from the JWT token. No need to provide it manually.

**Response:**
```json
{
  "message": "Welcome! I'm your Pulmonologist Assistant...",
  "current_step": "patient_intake",
  "visit_id": "abc-123-def-456",
  "patient_id": 1,
  "emergency_flag": false
}
```

#### 2. Chat with Agent
```http
POST /diagnostic/chat
Authorization: Bearer {token}
Content-Type: multipart/form-data

message: "My name is John, I'm 35 years old, male..."
visit_id: "abc-123-def-456"
xray_image: <file>  // Optional: X-ray image file
```

**Note**: `patient_id` is automatically extracted from the token.

**Response:**
```json
{
  "message": "Thank you for the information. Please confirm...",
  "current_step": "patient_intake",
  "visit_id": "abc-123-def-456",
  "patient_id": 1,
  "emergency_flag": false
}
```

#### 3. Get Current State
```http
GET /diagnostic/state/{visit_id}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "patient_name": "John Doe",
  "patient_age": 35,
  "symptoms": "cough, fever",
  "diagnosis": "Upper respiratory infection",
  "treatment_plan": ["Medication A", "Medication B"],
  "current_step": "treatment_approval"
}
```

#### 4. Delete Session
```http
DELETE /diagnostic/delete/{visit_id}
Authorization: Bearer {token}
```

### Other Endpoints

- **Patients**: `/patients/*` - Patient CRUD operations
- **Visits**: `/visits/*` - Visit management
- **Imaging**: `/imaging/*` - X-ray image processing
- **Spirometry**: `/spirometry/*` - Spirometry test processing
- **Lab Results**: `/lab_results/*` - CBC and other lab results
- **RAG**: `/rag/*` - RAG system endpoints

### API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

---

## 🔄 Workflow

### Complete Diagnostic Flow

```
1. START
   ↓
2. Patient Intake Agent
   - Collects: name, age, gender, weight, symptoms, medical history
   - Asks for confirmation (HITL)
   ↓
3. Emergency Detector Agent
   - Checks for critical symptoms
   - Routes to END if emergency detected
   ↓
4. Doctor Note Generator Agent
   - Creates clinical note summarizing patient information
   ↓
5. Diagnostic Controller Agent
   - Recommends tests (X-ray, Spirometry, CBC)
   - Processes test results using ML models
   ↓
6. RAG Specialist Agent
   - Retrieves relevant medical knowledge
   - Generates diagnosis and treatment plan
   ↓
7. Treatment Approval Agent (HITL)
   - Presents treatment plan to patient
   - Allows questions and modifications
   ↓
8. Dosage Calculator Agent
   - Calculates medication dosages based on age/weight
   - Updates treatment plan with specific dosages
   ↓
9. Report Generator Agent
   - Creates comprehensive final report
   ↓
10. History Saver Agent
    - Saves patient, visit, and diagnosis to database
    ↓
11. Follow-up Agent
    - Retrieves previous visits (if returning patient)
    - Compares current vs previous
    - Generates progress summary
    ↓
12. END
```

### State Persistence

- Each workflow step saves state to LangGraph checkpointer
- State is keyed by `visit_id` (used as `thread_id`)
- State persists across API calls
- Automatic recovery on subsequent requests

---

## 🤖 ML Models

### 1. X-ray Analysis (`app/ml_models/xray/`)

**Model**: ResNet50 fine-tuned for pneumonia detection

**Input**: X-ray image (chest X-ray)

**Output**:
```json
{
  "prediction": {
    "disease_name": "Pneumonia" | "No disease",
    "confidence": 0.95
  }
}
```

**Usage**:
```python
from app.ml_models.xray.preprocessor import predict_xray

result = predict_xray(image_path)
```

### 2. Spirometry Analysis (`app/ml_models/spirometry/`)

**Models**: XGBoost models for pattern classification

**Input**: Spirometry values (FEV1, FVC, FEV1/FVC ratio)

**Output**:
```json
{
  "pattern": "obstruction" | "restriction" | "mixed" | "normal",
  "severity": "mild" | "moderate" | "severe",
  "confidence": 0.92
}
```

**Usage**:
```python
from app.ml_models.spirometry.featurizer import predict_spirometry

result = predict_spirometry(fev1=2.1, fvc=2.8, fev1_fvc=0.75)
```

### 3. CBC Analysis (`app/ml_models/bloodcount_report/`)

**Model**: XGBoost for blood disease classification

**Input**: CBC values (WBC, RBC, Hemoglobin, etc.)

**Output**:
```json
{
  "prediction": "Anemia" | "Infection" | "Normal" | ...,
  "confidence": 0.88
}
```

**Usage**:
```python
from app.ml_models.bloodcount_report.feature import predict_blood_disease

result = predict_blood_disease(cbc_values_dict)
```

---

## 🗄️ Database Schema

### Tables

#### 1. **Patient**
```sql
- id (Primary Key)
- name
- age
- gender
- smoker (boolean)
- chronic_conditions
- occupation
- created_at
- updated_at
```

#### 2. **User**
```sql
- id (Primary Key)
- email (Unique)
- password_hash
- patient_id (Foreign Key → Patient.id, Unique)
- created_at
```

#### 3. **Visit**
```sql
- id (Primary Key)
- patient_id (Foreign Key → Patient.id)
- visit_id (Unique identifier for LangGraph thread)
- symptoms
- symptom_duration
- xray_result (JSON)
- spirometry_result (JSON)
- cbc_result (JSON)
- created_at
- updated_at
```

#### 4. **Diagnosis**
```sql
- id (Primary Key)
- visit_id (Foreign Key → Visit.id)
- diagnosis (text)
- treatment_plan (JSON array)
- followup_instruction
- created_at
```

#### 5. **Imaging**
```sql
- id (Primary Key)
- visit_id (Foreign Key → Visit.id)
- image_path
- prediction_result (JSON)
- created_at
```

#### 6. **Lab_Results**
```sql
- id (Primary Key)
- visit_id (Foreign Key → Visit.id)
- test_type
- test_result (JSON)
- created_at
```

---

## 📚 RAG System

### Overview

The RAG (Retrieval-Augmented Generation) system enhances diagnosis generation by retrieving relevant medical knowledge from a curated document collection.

### Components

1. **Document Loader** (`app/agents/rag/document_loader.py`)
   - Loads PDF and text documents
   - Extracts text content

2. **Embeddings** (`app/agents/rag/embeddings.py`)
   - Generates embeddings using `sentence-transformers`
   - Model: `all-MiniLM-L6-v2`

3. **Vector Store** (`app/agents/rag/vector_store.py`)
   - FAISS index for similarity search
   - Persistent storage in `data/rag_index/`

4. **Retriever** (`app/agents/rag/retriever.py`)
   - Retrieves top-k relevant documents
   - Returns context for LLM

5. **RAG Agent** (`app/agents/rag/rag_agent.py`)
   - Integrates retrieval with LLM generation
   - Used by RAG Specialist Agent

### Medical Documents

Located in `app/agents/rag/documents/`:
- Asthma treatment guidelines.pdf
- COPD management protocols.pdf
- pneumonia-diagnosis-and-management.pdf
- clinical-protocols.pdf
- Drug interaction guides.pdf
- tb_diagnosis_treatment.txt
- research.pdf

### Usage

```python
from app.agents.rag.rag_agent import get_rag_agent

rag_agent = get_rag_agent()
context = rag_agent.retrieve("patient with cough and fever")
```

---

## ⚠️ Error Handling

### Error Tracking

The system implements robust error handling:

1. **Error Count**: Tracks consecutive failures (`error_count`)
2. **Workflow Error**: Stores error message (`workflow_error`)
3. **Retry Logic**: Up to 3 attempts for LLM calls
4. **Graceful Degradation**: Falls back to default behavior on failures
5. **JSON Parsing**: Robust extraction even with malformed LLM responses

### Error Prevention

- **Recursion Limit**: Set to 50 to prevent infinite loops
- **Conditional Routing**: Error paths route to END node
- **State Validation**: Validates state before processing
- **Exception Handling**: Try-catch blocks in all agents

### Error Response Format

```json
{
  "detail": "Error message here",
  "error_type": "LLMError" | "ValidationError" | "DatabaseError"
}
```

---

## 🧪 Testing

### Manual Testing

1. **Start the server**
   ```bash
   uvicorn app.main:app --reload
   ```

2. **Use Swagger UI**
   - Navigate to `http://127.0.0.1:8000/docs`
   - Test endpoints interactively

3. **Use cURL**
   ```bash
   # Sign up
   curl -X POST "http://127.0.0.1:8000/auth/signup" \
        -H "Content-Type: application/json" \
        -d '{"email": "test@example.com", "password": "test123"}'
   
   # Login (save the token from response)
   curl -X POST "http://127.0.0.1:8000/auth/login" \
        -H "Content-Type: application/json" \
        -d '{"email": "test@example.com", "password": "test123"}'
   
   # Start diagnostic session (use token from login)
   curl -X POST "http://127.0.0.1:8000/diagnostic/start" \
        -H "Authorization: Bearer YOUR_TOKEN_HERE"
   
   # Chat
   curl -X POST "http://127.0.0.1:8000/diagnostic/chat" \
        -H "Authorization: Bearer YOUR_TOKEN_HERE" \
        -F "message=My name is John, age 35" \
        -F "visit_id=abc-123-def-456"
   ```

### Testing Workflow

1. **Sign up** or **login** to get JWT token
2. **Start a new diagnostic session** (token automatically provides patient_id)
3. Provide patient information
4. Confirm patient data (HITL checkpoint)
5. Provide test results (X-ray, Spirometry, CBC)
6. Review treatment plan
7. Approve/modify treatment (HITL checkpoint)
8. Review final report
9. Check database for saved data

### Example Test Case

See `COMPLETE_WORKFLOW_EXAMPLE.md` for a detailed end-to-end example.

---

## 📁 Project Structure

```
backend/
├── app/
│   ├── agents/
│   │   ├── config.py              # LLM configuration
│   │   ├── state.py               # AgentState TypedDict
│   │   ├── graph.py               # LangGraph workflow definition
│   │   ├── patient_intake.py      # Patient intake agent
│   │   ├── emergency_detector.py  # Emergency detection agent
│   │   ├── diagnostic_controller.py # Test orchestration agent
│   │   ├── dosage_calculator.py   # Dosage calculation agent
│   │   ├── session_manager.py    # (Legacy, not used)
│   │   └── rag/
│   │       ├── document_loader.py
│   │       ├── embeddings.py
│   │       ├── vector_store.py
│   │       ├── retriever.py
│   │       ├── rag_agent.py
│   │       └── documents/        # Medical knowledge base
│   ├── core/
│   │   ├── database.py            # SQLAlchemy setup
│   │   └── init_db.py             # Database initialization
│   ├── db_models/
│   │   ├── user.py              # User authentication model
│   │   ├── patient.py
│   │   ├── visit.py
│   │   ├── diagnosis.py
│   │   ├── imaging.py
│   │   └── lab_results.py
│   ├── core/
│   │   ├── database.py            # SQLAlchemy setup
│   │   ├── init_db.py             # Database initialization
│   │   └── auth.py                # Authentication utilities (JWT, password hashing)
│   ├── fastapi_routers/
│   │   ├── auth.py                # Authentication endpoints (signup, login, me)
│   │   ├── diagnostic.py          # Main diagnostic endpoints (protected)
│   │   ├── patients.py
│   │   ├── visits.py
│   │   ├── imaging.py
│   │   ├── spirometry.py
│   │   ├── lab_results.py
│   │   └── rag.py
│   ├── schemas/
│   │   ├── auth.py                # Authentication schemas
│   │   ├── patient.py
│   │   ├── visit.py
│   │   ├── diagnosis.py
│   │   ├── diagnostic.py
│   │   ├── imaging.py
│   │   ├── spirometry.py
│   │   └── lab_results.py
│   ├── ml_models/
│   │   ├── xray/
│   │   │   ├── preprocessor.py
│   │   │   └── pneumonia_resnet50.pth
│   │   ├── spirometry/
│   │   │   ├── featurizer.py
│   │   │   └── *.pkl (XGBoost models)
│   │   └── bloodcount_report/
│   │       ├── feature.py
│   │       └── *.pkl (XGBoost models)
│   ├── schemas/
│   │   ├── patient.py
│   │   ├── visit.py
│   │   ├── diagnosis.py
│   │   ├── diagnostic.py
│   │   ├── imaging.py
│   │   ├── spirometry.py
│   │   └── lab_results.py
│   └── main.py                    # FastAPI application entry point
├── data/
│   └── rag_index/                 # FAISS vector store
├── requirements.txt
├── README.md                      # This file
└── COMPLETE_WORKFLOW_EXAMPLE.md   # Detailed workflow example
```

---

## 🔧 Key Implementation Details

### LangGraph Checkpointing

- Uses `MemorySaver` (development) or `SqliteSaver` (production)
- State persisted using `visit_id` as `thread_id`
- Automatic state recovery on each API call
- Supports long-running workflows

### Human-in-the-Loop (HITL)

1. **Patient Data Confirmation**
   - After patient intake, system asks: "Please confirm your information..."
   - User responds: "Yes" → proceed, "No" → loop back

2. **Treatment Approval**
   - After RAG specialist generates treatment, system presents it
   - User can ask questions or request modifications
   - User approves → proceed to dosage calculation

### Dosage Calculation

- **Rule-based**: For common medications (e.g., weight-based dosing)
- **LLM-based**: For complex or uncommon medications
- Supports: mg/kg, mg/day, frequency adjustments
- Handles unit conversions (lbs → kg)

### Follow-up Agent Logic

- Only runs if `patient_id` exists
- Retrieves last 3 previous visits
- Compares:
  - Symptoms (new, improved, worsening, recurring)
  - Test results (trends, changes)
  - Diagnoses (same, new, resolved, evolved)
  - Treatments (effectiveness, changes)
- Generates progress summary using LLM

### Authentication Flow

1. **Signup**: User creates account → System creates Patient record → Assigns `patient_id` → Returns JWT token
2. **Login**: User authenticates → Returns JWT token with `patient_id`
3. **Protected Endpoints**: All diagnostic endpoints require JWT token
4. **Automatic patient_id**: Token contains `patient_id`, automatically used in diagnostic workflow

### LLM Provider Selection

The system uses a smart provider selection mechanism:

1. **Startup Check**: Checks for `OPENAI_API_KEY` and `GROQ_API_KEY`
2. **Priority**: OpenAI > Groq (if both available)
3. **Runtime Fallback**: If OpenAI call fails, automatically falls back to Groq
4. **Transparent**: All agents use `call_llm()` which handles provider selection automatically

**Example Configuration:**
```env
# Use OpenAI (preferred)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo

# Or use Groq (fallback)
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.1-8b-instant

# Or both (OpenAI preferred, Groq as fallback)
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

---

## 🚀 Future Enhancements

- [x] Authentication and authorization ✅
- [x] Multi-LLM provider support (OpenAI + Groq) ✅
- [ ] Rate limiting
- [ ] Comprehensive unit and integration tests
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Monitoring and logging (Prometheus, Grafana)
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Real-time chat (WebSockets)
- [ ] Integration with external EHR systems
- [ ] HIPAA compliance features
- [ ] Advanced analytics dashboard

---

## 📝 Notes

- The system includes **authentication and security** features (JWT tokens, password hashing)
- For production use, ensure:
  - Strong `JWT_SECRET_KEY` is set (use `secrets.token_urlsafe(32)`)
  - HTTPS is enabled
  - Rate limiting is implemented
  - Proper error logging and monitoring
  - HIPAA compliance measures (if handling real patient data)
- ML models should be regularly updated and validated
- RAG knowledge base should be periodically updated with latest medical guidelines
- Database should be backed up regularly
- **LLM Provider**: System automatically selects best available provider (OpenAI preferred, Groq fallback)

---

## 🤝 Contributing

This is a project for learning and development. Contributions and improvements are welcome!



---

## 👤 Author

MUHAMMAD HASSAN SHAHBAZ

---

**Last Updated**: 12-DEC-2025

# PulmoAI-Assistant_backend
