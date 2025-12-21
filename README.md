# Doctor Assistant - Backend API

A comprehensive FastAPI backend for an AI-powered medical diagnostic assistant system with multi-agent orchestration, ML model integration, and RAG-based treatment planning.

## 🏗️ Architecture Overview

The backend is built using:
- **FastAPI** - Modern Python web framework
- **LangGraph** - Multi-agent workflow orchestration
- **SQLAlchemy** - Database ORM (SQLite)
- **Groq/OpenAI** - LLM providers for AI reasoning
- **PyTorch/Scikit-learn** - ML models for medical diagnosis

## 📁 Project Structure

```
backend/
├── app/
│   ├── agents/              # Multi-agent system
│   │   ├── graph.py         # LangGraph workflow definition
│   │   ├── supervisor.py    # Workflow orchestrator
│   │   ├── patient_intake.py # Patient data collection agent
│   │   ├── emergency_detector.py # Emergency triage agent
│   │   ├── test_collector.py # Diagnostic test collection agent
│   │   ├── rag/             # RAG system for medical knowledge
│   │   └── tools.py         # Shared tools for agents
│   ├── core/                # Core utilities
│   │   ├── database.py      # Database configuration
│   │   ├── init_db.py       # Database initialization
│   │   ├── migrations.py    # Database migrations
│   │   ├── auth.py          # Authentication
│   │   └── error_handling.py # Error handling system
│   ├── db_models/           # SQLAlchemy models
│   ├── fastapi_routers/     # API endpoints
│   ├── ml_models/           # ML model implementations
│   │   ├── xray/            # X-ray pneumonia detection
│   │   ├── spirometry/      # Lung function analysis
│   │   └── bloodcount_report/ # CBC blood test analysis
│   ├── schemas/             # Pydantic schemas
│   └── main.py              # FastAPI application entry
├── reports/                 # Generated PDF reports
├── data/                    # RAG vector store
├── requirements.txt         # Python dependencies
└── run_migration.py         # Database migration script
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- SQLite (included)
- Tesseract OCR (for PDF processing)
- ML model files (included in repo)

### Installation

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file:
```env
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
LLM_TEMPERATURE=0.7
```

4. **Initialize database:**
```bash
python run_migration.py
```

5. **Run the server:**
```bash
uvicorn app.main:app --reload --port 8000
```

## 🔌 API Endpoints

### Diagnostic Workflow
- `POST /diagnostic/start` - Start new diagnostic session
- `POST /diagnostic/chat` - Main chat endpoint (handles all workflow)

### Patient Management
- `GET /patients/{id}` - Get patient details
- `POST /patients/` - Create new patient

### Visit History
- `GET /visits/by_patient/{patient_id}` - Get patient visit history
- `GET /visits/{visit_id}/report` - Download PDF report

### Test Analysis
- `POST /imaging/analyze` - Analyze X-ray images
- `POST /spirometry/analyze` - Analyze spirometry data
- `POST /lab_results/analyze` - Analyze CBC blood tests

### RAG System
- `POST /rag/upload` - Upload medical documents
- `GET /rag/search` - Search medical knowledge base

### Authentication
- `POST /auth/register` - User registration
- `POST /auth/login` - User login

**Full API Documentation:** http://localhost:8000/docs

## 🤖 Multi-Agent System

The system uses **10 specialized agents** orchestrated by LangGraph:

1. **Patient Intake Agent** - Collects and validates patient information
2. **Emergency Detector** - Triage for life-threatening conditions
3. **Doctor Note Generator** - Creates clinical assessment
4. **Test Collector** - Sequentially collects diagnostic tests
5. **RAG Specialist** - Generates diagnosis using medical knowledge
6. **Treatment Approval** - Handles user approval workflow
7. **Report Generator** - Creates comprehensive medical reports
8. **History Saver** - Persists visit data to database
9. **Follow-up Agent** - Compares current visit with history
10. **Supervisor** - Orchestrates workflow sequence

### Workflow Sequence

```
Patient Intake → Emergency Check → Clinical Assessment → 
Test Collection → RAG Diagnosis → Treatment Approval → 
Report Generation → History Save → Follow-up Analysis
```

## 🧠 ML Models Integration

### 1. X-ray Analysis (Pneumonia Detection)
- **Model**: ResNet-50 trained on chest X-rays
- **Location**: `app/ml_models/xray/`
- **Output**: Disease classification (No disease, Bacterial, Viral) with confidence scores

### 2. Spirometry Analysis (Lung Function)
- **Model**: XGBoost ensemble (4 models for different patterns)
- **Location**: `app/ml_models/spirometry/`
- **Output**: Pattern (Normal, Obstruction, Restriction, Mixed) with severity

### 3. CBC Analysis (Blood Test)
- **Model**: XGBoost classifier
- **Location**: `app/ml_models/bloodcount_report/`
- **Output**: Disease prediction with confidence

## 📚 RAG System

The RAG (Retrieval-Augmented Generation) system:
- **Vector Store**: FAISS-based semantic search
- **Embeddings**: Sentence Transformers
- **Knowledge Base**: Medical guidelines, protocols, research papers
- **Location**: `app/agents/rag/`

Documents are automatically indexed and retrieved during diagnosis generation.

## 🗄️ Database Schema

### Tables
- **patients** - Patient demographic information
- **visits** - Visit records with symptoms, notes, test results
- **diagnosis** - Diagnosis, treatment plans, follow-up instructions
- **users** - User authentication

### Key Fields
- `visits.pdf_report_path` - Path to generated PDF reports
- `visits.xray_result`, `spirometry_result`, `cbc_result` - JSON test results
- `diagnosis.treatment_plan` - JSON array of treatments

## 🔧 Configuration

### LLM Provider Selection
The system automatically selects LLM provider:
1. **OpenAI** (if `OPENAI_API_KEY` is set)
2. **Groq** (if `GROQ_API_KEY` is set, fallback)

### Error Handling
Comprehensive error handling with:
- Retry logic for rate limits
- Automatic fallback between providers
- Graceful degradation
- User-friendly error messages

## 📊 State Management

The system uses `AgentState` (TypedDict) to maintain workflow state:
- Patient information
- Test results
- Diagnosis and treatment
- Conversation history
- Workflow flags

State is persisted via LangGraph checkpoints.

## 🐳 Docker Support

See `Dockerfile` for containerized deployment.

```bash
docker build -t doctor-assistant-backend .
docker run -p 8000:8000 doctor-assistant-backend
```

## 🧪 Testing

Run migrations:
```bash
python run_migration.py
```

Test API:
```bash
curl http://localhost:8000/health
```

## 📝 Key Features

- ✅ Multi-agent orchestration with LangGraph
- ✅ Three ML models for medical diagnosis
- ✅ RAG-based treatment planning
- ✅ PDF report generation
- ✅ Patient history tracking
- ✅ Progress comparison across visits
- ✅ Comprehensive error handling
- ✅ Database migrations
- ✅ Health checks
- ✅ API documentation (Swagger)

## 🔒 Security

- JWT-based authentication
- Password hashing (bcrypt)
- CORS configuration
- Input validation (Pydantic)
- SQL injection protection (SQLAlchemy)

## 📈 Performance

- Async FastAPI endpoints
- Efficient ML model loading
- Vector store caching
- Database connection pooling
- Health check monitoring

## 🐛 Troubleshooting

### Database Issues
```bash
python run_migration.py
```

### ML Model Loading
Ensure model files are in `app/ml_models/` directories.

### LLM API Errors
Check API keys in `.env` file. System will fallback automatically.

## 📚 Additional Documentation

- See root `README.md` for full project overview
- See `DOCKER_SETUP.md` for Docker deployment
- See `TESTING_GUIDE.md` for testing workflows

## 🤝 Contributing

This is a production-ready medical diagnostic system. Ensure all changes maintain:
- Error handling
- Type safety
- Database integrity
- Security best practices

---

**Version**: 1.0.0  
**Last Updated**: 2024
