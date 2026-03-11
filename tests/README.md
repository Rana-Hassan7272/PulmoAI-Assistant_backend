# Test Suite

This directory contains comprehensive tests for the Doctor Assistant backend.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_agents/            # Agent function tests
│   ├── test_patient_intake.py
│   ├── test_emergency_detector.py
│   ├── test_supervisor.py
│   └── test_test_collector.py
├── test_api/               # API endpoint tests
│   ├── test_auth.py
│   ├── test_diagnostic.py
│   ├── test_imaging.py
│   ├── test_spirometry.py
│   └── test_lab_results.py
├── test_ml_models/         # ML model tests
│   ├── test_xray.py
│   ├── test_spirometry.py
│   └── test_bloodcount.py
├── test_database/          # Database operation tests
│   └── test_crud.py
└── test_rag/               # RAG system tests
    └── test_rag_agent.py
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=app --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_api/test_auth.py
```

### Run specific test class
```bash
pytest tests/test_agents/test_patient_intake.py::TestPatientIntakeAgent
```

### Run specific test function
```bash
pytest tests/test_api/test_auth.py::TestAuthEndpoints::test_login_success
```

### Run with verbose output
```bash
pytest -v
```

### Run only fast tests (skip slow/ML model tests)
```bash
pytest -m "not slow and not ml_model"
```

## Test Coverage

The test suite covers:
- ✅ Agent functions (patient intake, emergency detection, supervisor, test collector)
- ✅ API endpoints (auth, diagnostic, imaging, spirometry, lab results)
- ✅ ML model predictions (X-ray, spirometry, blood count)
- ✅ Database CRUD operations
- ✅ RAG system functionality

## Notes

- Some tests may skip if ML model files are not available (this is expected)
- Database tests use temporary SQLite databases that are cleaned up after each test
- API tests use FastAPI TestClient for fast, isolated testing
- Mock objects are used to avoid external API calls during testing
