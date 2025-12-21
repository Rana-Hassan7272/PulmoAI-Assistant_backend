from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from .core.init_db import init_db
from .core.error_handling import (
    LLMError, DatabaseError, PDFGenerationError, MLModelError,
    format_error_for_user, log_error_with_context
)
import logging
import traceback

from .fastapi_routers import patients, visits, lab_results, imaging, spirometry, diagnostic, rag, auth

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Doctor Assistant API",
    description="Medical diagnostic assistant with AI-powered analysis",
    version="1.0.0"
)

# CORS Configuration
# Allow requests from frontend (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React default
        "http://localhost:3001",  # Alternative React port
        "http://localhost:5173",  # Vite default
        "http://localhost:5174",  # Alternative Vite port
        "http://localhost:8080",  # Vue default
        "http://localhost:4200",  # Angular default
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        # Add production frontend URL here when deployed
        # "https://your-frontend-domain.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Global Exception Handlers
@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    """Handle LLM-related errors"""
    log_error_with_context(exc, {"path": request.url.path, "method": request.method})
    return JSONResponse(
        status_code=503,  # Service Unavailable
        content={
            "error": "AI Service Error",
            "message": format_error_for_user(exc, "processing your request"),
            "detail": "The AI service is temporarily unavailable. Please try again in a moment."
        }
    )

@app.exception_handler(DatabaseError)
async def database_error_handler(request: Request, exc: DatabaseError):
    """Handle database-related errors"""
    log_error_with_context(exc, {"path": request.url.path, "method": request.method})
    return JSONResponse(
        status_code=500,
        content={
            "error": "Database Error",
            "message": format_error_for_user(exc, "saving data"),
            "detail": "A database error occurred. Please try again."
        }
    )

@app.exception_handler(PDFGenerationError)
async def pdf_error_handler(request: Request, exc: PDFGenerationError):
    """Handle PDF generation errors"""
    log_error_with_context(exc, {"path": request.url.path, "method": request.method})
    return JSONResponse(
        status_code=500,
        content={
            "error": "PDF Generation Error",
            "message": format_error_for_user(exc),
            "detail": "Unable to generate PDF report. The visit data was saved successfully."
        }
    )

@app.exception_handler(MLModelError)
async def ml_model_error_handler(request: Request, exc: MLModelError):
    """Handle ML model errors"""
    log_error_with_context(exc, {"path": request.url.path, "method": request.method})
    return JSONResponse(
        status_code=500,
        content={
            "error": "AI Model Error",
            "message": format_error_for_user(exc, "analyzing test results"),
            "detail": "Unable to process test results. Please try again."
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Invalid request data. Please check your input.",
            "detail": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other unhandled exceptions"""
    log_error_with_context(exc, {
        "path": request.url.path,
        "method": request.method,
        "traceback": traceback.format_exc()
    })
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again or contact support.",
            "detail": "The error has been logged and will be investigated."
        }
    )

@app.on_event("startup")
def startup():
    try:
        init_db()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint for Docker/Kubernetes"""
    return {"status": "healthy", "service": "doctor-assistant-api"}

app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(visits.router)
app.include_router(lab_results.router)
app.include_router(imaging.router)
app.include_router(spirometry.router)
app.include_router(diagnostic.router)
app.include_router(rag.router)
