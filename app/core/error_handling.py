"""
Centralized Error Handling Module

Provides custom exceptions, error handlers, and utility functions for graceful error handling.
"""
import logging
import traceback
from typing import Optional, Dict, Any
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Custom Exceptions
class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM API call times out"""
    pass


class LLMRateLimitError(LLMError):
    """Raised when LLM API rate limit is exceeded"""
    pass


class LLMConnectionError(LLMError):
    """Raised when LLM API connection fails"""
    pass


class LLMInvalidResponseError(LLMError):
    """Raised when LLM returns invalid or unparseable response"""
    pass


class DatabaseError(Exception):
    """Base exception for database-related errors"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails"""
    pass


class DatabaseIntegrityError(DatabaseError):
    """Raised when database integrity constraint is violated"""
    pass


class FileOperationError(Exception):
    """Base exception for file operation errors"""
    pass


class PDFGenerationError(FileOperationError):
    """Raised when PDF generation fails"""
    pass


class MLModelError(Exception):
    """Base exception for ML model errors"""
    pass


class MLModelLoadError(MLModelError):
    """Raised when ML model fails to load"""
    pass


class MLModelPredictionError(MLModelError):
    """Raised when ML model prediction fails"""
    pass


def handle_llm_error(func):
    """Decorator to handle LLM errors gracefully"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LLMTimeoutError as e:
            logger.error(f"LLM timeout error in {func.__name__}: {e}")
            return {"status": "error", "message": "The AI service is taking longer than expected. Please try again in a moment."}
        except LLMRateLimitError as e:
            logger.error(f"LLM rate limit error in {func.__name__}: {e}")
            return {"status": "error", "message": "The system is currently busy. Please wait a moment and try again."}
        except LLMConnectionError as e:
            logger.error(f"LLM connection error in {func.__name__}: {e}")
            return {"status": "error", "message": "Unable to connect to AI service. Please check your internet connection and try again."}
        except LLMInvalidResponseError as e:
            logger.error(f"LLM invalid response error in {func.__name__}: {e}")
            return {"status": "error", "message": "Received an unexpected response. Please try again."}
        except LLMError as e:
            logger.error(f"LLM error in {func.__name__}: {e}")
            return {"status": "error", "message": "An error occurred while processing your request. Please try again."}
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}\n{traceback.format_exc()}")
            return {"status": "error", "message": "An unexpected error occurred. Please try again or contact support."}
    return wrapper


def handle_database_error(func):
    """Decorator to handle database errors gracefully"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DatabaseConnectionError as e:
            logger.error(f"Database connection error in {func.__name__}: {e}")
            return {"status": "error", "message": "Unable to connect to database. Please try again later."}
        except DatabaseIntegrityError as e:
            logger.error(f"Database integrity error in {func.__name__}: {e}")
            return {"status": "error", "message": "Data validation error. Please check your input and try again."}
        except DatabaseError as e:
            logger.error(f"Database error in {func.__name__}: {e}")
            return {"status": "error", "message": "A database error occurred. Please try again."}
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}\n{traceback.format_exc()}")
            return {"status": "error", "message": "An unexpected error occurred. Please try again."}
    return wrapper


def safe_execute(func, default_return=None, error_message="An error occurred"):
    """
    Safely execute a function and return default value on error.
    
    Args:
        func: Function to execute
        default_return: Value to return on error
        error_message: Error message to log
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func()
    except Exception as e:
        logger.error(f"{error_message}: {e}\n{traceback.format_exc()}")
        return default_return


def format_error_for_user(error: Exception, context: str = "") -> str:
    """
    Format error message for user display.
    
    Args:
        error: Exception that occurred
        context: Additional context about where error occurred
        
    Returns:
        User-friendly error message
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Map technical errors to user-friendly messages
    user_messages = {
        "LLMTimeoutError": "The AI service is taking longer than expected. Please try again in a moment.",
        "LLMRateLimitError": "The system is currently busy. Please wait a moment and try again.",
        "LLMConnectionError": "Unable to connect to AI service. Please check your internet connection.",
        "LLMInvalidResponseError": "Received an unexpected response. Please try again.",
        "DatabaseConnectionError": "Unable to connect to database. Please try again later.",
        "DatabaseIntegrityError": "Data validation error. Please check your input.",
        "PDFGenerationError": "Unable to generate PDF report. Please try again.",
        "MLModelLoadError": "Unable to load AI model. Please try again later.",
        "MLModelPredictionError": "Unable to process test results. Please try again.",
    }
    
    # Check if we have a user-friendly message
    if error_type in user_messages:
        return user_messages[error_type]
    
    # Generic fallback
    if context:
        return f"An error occurred while {context}. Please try again or contact support if the problem persists."
    return "An unexpected error occurred. Please try again or contact support if the problem persists."


def log_error_with_context(error: Exception, context: Dict[str, Any] = None):
    """
    Log error with context information.
    
    Args:
        error: Exception that occurred
        context: Additional context dictionary
    """
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc()
    }
    
    if context:
        error_info["context"] = context
    
    logger.error(f"Error occurred: {error_info}")

