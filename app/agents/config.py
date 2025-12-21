"""
Configuration for LangGraph agents and LLM setup.
Supports both OpenAI and Groq with automatic preference (OpenAI > Groq).
"""
import os
import time
import random
import logging
from dotenv import load_dotenv
from typing import Optional, Literal
from groq import Groq
from ..core.error_handling import (
    LLMError, LLMTimeoutError, LLMRateLimitError, 
    LLMConnectionError, LLMInvalidResponseError,
    log_error_with_context
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# LLM Configuration - OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# LLM Configuration - Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# General Configuration
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Determine which LLM provider to use (preference: OpenAI > Groq)
LLM_PROVIDER: Optional[Literal["openai", "groq"]] = None

if OPENAI_API_KEY:
    LLM_PROVIDER = "openai"
    print("✅ Using OpenAI as LLM provider")
elif GROQ_API_KEY:
    LLM_PROVIDER = "groq"
    print("✅ Using Groq as LLM provider")
else:
    print("⚠️  WARNING: No LLM API key found!")
    print("   Please set either OPENAI_API_KEY or GROQ_API_KEY in .env file")
    print("   OpenAI: https://platform.openai.com/api-keys")
    print("   Groq: https://console.groq.com/")


def get_openai_client():
    """
    Get configured OpenAI client instance.
    
    Returns:
        OpenAI client instance
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
    
    return OpenAI(api_key=OPENAI_API_KEY)


def get_groq_client():
    """
    Get configured Groq client instance.
    
    Returns:
        Groq client instance
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env file.")
    
    return Groq(api_key=GROQ_API_KEY)


def call_llm(messages: list, temperature: float = None, provider: Optional[Literal["openai", "groq"]] = None, max_retries: int = 3, json_mode: bool = False, timeout: int = 60) -> str:
    """
    Call LLM with messages. Automatically uses OpenAI if available, otherwise Groq.
    Includes retry logic for rate limits with exponential backoff and comprehensive error handling.
    
    Args:
        messages: List of message dicts with "role" and "content" keys
        temperature: Temperature for generation (defaults to config value)
        provider: Force specific provider ("openai" or "groq"). If None, uses automatic selection.
        max_retries: Maximum number of retry attempts for rate limits (default: 3)
        json_mode: Whether to return response in JSON mode
        timeout: Request timeout in seconds (default: 60)
        
    Returns:
        Response content string
        
    Raises:
        LLMError: Base exception for LLM-related errors
        LLMTimeoutError: When request times out
        LLMRateLimitError: When rate limit is exceeded
        LLMConnectionError: When connection fails
        LLMInvalidResponseError: When response is invalid
    """
    # Use provided provider or auto-select
    selected_provider = provider or LLM_PROVIDER
    
    if not selected_provider:
        raise LLMConnectionError(
            "No LLM provider available. Please set either OPENAI_API_KEY or GROQ_API_KEY in .env file."
        )
    
    temp = temperature if temperature is not None else TEMPERATURE
    
    # Call OpenAI
    if selected_provider == "openai":
        if not OPENAI_API_KEY:
            raise LLMConnectionError("OPENAI_API_KEY not found. Cannot use OpenAI provider.")
        
        for attempt in range(max_retries):
            try:
                client = get_openai_client()
                params = {
                    "model": OPENAI_MODEL,
                    "messages": messages,
                    "temperature": temp,
                    "timeout": timeout,
                }
                if json_mode:
                    params["response_format"] = {"type": "json_object"}
                
                response = client.chat.completions.create(**params)
                
                if not response or not response.choices or not response.choices[0].message.content:
                    raise LLMInvalidResponseError("OpenAI returned empty or invalid response")
                
                return response.choices[0].message.content
                
            except TimeoutError as e:
                log_error_with_context(e, {"provider": "openai", "attempt": attempt + 1})
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"OpenAI timeout. Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    if GROQ_API_KEY:
                        logger.info("Falling back to Groq after OpenAI timeout...")
                        return call_llm(messages, temperature, provider="groq", max_retries=max_retries, json_mode=json_mode, timeout=timeout)
                    raise LLMTimeoutError("Request timed out. Please try again.")
                    
            except Exception as e:
                error_str = str(e).lower()
                log_error_with_context(e, {"provider": "openai", "attempt": attempt + 1})
                
                # Check if it's a rate limit error
                if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"OpenAI rate limit. Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"OpenAI rate limit exceeded after {max_retries} attempts. Falling back to Groq...")
                        if GROQ_API_KEY:
                            return call_llm(messages, temperature, provider="groq", max_retries=max_retries, json_mode=json_mode, timeout=timeout)
                        raise LLMRateLimitError("Rate limit exceeded. Please try again in a moment.")
                
                # Check for connection errors
                elif "connection" in error_str or "network" in error_str or "timeout" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"OpenAI connection error. Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        if GROQ_API_KEY:
                            logger.info("Falling back to Groq after OpenAI connection error...")
                            return call_llm(messages, temperature, provider="groq", max_retries=max_retries, json_mode=json_mode, timeout=timeout)
                        raise LLMConnectionError("Unable to connect to OpenAI. Please check your internet connection.")
                
                # Other errors - try fallback to Groq if available
                else:
                    logger.error(f"OpenAI API error: {e}")
                    if GROQ_API_KEY and attempt == max_retries - 1:
                        logger.info("Falling back to Groq after OpenAI error...")
                        return call_llm(messages, temperature, provider="groq", max_retries=max_retries, json_mode=json_mode, timeout=timeout)
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        time.sleep(wait_time)
                        continue
                    raise LLMError(f"OpenAI API error: {str(e)}")
        
        raise LLMError("Failed to get response after all retries")
    
    # Call Groq
    elif selected_provider == "groq":
        if not GROQ_API_KEY:
            raise LLMConnectionError("GROQ_API_KEY not found. Cannot use Groq provider.")
        
        for attempt in range(max_retries):
            try:
                client = get_groq_client()
                params = {
                    "model": GROQ_MODEL,
                    "messages": messages,
                    "temperature": temp,
                    "timeout": timeout,
                }
                if json_mode:
                    params["response_format"] = {"type": "json_object"}
                
                response = client.chat.completions.create(**params)
                
                if not response or not response.choices or not response.choices[0].message.content:
                    raise LLMInvalidResponseError("Groq returned empty or invalid response")
                
                return response.choices[0].message.content
                
            except TimeoutError as e:
                log_error_with_context(e, {"provider": "groq", "attempt": attempt + 1})
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(f"Groq timeout. Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise LLMTimeoutError("Request timed out. Please try again.")
                    
            except Exception as e:
                error_str = str(e).lower()
                log_error_with_context(e, {"provider": "groq", "attempt": attempt + 1})
                
                # Check if it's a rate limit error
                if "rate limit" in error_str or "429" in error_str or "quota" in error_str or "too many requests" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 0.5)
                        logger.warning(f"Groq rate limit. Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise LLMRateLimitError("Rate limit exceeded. The system is processing your request. Please wait a moment and try again.")
                
                # Check for connection errors
                elif "connection" in error_str or "network" in error_str or "timeout" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 0.5)
                        logger.warning(f"Groq connection error. Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise LLMConnectionError("Unable to connect to Groq. Please check your internet connection.")
                
                # Other errors
                else:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 0.5)
                        time.sleep(wait_time)
                        continue
                    raise LLMError(f"Groq API error: {str(e)}")
        
        raise LLMError("Failed to get response after all retries")
    
    else:
        raise ValueError(f"Unknown provider: {selected_provider}")


# Backward compatibility: Keep old function name
def call_groq_llm(messages: list, temperature: float = None, json_mode: bool = False) -> str:
    """
    Call LLM (backward compatibility wrapper).
    Now uses automatic provider selection (OpenAI > Groq).
    
    Args:
        messages: List of message dicts with "role" and "content" keys
        temperature: Temperature for generation (defaults to config value)
        json_mode: Whether to return response in JSON mode
        
    Returns:
        Response content string
    """
    return call_llm(messages, temperature, json_mode=json_mode)

