"""
Configuration for LangGraph agents and LLM setup.
Supports both OpenAI and Groq with automatic preference (OpenAI > Groq).
"""
import os
from dotenv import load_dotenv
from typing import Optional, Literal
from groq import Groq

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


def call_llm(messages: list, temperature: float = None, provider: Optional[Literal["openai", "groq"]] = None) -> str:
    """
    Call LLM with messages. Automatically uses OpenAI if available, otherwise Groq.
    
    Args:
        messages: List of message dicts with "role" and "content" keys
        temperature: Temperature for generation (defaults to config value)
        provider: Force specific provider ("openai" or "groq"). If None, uses automatic selection.
        
    Returns:
        Response content string
        
    Raises:
        ValueError: If no API key is configured
    """
    # Use provided provider or auto-select
    selected_provider = provider or LLM_PROVIDER
    
    if not selected_provider:
        raise ValueError(
            "No LLM provider available. Please set either OPENAI_API_KEY or GROQ_API_KEY in .env file."
        )
    
    temp = temperature if temperature is not None else TEMPERATURE
    
    # Call OpenAI
    if selected_provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found. Cannot use OpenAI provider.")
        
        try:
            client = get_openai_client()
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=temp,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️  OpenAI API call failed: {e}")
            # Fallback to Groq if available
            if GROQ_API_KEY:
                print("   Falling back to Groq...")
                return call_llm(messages, temperature, provider="groq")
            raise
    
    # Call Groq
    elif selected_provider == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found. Cannot use Groq provider.")
        
        client = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temp,
        )
        return response.choices[0].message.content
    
    else:
        raise ValueError(f"Unknown provider: {selected_provider}")


# Backward compatibility: Keep old function name
def call_groq_llm(messages: list, temperature: float = None) -> str:
    """
    Call LLM (backward compatibility wrapper).
    Now uses automatic provider selection (OpenAI > Groq).
    
    Args:
        messages: List of message dicts with "role" and "content" keys
        temperature: Temperature for generation (defaults to config value)
        
    Returns:
        Response content string
    """
    return call_llm(messages, temperature)

