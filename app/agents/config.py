"""
Configuration for LangGraph agents and LLM setup.
"""
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Validate API key on import
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY not found in environment variables!")
    print("   Please set GROQ_API_KEY in .env file or environment variables.")
    print("   Get your API key from: https://console.groq.com/")
    print("   Example: Create .env file with: GROQ_API_KEY=your_api_key_here")


def get_groq_client():
    """
    Get configured Groq client instance.
    
    Returns:
        Groq client instance
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env file.")
    
    return Groq(api_key=GROQ_API_KEY)


def call_groq_llm(messages: list, temperature: float = None) -> str:
    """
    Call Groq LLM with messages.
    
    Args:
        messages: List of message dicts with "role" and "content" keys
        temperature: Temperature for generation (defaults to config value)
        
    Returns:
        Response content string
    """
    client = get_groq_client()
    temp = temperature if temperature is not None else TEMPERATURE
    
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temp,
    )
    
    return response.choices[0].message.content

