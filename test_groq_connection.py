"""
Minimal script to test Groq API connection and check for rate limits.
"""
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY not found in .env file")
    exit(1)

print(f"🔑 API Key: {GROQ_API_KEY[:10]}...{GROQ_API_KEY[-5:]}")
print(f"🤖 Model: {GROQ_MODEL}")
print("\n" + "="*50)
print("Testing Groq API connection...")
print("="*50 + "\n")

try:
    client = Groq(api_key=GROQ_API_KEY)
    
    # Simple test message
    messages = [
        {
            "role": "user",
            "content": "Say 'OK' if you can read this."
        }
    ]
    
    print("📤 Sending test request...")
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
    )
    
    result = response.choices[0].message.content
    print(f"✅ SUCCESS! Groq API is responding.")
    print(f"📥 Response: {result}")
    print(f"\n⏱️  Tokens used: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
    
except Exception as e:
    error_str = str(e).lower()
    
    if "rate limit" in error_str or "429" in error_str or "quota" in error_str or "too many requests" in error_str:
        print("⚠️  RATE LIMIT REACHED!")
        print(f"   Error: {str(e)}")
        print("\n💡 Solutions:")
        print("   1. Wait a few minutes and try again")
        print("   2. Check your Groq dashboard for quota limits")
        print("   3. Consider upgrading your Groq plan")
    elif "401" in error_str or "unauthorized" in error_str or "invalid" in error_str:
        print("❌ AUTHENTICATION ERROR!")
        print(f"   Error: {str(e)}")
        print("\n💡 Solutions:")
        print("   1. Check your GROQ_API_KEY in .env file")
        print("   2. Verify the API key is correct at https://console.groq.com/")
        print("   3. Make sure there are no extra spaces or quotes")
    else:
        print(f"❌ ERROR: {str(e)}")
        print(f"   Error type: {type(e).__name__}")

print("\n" + "="*50)

