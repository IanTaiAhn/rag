import os
from groq import Groq

# Load Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client once (recommended)
client = Groq(api_key=GROQ_API_KEY)

def generate_answer(prompt: str, max_tokens: int = 256):
    """
    Generate an answer using Groq's llama-3.3-70b-versatile model.
    """

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in the environment.")

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )

        return completion.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"Groq API error: {e}")
