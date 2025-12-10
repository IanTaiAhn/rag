# generation/generator.py
import os
from typing import List
OPENAI = os.getenv("OPENAI_API_KEY") is not None

if OPENAI:
    import openai


def generate_answer(prompt: str, max_tokens: int = 256):
    if OPENAI:
        model = os.getenv('OPENAI_COMPLETION_MODEL', 'gpt-4o-mini')
        res = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return res['choices'][0]['message']['content']
    else:
        # fallback: local small model using transformers (approx)
        print("I tried to use a small gpt-2")
        from transformers import pipeline
        pipe = pipeline('text-generation', model='gpt2', device=-1)
        out = pipe(prompt, max_length=512, do_sample=False)
        return out[0]['generated_text']