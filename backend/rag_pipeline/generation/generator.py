# generation/generator.py
import os
from typing import List
OPENAI = os.getenv("OPENAI_API_KEY") is not None
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/rag_pipeline
MODEL_DIR = BASE_DIR / "models" / "qwen2.5"

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
        print("I'm using a locally loaded model")
        from transformers import pipeline
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # model_path = "../models/gpt2"
        model_path = MODEL_DIR

        # # GPT2 tokenizer and model
        # tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        # model = GPT2LMHeadModel.from_pretrained(model_path)

        # Load tokenizer + model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1
        )

        out = pipe(prompt, max_length=512, do_sample=False)
        # print('print from generator.py', out[0]["generated_text"])
        return out[0]['generated_text']