import os
import requests
from pathlib import Path

# OPENAI = os.getenv("OPENAI_API_KEY") is not None
HF_API_KEY = os.getenv("HF_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "qwen2.5"

# if OPENAI:
#     import openai


def generate_answer(prompt: str, max_tokens: int = 256):
    # -------------------------
    # 1. OpenAI path
    # -------------------------
    # if OPENAI:
    #     model = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4o-mini")
    #     res = openai.ChatCompletion.create(
    #         model=model,
    #         messages=[{"role": "user", "content": prompt}],
    #         max_tokens=max_tokens,
    #         temperature=0.0,
    #     )
    #     return res["choices"][0]["message"]["content"]

    # -------------------------
    # 2. Hugging Face API path
    # -------------------------
    if HF_API_KEY:
        model_name = os.getenv("HF_COMPLETION_MODEL", "Qwen/Qwen2.5-1.5B")
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"

        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.0,
                "return_full_text": False
            }
        }

        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"HuggingFace API error {response.status_code}: {response.text}"
            )

        data = response.json()

        # HF returns a list of dicts with "generated_text"
        return data[0]["generated_text"]

    # -------------------------
    # 3. Local fallback
    # -------------------------
    # print("Using locally loaded Qwen2.5 model")

    # from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

    # tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    # pipe = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=-1
    # )

    # out = pipe(prompt, max_length=max_tokens + len(prompt), do_sample=False)
    # return out[0]["generated_text"]
