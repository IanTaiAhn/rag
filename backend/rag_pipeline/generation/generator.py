import os
import requests
from pathlib import Path

HF_API_KEY = os.getenv("HF_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "qwen2.5"

def generate_answer(prompt: str, max_tokens: int = 256):
    if HF_API_KEY:
        model_name = os.getenv("HF_COMPLETION_MODEL", "Qwen/Qwen2.5-1.5B")

        api_url = "https://router.huggingface.co/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0
        }

        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"HuggingFace API error {response.status_code}: {response.text}"
            )

        data = response.json()

        return data["choices"][0]["message"]["content"]

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
