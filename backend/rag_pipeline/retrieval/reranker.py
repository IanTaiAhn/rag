import os
import requests
# from pathlib import Path
# from sentence_transformers import CrossEncoder

HF_API_KEY = os.getenv("HF_API_KEY")

# BASE_DIR = Path(__file__).resolve().parent.parent
# DEFAULT_MODEL_DIR = BASE_DIR / "models" / "minilm_reranker"


class Reranker:
    def __init__(self):
        # self.model_path = model_path
        self.hf_model = os.getenv(
            "HF_RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        # If HF key exists, use API. Otherwise load local model.
        if HF_API_KEY:
            self.use_hf = True
            self.api_url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
            self.headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        # else:
        #     self.use_hf = False
        #     self.model = CrossEncoder(str(model_path))

    def rerank(self, query: str, candidates: list, top_k: int = 5):
        pairs = [(query, c["metadata"]["text"]) for c in candidates]

        # -------------------------
        # Hugging Face API path
        # -------------------------
        if self.use_hf:
            payload = {
                "inputs": pairs,
                "parameters": {"truncate": True}
            }

            response = requests.post(self.api_url, headers=self.headers, json=payload)

            if response.status_code != 200:
                raise RuntimeError(
                    f"HuggingFace API error {response.status_code}: {response.text}"
                )

            scores = response.json()  # list of floats

        # -------------------------
        # Local fallback
        # -------------------------
        # else:
        #     scores = self.model.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [r[0] for r in ranked[:top_k]]
