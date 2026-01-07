import os
import requests
# from pathlib import Path
# from sentence_transformers import CrossEncoder

HF_API_KEY = os.getenv("HF_API_KEY")

# BASE_DIR = Path(__file__).resolve().parent.parent
# DEFAULT_MODEL_DIR = BASE_DIR / "models" / "minilm_reranker"

class Reranker:
    def __init__(self):
        self.hf_model = os.getenv(
            "HF_RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        if HF_API_KEY:
            self.use_hf = True
            self.api_url = "https://router.huggingface.co/rerank"
            self.headers = {
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            }
        # else:
        #     self.use_hf = False
        #     self.model = CrossEncoder(str(model_path))

    def rerank(self, query: str, candidates: list, top_k: int = 5):
        pairs = [(query, c["metadata"]["text"]) for c in candidates]

        if self.use_hf:
            payload = {
                "model": self.hf_model,
                "query": query,
                "documents": [c["metadata"]["text"] for c in candidates]
            }

            response = requests.post(self.api_url, headers=self.headers, json=payload)

            if response.status_code != 200:
                raise RuntimeError(
                    f"HuggingFace API error {response.status_code}: {response.text}"
                )

            data = response.json()

            # HF returns: {"results": [{"index": i, "score": float}, ...]}
            scores = [0] * len(candidates)
            for item in data["results"]:
                scores[item["index"]] = item["score"]

        # Local fallback (if you ever add it back)
        # else:
        #     scores = self.model.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [r[0] for r in ranked[:top_k]]
