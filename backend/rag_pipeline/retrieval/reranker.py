import os
import requests


class Reranker:
    """
    Remote reranker using Jina AI's hosted Rerank API.
    """

    def __init__(self, model_name: str = None):
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY environment variable is required for Jina Reranker")

        # Default recommended model
        self.model_name = model_name or os.getenv(
            "JINA_RERANKER_MODEL",
            "jina-reranker-v1"
        )

        self.api_url = "https://api.jina.ai/v1/rerank"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def rerank(self, query: str, candidates: list, top_k: int = 5):
        """
        Rerank candidates using Jina's cross-encoder reranker.
        candidates: list of dicts with c["metadata"]["text"]
        """

        documents = [c["metadata"]["text"] for c in candidates]

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_k
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"Jina Reranker API error {response.status_code}: {response.text}"
            )

        data = response.json()

        # Jina returns: { "results": [ { "index": i, "relevance_score": x }, ... ] }
        results = data.get("results", [])

        # Build a score array aligned with original candidates
        scores = [0] * len(candidates)
        for item in results:
            idx = item["index"]
            scores[idx] = item["relevance_score"]

        # Sort candidates by score
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [r[0] for r in ranked[:top_k]]
