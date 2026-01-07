import os
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Local reranker using sentence-transformers CrossEncoder.
    """

    def __init__(self, model_name: str = None):
        # Default to MiniLM reranker
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        print(">>> Loading local reranker model:", self.model_name)

        # Load cross-encoder once at startup
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: list, top_k: int = 5):
        """
        Rerank candidates using a local cross-encoder.
        candidates: list of dicts with c["metadata"]["text"]
        """

        # Build (query, document) pairs
        pairs = [(query, c["metadata"]["text"]) for c in candidates]

        # Compute relevance scores
        scores = self.model.predict(pairs).tolist()

        # Sort by score descending
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top_k candidate objects
        return [r[0] for r in ranked[:top_k]]
