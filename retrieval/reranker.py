# retrieval/reranker.py
# Placeholder for cross-encoder reranker using sentence-transformers
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list, top_k: int = 5):
        # candidates: list of dicts with metadata.text
        pairs = [(query, c['metadata']['text']) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [r[0] for r in ranked[:top_k]]