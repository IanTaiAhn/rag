from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_path: str = "../models/minilm_reranker"):
        self.model = CrossEncoder(model_path)

    def rerank(self, query: str, candidates: list, top_k: int = 5):
        pairs = [(query, c['metadata']['text']) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [r[0] for r in ranked[:top_k]]
