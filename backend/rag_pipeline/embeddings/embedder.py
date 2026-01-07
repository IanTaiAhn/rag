import os
from typing import List
from sentence_transformers import SentenceTransformer


class BaseEmbedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


# -------------------------
# Local SentenceTransformer embedder
# -------------------------
class LocalEmbedder(BaseEmbedder):
    """
    Uses a local sentence-transformers model for embeddings.
    """

    def __init__(self, model_name: str = None):
        # Default to MiniLM if not provided
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print(">>> Loading local embedding model:", self.model_name)

        # Load model once at startup
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        # Returns List[List[float]]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # optional but recommended for FAISS
        )

        return embeddings.tolist()


# -------------------------
# Embedder factory
# -------------------------
def get_embedder():
    """
    Always return the local embedder.
    HF API is no longer used for embeddings.
    """
    return LocalEmbedder()
