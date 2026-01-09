import os
import requests
from typing import List

class BaseEmbedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class JinaEmbedder(BaseEmbedder):
    """
    Uses Jina AI's hosted Embeddings API.
    Docs: https://docs.jina.ai/embeddings
    """

    def __init__(self, model_name: str = None):
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY environment variable is required for JinaEmbedder")

        # Default model (recommended)
        self.model_name = model_name or os.getenv(
            "JINA_EMBEDDING_MODEL",
            "jina-embeddings-v3"
        )

        self.api_url = "https://api.jina.ai/v1/embeddings"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def embed(self, texts: List[str]):
        """
        Sends a batch of texts to Jina's embedding endpoint.
        Returns a list of embedding vectors.
        """

        payload = {
            "model": self.model_name,
            "input": texts
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"Jina API error {response.status_code}: {response.text}"
            )

        data = response.json()

        # Jina returns: { "data": [ { "embedding": [...] }, ... ] }
        embeddings = [item["embedding"] for item in data["data"]]

        return embeddings


# -------------------------
# Embedder factory
# -------------------------
def get_embedder():
    """
    Returns the Jina embedder if JINA_API_KEY is set.
    """
    if os.getenv("JINA_API_KEY"):
        return JinaEmbedder()

    raise RuntimeError("No embedding backend configured. Set JINA_API_KEY.")




