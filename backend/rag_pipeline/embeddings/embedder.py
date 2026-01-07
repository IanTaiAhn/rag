import os
from typing import List
import requests

HF_API_KEY = os.getenv("HF_API_KEY")


class BaseEmbedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


# -------------------------
# Hugging Face API embedder
# -------------------------
class HuggingFaceEmbedder(BaseEmbedder):
    """
    Uses Hugging Face Inference API for embeddings.
    """

    def __init__(self, model_name: str = None):
        if HF_API_KEY is None:
            raise ValueError("HF_API_KEY environment variable is required for HuggingFaceEmbedder")

        # Default embedding model
        self.model_name = model_name or os.getenv(
            "HF_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Correct Inference API endpoint (model-addressed)
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"

        self.headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }

        print(">>> Using embedding model:", self.model_name)
        print(">>> HF API URL:", self.api_url)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Calls the Hugging Face Inference API for feature extraction (embeddings).
        """

        # HF accepts either a single string or a list of strings
        payload = {
            "inputs": texts
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"HuggingFace API error {response.status_code}: {response.text}"
            )

        data = response.json()

        # Response format:
        # - Single input: [dim]
        # - Batch input: [[dim], [dim], ...]
        if isinstance(data, list) and isinstance(data[0], list):
            return data
        elif isinstance(data, list):
            return [data]
        else:
            raise RuntimeError(f"Unexpected HuggingFace response format: {data}")


# -------------------------
# Embedder factory
# -------------------------
def get_embedder():
    if HF_API_KEY:
        return HuggingFaceEmbedder()
    else:
        raise ValueError("No embedding backend configured.")
