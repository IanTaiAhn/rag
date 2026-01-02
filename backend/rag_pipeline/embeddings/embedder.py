import os
from typing import List
import requests

# OPENAI = os.getenv("OPENAI_API_KEY") is not None
HF_API_KEY = os.getenv("HF_API_KEY")


class BaseEmbedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


# -------------------------
# OpenAI embedder (unchanged)
# -------------------------
# if OPENAI:
#     import openai

#     class OpenAIEmbedder(BaseEmbedder):
#         def __init__(self, model: str = None):
#             self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL")

#         def embed(self, texts: List[str]):
#             res = openai.Embedding.create(model=self.model, input=texts)
#             return [r["embedding"] for r in res["data"]]


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

        # Default to a MiniLM embedding model
        self.model_name = model_name or os.getenv(
            "HF_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
        self.headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    def embed(self, texts: List[str]):
        """
        HF API accepts a list of strings and returns a list of embeddings.
        """
        payload = {"inputs": texts}

        response = requests.post(self.api_url, headers=self.headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"HuggingFace API error {response.status_code}: {response.text}"
            )

        return response.json()


# -------------------------
# Embedder factory
# -------------------------
def get_embedder():
    # if OPENAI:
    #     return OpenAIEmbedder()

    # Prefer HF API over local models
    if HF_API_KEY:
        return HuggingFaceEmbedder()

    # Fallback: local model (legacy)
    # from sentence_transformers import SentenceTransformer

    # class SentenceTransformerEmbedder(BaseEmbedder):
    #     def __init__(self, model_name: str = None):
    #         self.model_name = model_name or os.getenv(
    #             "SENT_TRANSFORMER_MODEL",
    #             "backend/rag_pipeline/models/minilm"
    #         )
    #         self.model = SentenceTransformer(self.model_name)

    #     def embed(self, texts: List[str]):
    #         return self.model.encode(
    #             texts,
    #             show_progress_bar=False,
    #             convert_to_numpy=True
    #         ).tolist()

    # return SentenceTransformerEmbedder()
