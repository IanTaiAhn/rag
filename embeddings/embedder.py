# embeddings/embedder.py
import os
from typing import List

OPENAI = os.getenv("OPENAI_API_KEY") is not None


class BaseEmbedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


if OPENAI:
    import openai

    class OpenAIEmbedder(BaseEmbedder):
        def __init__(self, model: str = None):
            self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL")

        def embed(self, texts: List[str]):
            # batch-friendly
            res = openai.Embedding.create(model=self.model, input=texts)
            return [r['embedding'] for r in res['data']]


# fallback: sentence-transformers
from sentence_transformers import SentenceTransformer


# class SentenceTransformerEmbedder(BaseEmbedder):
#     def __init__(self, model_name: str = None):
#         self.model_name = model_name or os.getenv("SENT_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
#         self.model = SentenceTransformer(self.model_name)

#     def embed(self, texts: List[str]):
#         return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

# loaded it in locally.
class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = None):
        # Default to local path instead of remote model name
        self.model_name = model_name or os.getenv("SENT_TRANSFORMER_MODEL", "./models/minilm")
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]):
        return self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()


def get_embedder():
    if OPENAI:
        return OpenAIEmbedder()
    return SentenceTransformerEmbedder()