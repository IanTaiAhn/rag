import os
import faiss
import numpy as np
import pickle
from typing import List, Dict

INDEX_PATH = os.getenv("VECTOR_STORE_PATH", "./vectorstore")

class FaissStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine if normalized
        self.metadatas = []

    def add(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        arr = np.vstack(vectors).astype('float32')
        faiss.normalize_L2(arr)
        self.index.add(arr)
        self.metadatas.extend(metadatas)

    def save(self, path: str = INDEX_PATH):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/meta.pkl", "wb") as f:
            pickle.dump(self.metadatas, f)

    @classmethod
    def load(cls, dim: int, path: str = INDEX_PATH):
        inst = cls(dim)
        inst.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/meta.pkl", "rb") as f:
            inst.metadatas = pickle.load(f)
        return inst

    def query(self, vector, top_k: int = 5):
        v = np.array(vector).astype('float32')
        faiss.normalize_L2(v.reshape(1, -1))
        D, I = self.index.search(v.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            meta = self.metadatas[idx]
            results.append({"score": float(score), "metadata": meta})
        return results
