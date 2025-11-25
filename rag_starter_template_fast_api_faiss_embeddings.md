# RAG Starter Template — FastAPI + FAISS + Embeddings

This document is a complete starter template for a Retrieval-Augmented Generation (RAG) system. It contains:

- Project structure and minimal, runnable code for each component
- A FastAPI endpoint `/query` that performs retrieval + generation
- A FAISS-backed vector store example (with disk persistence)
- Support for OpenAI embeddings *or* sentence-transformers
- A simple MMR retriever + optional reranker stub
- Dockerfile, requirements, and `env.example`
- A mermaid system diagram showing the pipeline

> **How to use this document:** Each code block is a file in the repo. Copy files into the structure shown, install requirements, set environment variables, and run.

---

## Project structure

```
rag-starter/
├── README.md
├── env.example
├── Dockerfile
├── requirements.txt
├── data/
│   └── raw_docs/
├── ingestion/
│   ├── pdf_loader.py
│   └── text_loader.py
├── chunking/
│   └── chunker.py
├── embeddings/
│   ├── embedder.py
│   └── vectorstore.py
├── retrieval/
│   ├── retriever.py
│   └── reranker.py
├── generation/
│   ├── prompt.py
│   └── generator.py
├── api/
│   ├── main.py
│   └── schemas.py
└── eval/
    └── eval_stub.py
```

---

## `env.example`

```env
# OpenAI
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_COMPLETION_MODEL=gpt-4o-mini

# Optional sentence-transformers model
SENT_TRANSFORMER_MODEL=all-MiniLM-L6-v2

# Vector DB path
VECTOR_STORE_PATH=./data/faiss_index

# Server
HOST=0.0.0.0
PORT=8000
```

---

## `requirements.txt`

```
fastapi
uvicorn[standard]
pydantic
python-multipart
requests
faiss-cpu
sentence-transformers
transformers
torch
tqdm
pdfplumber
numpy
scipy
openai
python-dotenv
```

---

## `ingestion/pdf_loader.py`

```python
# ingestion/pdf_loader.py
import pdfplumber
from pathlib import Path


def load_pdf_text(path: str) -> str:
    text_chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_chunks.append(text)
    return "\n\n".join(text_chunks)


if __name__ == "__main__":
    p = Path(__file__).parent.parent / "data" / "raw_docs"
    for f in p.glob("*.pdf"):
        print("Loading:", f)
        txt = load_pdf_text(str(f))
        out = f.with_suffix('.txt')
        out.write_text(txt)
        print("Wrote:", out)
```

---

## `ingestion/text_loader.py`

```python
# ingestion/text_loader.py
from pathlib import Path


def load_text_file(path: str) -> str:
    return Path(path).read_text()


if __name__ == "__main__":
    p = Path(__file__).parent.parent / "data" / "raw_docs"
    for f in p.glob("*.txt"):
        print(f, len(f.read_text()))
```

---

## `chunking/chunker.py`

```python
# chunking/chunker.py
import re
from typing import List, Dict


def sentence_split(text: str) -> List[str]:
    # Very simple sentence splitter; replace with nltk if desired
    toks = re.split(r'(?<=[.!?])\s+', text)
    return [t.strip() for t in toks if t.strip()]


def chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> List[Dict]:
    """
    Create overlapping chunks by sentence boundaries. Returns list of dicts:
      {"chunk_id": int, "text": str}
    """
    sents = sentence_split(text)
    chunks = []
    current = []
    curr_len = 0

    def flush(i):
        if not current:
            return None
        chunk_text = " ".join(current)
        return {"chunk_id": i, "text": chunk_text}

    i = 0
    for sent in sents:
        # crude token estimate: words ~= tokens
        tok_est = len(sent.split())
        if curr_len + tok_est <= max_tokens or not current:
            current.append(sent)
            curr_len += tok_est
        else:
            c = flush(i)
            if c:
                chunks.append(c)
                i += 1
            # start new chunk with overlap
            if overlap > 0:
                # keep last sentences that fit into overlap
                keep = []
                keep_len = 0
                while current and keep_len + len(current[-1].split()) <= overlap:
                    keep.insert(0, current.pop())
                    keep_len += len(keep[0].split())
                current = keep
                curr_len = keep_len
            else:
                current = []
                curr_len = 0
            current.append(sent)
            curr_len += tok_est
    # flush final
    c = flush(i)
    if c:
        chunks.append(c)
    return chunks


if __name__ == "__main__":
    t = Path(__file__).parent.parent / "data" / "raw_docs" / "sample.txt"
    if t.exists():
        text = t.read_text()
        ch = chunk_text(text)
        print("Chunks:", len(ch))
```

---

## `embeddings/embedder.py`

```python
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


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("SENT_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()


def get_embedder():
    if OPENAI:
        return OpenAIEmbedder()
    return SentenceTransformerEmbedder()
```

---

## `embeddings/vectorstore.py`

```python
# embeddings/vectorstore.py
import os
import faiss
import numpy as np
import pickle
from typing import List, Dict

INDEX_PATH = os.getenv("VECTOR_STORE_PATH", "./data/faiss_index")


class FaissStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product for cosine if vectors normalized
        self.metadatas = []

    def add(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        arr = np.vstack(vectors).astype('float32')
        # normalize for cosine
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
```

---

## `retrieval/retriever.py`

```python
# retrieval/retriever.py
from typing import List, Dict
import numpy as np


def mmr(ranked_list, query_embedding, lambda_mult=0.7, top_k=5):
    # very small MMR implementation wrapper for metadata list
    # ranked_list: list of dict {"embedding": , "metadata": }
    if not ranked_list:
        return []

    selected = []
    candidates = ranked_list.copy()

    # precompute similarities
    embs = np.vstack([c['embedding'] for c in candidates])
    q = np.array(query_embedding)
    from numpy.linalg import norm
    def cos(a,b):
        return np.dot(a,b) / (norm(a)*norm(b)+1e-12)

    sims = [cos(q, e) for e in embs]
    # naive greedy pick
    selected_idx = []
    while len(selected_idx) < min(top_k, len(candidates)):
        best = None
        best_score = -1e9
        for i, c in enumerate(candidates):
            if i in selected_idx:
                continue
            relevance = sims[i]
            diversity = 0
            for si in selected_idx:
                diversity = max(diversity, cos(embs[si], embs[i]))
            score = lambda_mult * relevance - (1 - lambda_mult) * diversity
            if score > best_score:
                best_score = score
                best = i
        selected_idx.append(best)
    out = [candidates[i] for i in selected_idx]
    return out


class Retriever:
    def __init__(self, embedder, store, top_k=8):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str):
        q_emb = self.embedder.embed([query])[0]
        raw = self.store.query(q_emb, top_k=self.top_k*3)  # retrieve more for reranking/MMR
        # attach embeddings to raw (if metadata contains embedding skip recompute)
        enriched = []
        for r in raw:
            meta = r['metadata']
            emb = meta.get('embedding')
            if emb is None:
                # fallback: we could store embeddings in meta; for now assume not
                emb = None
            enriched.append({"score": r['score'], "metadata": meta, "embedding": emb})
        # run mmr if embeddings present (otherwise simple top-k)
        with_emb = [e for e in enriched if e['embedding'] is not None]
        if with_emb:
            selected = mmr(with_emb, q_emb, top_k=5)
        else:
            selected = enriched[:5]
        return selected
```

---

## `retrieval/reranker.py` (optional)

```python
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
```

---

## `generation/prompt.py`

```python
# generation/prompt.py
from typing import List


PROMPT_TEMPLATE = """
Use ONLY the context below to answer the question. If the context does not contain the answer, say "I cannot answer from the provided documents." Provide short, citation-backed answers. Cite chunks as [doc_id:chunk_id].

Context:
{context}

Question: {question}

Answer:
"""


def build_prompt(chunks: List[dict], question: str) -> str:
    ctx = []
    for c in chunks:
        md = c['metadata']
        text = md.get('text') or md.get('chunk_text') or '"
        src = f"[{md.get('doc_id','unknown')}:{md.get('chunk_id','?')}]"
        ctx.append(f"{src} {text}")
    context = "\n\n".join(ctx)
    return PROMPT_TEMPLATE.format(context=context, question=question)
```

---

## `generation/generator.py`

```python
# generation/generator.py
import os
from typing import List
OPENAI = os.getenv("OPENAI_API_KEY") is not None

if OPENAI:
    import openai


def generate_answer(prompt: str, max_tokens: int = 256):
    if OPENAI:
        model = os.getenv('OPENAI_COMPLETION_MODEL', 'gpt-4o-mini')
        res = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return res['choices'][0]['message']['content']
    else:
        # fallback: local small model using transformers (approx)
        from transformers import pipeline
        pipe = pipeline('text-generation', model='gpt2', device=-1)
        out = pipe(prompt, max_length=512, do_sample=False)
        return out[0]['generated_text']
```

---

## `api/schemas.py`

```python
# api/schemas.py
from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class ChunkOut(BaseModel):
    doc_id: str
    chunk_id: int
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[ChunkOut]
```

---

## `api/main.py` — FastAPI endpoint

```python
# api/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.schemas import QueryRequest, QueryResponse, ChunkOut
from embeddings.embedder import get_embedder
from embeddings.vectorstore import FaissStore
from retrieval.retriever import Retriever
from generation.prompt import build_prompt
from generation.generator import generate_answer

app = FastAPI()

# simple init: load embedder and vectorstore
embedder = get_embedder()
# NOTE: dimension must match embedder output size; use 384 or 1536 depending on model
VECTOR_DIM = 1536
STORE_PATH = os.getenv('VECTOR_STORE_PATH', './data/faiss_index')

try:
    store = FaissStore.load(VECTOR_DIM, STORE_PATH)
except Exception as e:
    # in prod you'd create index if missing; for template raise
    print("Could not load store:", e)
    store = None

retriever = Retriever(embedder, store)

@app.post('/query', response_model=QueryResponse)
def query(q: QueryRequest):
    if store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    hits = retriever.retrieve(q.question)
    # prepare prompt
    # transform hits into expected metadata format for prompt builder
    chunks = []
    for h in hits:
        md = h['metadata']
        chunks.append({'metadata': md})
    prompt = build_prompt(chunks, q.question)
    answer = generate_answer(prompt)

    sources = []
    for h in hits:
        md = h['metadata']
        sources.append(ChunkOut(doc_id=md.get('doc_id', 'unknown'), chunk_id=md.get('chunk_id', 0), text=md.get('text',''), score=h.get('score', 0.0)))
    return QueryResponse(answer=answer, sources=sources)


@app.get('/health')
def health():
    return {"status": "ok"}
```

---

## `eval/eval_stub.py`

```python
# eval/eval_stub.py
# Minimal evaluation harness. Replace judge with LLM pairwise or human labels.

from typing import List


def check_retrieval_gold(hits: List[dict], gold_doc_ids: List[str]):
    # percent of gold docs retrieved
    found = sum(1 for h in hits if h['metadata'].get('doc_id') in gold_doc_ids)
    return found / max(1, len(gold_doc_ids))


if __name__ == '__main__':
    print('Add labeled QA pairs and implement LLM-based judge to compute faithfulness.')
```

---

## `Dockerfile`

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Mermaid system diagram

```mermaid
flowchart LR
    A[Raw docs: PDF/HTML/MD] --> B[Ingestion (pdfplumber, html parser)]
    B --> C[Chunking (sentence / semantic splitter)]
    C --> D[Embedder (OpenAI / sentence-transformers)]
    D --> E[Vector DB (FAISS / Chroma / Milvus)]
    subgraph Runtime
      Q[User Query] --> R[Query Embedder]
      R --> S[Retriever: kNN + MMR]
      S --> T[Reranker - optional]
      T --> U[Context Assembly]
      U --> V[Generator (LLM) -> answer]
      V --> W[Return answer + citations]
    end
    E --> S
    E --> Eval[Evaluation harness]

```

---

## Quick start (local)

1. Copy files into repo structure.
2. `pip install -r requirements.txt`
3. Set `.env` from `env.example` (OpenAI API key or leave blank to use sentence-transformers fallback)
4. Ingest documents (put `.txt` or `.pdf` in `data/raw_docs` and run ingestion loader)
5. Chunk documents: write quick script to chunk and call embedder + FaissStore.add + save index
6. Start API: `uvicorn api.main:app --reload --port 8000`
7. POST `http://localhost:8000/query` with `{"question": "What is X?"}`

---

## Notes / Next steps

- **Persistence:** This template uses a simple faiss flat index. For production, switch to an HNSW index and persistent backing (Milvus, Weaviate, Pinecone).
- **Reranker:** Add a cross-encoder reranker for higher accuracy.
- **Citation fidelity:** Use chunk metadata to include exact source and offsets in answers.
- **Monitoring:** log retrieval precision/recall, token usage, and hallucination rate.

---

If you'd like, I can:

- Generate the **chunking + embedding ingestion script** that builds the FAISS index from `/data/raw_docs` (I recommend we do that next).
- Create a **simple Streamlit UI** for demoing the RAG app.
- Provide **test QA pairs** and an LLM-based judge harness for automated evaluation.

