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