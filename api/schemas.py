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