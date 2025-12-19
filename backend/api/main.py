from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag_pipeline.scripts.ask_question import ask_question
from backend.rag_pipeline.scripts.build_index import load_index

app = FastAPI(title="Local RAG API")

# Load default index on startup
@app.on_event("startup")
def startup_event():
    load_index("default")

class QueryRequest(BaseModel):
    query: str
    index_name: str | None = "default"

class QueryResponse(BaseModel):
    answer: str

@app.post("/query")
def query_rag(request: QueryRequest):
    answer = ask_question(request.query, index_name=request.index_name)
    return QueryResponse(answer=answer)
