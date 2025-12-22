from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag_pipeline.scripts.ask_question import ask_question
from backend.rag_pipeline.scripts.build_index import load_index, build_index

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
    context: list[str]
    raw_output: str | None = None  # optional

@app.post("/ask_question", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    result = ask_question(request.query, index_name=request.index_name)
    return QueryResponse(**result)


# @app.post("/build_index")
# def build_index_rag(request: QueryRequest):
#     answer = build_index()
#     return QueryResponse(answer=answer)

#TODO add a normal page or something

#TODO make it so that we can query the code that triggers build_index

