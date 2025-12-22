from pydantic import BaseModel
from backend.rag_pipeline.scripts.ask_question import ask_question
from backend.rag_pipeline.scripts.build_index import load_index, build_index

from fastapi import FastAPI, HTTPException
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Local RAG API")

# ---------------------------------------------------------
# CORS MUST BE ADDED BEFORE ROUTES
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load default index on startup
@app.on_event("startup")
def startup_event():
    # import os
    # print(os.listdir("vectorstore"))
    # so now I have to load the correct index, and it doesn't just always get
    # renamed into index.faiss and meta.pkl or whatnot.
    load_index("Blueskin-WP-200-Product-Data-2140732")

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

from pydantic import BaseModel
from backend.rag_pipeline.scripts.build_index import build_index

class BuildIndexRequest(BaseModel):
    index_name: str | None = "default"

class BuildIndexResponse(BaseModel):
    message: str
    index_name: str

@app.post("/build_index", response_model=BuildIndexResponse)
def api_build_index(req: BuildIndexRequest):
    try:
        # If your build_index() function needs the index name,
        # modify it to accept a parameter. For now, we assume
        # it builds into INDEX_DIR based on index_name.
        build_index()  
        return BuildIndexResponse(
            message="Index built successfully",
            index_name=req.index_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI

from backend.rag_pipeline.scripts.build_index import INDEX_DIR

@app.get("/list_indexes")
def list_indexes():
    indexes = []

    for faiss_file in INDEX_DIR.glob("*.faiss"):
        name = faiss_file.stem  # removes .faiss
        indexes.append(name)

    return {"indexes": indexes}


#TODO add a normal page or something

#TODO make it so that we can query the code that triggers build_index

