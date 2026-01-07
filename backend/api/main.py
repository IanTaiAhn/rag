from pydantic import BaseModel
from rag_pipeline.scripts.ask_question import ask_question
from rag_pipeline.scripts.build_index import build_index
from rag_pipeline.scripts.build_index import INDEX_DIR

from pathlib import Path
from fastapi import UploadFile, File, HTTPException
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

class QueryRequest(BaseModel):
    query: str
    index_name: str | None = "default"

class QueryResponse(BaseModel):
    answer: str
    context: list[str]
    raw_output: str | None = None  # optional

class BuildIndexRequest(BaseModel):
    index_name: str | None = "default"

class BuildIndexResponse(BaseModel):
    message: str
    index_name: str

UPLOAD_DIR = Path("uploaded_docs")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/ask_question", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    result = ask_question(request.query, index_name=request.index_name)
    return QueryResponse(**result)

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

@app.delete("/delete_index/{index_name}")
def delete_index(index_name: str):
    faiss_file = INDEX_DIR / f"{index_name}.faiss"
    meta_file = INDEX_DIR / f"{index_name}_meta.pkl"  # if you store metadata

    deleted = False

    if faiss_file.exists():
        faiss_file.unlink()
        deleted = True

    if meta_file.exists():
        meta_file.unlink()
        deleted = True

    if not deleted:
        raise HTTPException(status_code=404, detail="Index not found")

    return {"message": "Index deleted"}

@app.get("/list_indexes")
def list_indexes():
    indexes = []

    for faiss_file in INDEX_DIR.glob("*.faiss"):
        name = faiss_file.stem  # removes .faiss
        indexes.append(name)

    return {"indexes": indexes}

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Only PDF or TXT files allowed")

    file_path = UPLOAD_DIR / file.filename

    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {"message": "File uploaded successfully", "filename": file.filename}

@app.get("/list_uploaded_docs")
def list_uploaded_docs():
    files = [f.name for f in UPLOAD_DIR.glob("*") if f.is_file()]
    return {"files": files}

@app.delete("/delete_uploaded_doc/{filename}")
def delete_uploaded_doc(filename: str):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()
        return {"message": "File deleted"}
    else:
        raise HTTPException(status_code=404, detail="File not found")


#TODO add a normal page or something
