import os
from backend.rag_pipeline.ingestion.pdf_loader import load_pdf_text
from backend.rag_pipeline.ingestion.text_loader import load_text_file
from backend.rag_pipeline.chunking.chunker import chunk_text
from backend.rag_pipeline.embeddings.embedder import get_embedder
from backend.rag_pipeline.embeddings.vectorstore import FaissStore
from transformers import AutoTokenizer

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/rag_pipeline
FRONTEND_BASE_DIR = Path(__file__).resolve().parents[3]

# DATA_DIR is for local file usage
DATA_DIR = BASE_DIR / "data" / "raw_docs"

# For frontend file uploads
FRONTEND_DATA_DIR = FRONTEND_BASE_DIR / "uploaded_docs"

INDEX_DIR = BASE_DIR / "vectorstore"
CURRENT_DIR = Path(__file__).resolve().parent
MODEL_DIR = CURRENT_DIR.parent / "models" / "qwen2.5"

def load_all_documents():
    print('load docs portion: ', os.getcwd())
    docs = []
    print('build_index load_all_documents function FRONTEND_DATA_DIR: ', FRONTEND_DATA_DIR)
    # Switch DATA_DIR w/ FRONTEND_DATA_DIR for local or frontend usage
    for file in FRONTEND_DATA_DIR.iterdir():
        ext = file.suffix.lower()

        if ext == ".pdf":
            print(f"Loading PDF: {file}")
            text = load_pdf_text(file)

        elif ext in [".txt", ".md"]:
            print(f"Loading text file: {file}")
            text = load_text_file(file)

        else:
            print(f"Skipping unsupported file: {file}")
            continue

        docs.append((file, text))

    return docs

def build_index():
    INDEX_DIR.mkdir(exist_ok=True)

    print("Loading embedding model to build index...")
    embed = get_embedder()

    # Load Qwen2.5 tokenizer from HF Hub (no model weights downloaded)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

    store = None
    docs = load_all_documents()
    if not docs:
        raise RuntimeError("No documents found in uploaded_docs/. Cannot build index.")

    index_name = docs[0][0].stem

    print(f"Loaded {len(docs)} documents.")

    for doc_name, text in docs:
        print(f"\nChunking {doc_name}...")

        chunks = chunk_text(
            text,
            tokenizer=tokenizer,
            max_tokens=400,
            overlap=100,
            min_chunk_tokens=150
        )

        chunk_texts = [c["text"] for c in chunks]

        print(f"Embedding {len(chunk_texts)} chunks...")

        vectors = embed.embed(chunk_texts)

        if store is None:
            dim = len(vectors[0])
            store = FaissStore(dim)

        metadatas = [
            {
                "doc_name": doc_name,
                "chunk_id": c["chunk_id"],
                "text": c["text"]
            }
            for c in chunks
        ]

        store.add(vectors, metadatas)

    print("\nSaving FAISS store...")
    store.save(INDEX_DIR, index_name)

    print("\nIndex build complete!")
    print(f"Total vectors stored: {store.index.ntotal}")


# Global cache
STORE = None
EMBEDDER = None
CURRENT_INDEX = None

def load_index(index_name="default"):
    global STORE, EMBEDDER
    print('INDEX PATHS DIR : ',INDEX_DIR)
    # index_dir = INDEX_ROOT  # all files live directly in vectorstore/

    # Load embedder
    EMBEDDER = get_embedder()

    # Load FAISS + metadata
    dim = len(EMBEDDER.embed(["test"])[0])
    STORE = FaissStore.load(dim, path=str(INDEX_DIR), name=index_name)
    CURRENT_INDEX = index_name
    print('current index build_index: ', CURRENT_INDEX)
    print(f"Loaded index build_index: {index_name}")

if __name__ == "__main__":
    print('build_index passed')
    # build_index()
