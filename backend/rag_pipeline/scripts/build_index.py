import os
from backend.rag_pipeline.ingestion.pdf_loader import load_pdf_text
from backend.rag_pipeline.ingestion.text_loader import load_text_file
from backend.rag_pipeline.chunking.chunker import chunk_text
from backend.rag_pipeline.embeddings.embedder import get_embedder
from backend.rag_pipeline.embeddings.vectorstore import FaissStore
from transformers import AutoTokenizer

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/rag_pipeline
DATA_DIR = BASE_DIR / "data" / "raw_docs"
INDEX_DIR = BASE_DIR / "vectorstore"

def load_all_documents():
    print('load docs portion: ', os.getcwd())
    docs = []

    for file in DATA_DIR.iterdir():
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
    print("embed model:", embed)

    model_path = "../models/qwen2.5"   # your local Qwen2.5 folder
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # # GPT2 tokenizer and model
    # tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    # model = GPT2LMHeadModel.from_pretrained(model_path)
    # Load tokenizer + model
    store = None

    docs = load_all_documents()
    print(f"Loaded {len(docs)} documents.")

    for doc_name, text in docs:
        print(f"\nChunking {doc_name}...")

        # Pass tokenizer into chunk_text
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
        print("vectors type:", type(vectors))

        if store is None:
            dim = len(vectors[0])
            store = FaissStore(dim)

        # store the actual chunk text, not the whole dict
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
    store.save(INDEX_DIR)

    print("\nIndex build complete!")
    print(f"Total vectors stored: {store.index.ntotal}")

# Global cache
STORE = None
EMBEDDER = None

def load_index(index_name="default"):
    global STORE, EMBEDDER
    INDEX_DIR.mkdir(exist_ok=True)

    print("Loading embedding model to load index...")
    EMBEDDER = get_embedder()

    dim = len(EMBEDDER.embed(["test"])[0])
    print(f"Embedding dimension from load_index in build_index: {dim}")

    print(f"Loading FAISS index from load_index in build_index: {INDEX_DIR}")
    STORE = FaissStore.load(dim, path=str(INDEX_DIR))

    print("STORE id from build_index: ", id(STORE))
    print("EMBEDDER id from build_index: ", id(EMBEDDER))

    print("Index loaded successfully!")
    print(f"Total vectors in index: {STORE.index.ntotal}")

if __name__ == "__main__":
    print('build_index passed')

    # build_index()