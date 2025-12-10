# --- FIX IMPORT PATHING ---
import sys, os
SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_PATH))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- END FIX ---

import numpy as np
from pathlib import Path
from tqdm import tqdm

from ingestion.pdf_loader import load_pdf_text
from ingestion.text_loader import load_text_file
from chunking.chunker import chunk_text
from embeddings.embedder import get_embedder
from embeddings.vectorstore import FaissStore


DATA_DIR = Path("../data/raw_docs")
INDEX_DIR = Path("vectorstore")


def load_all_documents():
    """Load all .txt, .md, or .pdf files into memory."""

    # break here TODO
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
            docs.append((file, text))

        else:
            print(f"Skipping unsupported file: {file}")
            continue


        # print('cleaned?: ')
        # cleaned = load_text_file(text)
        # docs.append((file.name, cleaned))

    return docs


def build_index():
    INDEX_DIR.mkdir(exist_ok=True)

    print("Loading embedding model...")
    embed = get_embedder()
    print('embed model: ', embed)

    store = None

    docs = load_all_documents()
    print(f"Loaded {len(docs)} documents.")

    for doc_name, text in docs:
        print(f"\nChunking {doc_name}...")
        chunks = chunk_text(text)
        # print("DEBUG chunks:", chunks[:3], [type(c) for c in chunks[:3]])

        chunk_texts = [c["text"] for c in chunks]

        print(f"Embedding {len(chunks)} chunks...")
        print(f"Embedding {len(chunk_texts)} chunks texts...")

        vectors = embed.embed(chunk_texts)  # returns list of numpy vectors
        # print('vectors: ', vectors)
        print('vectors type: ', type(vectors))
        

        if store is None:
            dim = len(vectors[0])
            store = FaissStore(dim)

        metadatas = [
            {"doc_name": doc_name, "chunk_id": i, "text": chunks[i]}
            for i in range(len(chunks))
        ]

        store.add(vectors, metadatas)

    print("\nSaving FAISS store...")
    store.save(INDEX_DIR)

    print("\nIndex build complete!")
    print(f"Total vectors stored: {store.index.ntotal}")


if __name__ == "__main__":
    build_index()
