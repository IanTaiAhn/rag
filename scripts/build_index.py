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

from generation.generator import generate_answer
from generation.prompt import build_prompt
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever

from ingestion.pdf_loader import load_pdf_text
from ingestion.text_loader import load_text_file
from chunking.chunker import chunk_text
from embeddings.embedder import get_embedder
from embeddings.vectorstore import FaissStore
from transformers import AutoModelForCausalLM, AutoTokenizer


DATA_DIR = Path("../data/raw_docs")
INDEX_DIR = Path("vectorstore")

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

    print("Loading embedding model...")
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

if __name__ == "__main__":
    # TODO I want to be able to switch indexs I use?
    build_index()