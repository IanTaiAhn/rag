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

        # 🔥 Pass tokenizer into chunk_text
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

        # 🔥 FIX: store the actual chunk text, not the whole dict
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


def answer_question(query: str):
    embedder = get_embedder()
    # use len() since embed returns list of floats
    print('got an embedder')
    dim = len(embedder.embed(["test"])[0])
    store = FaissStore.load(dim)
    print("i have a store?")
    retriever = Retriever(embedder, store, top_k=15)
    candidates = retriever.retrieve(query)

    # reranker
    reranker = Reranker()
    reranked = reranker.rerank(query, candidates, top_k=1)
    prompt = build_prompt(reranked, query)

    # # Without reranker
    # prompt = build_prompt(candidates, query)
    return generate_answer(prompt)


if __name__ == "__main__":
    # build_index()
    # answer_question("Explain modal jazz")
    # answer_question("I'm gonna launch a potato and the castle over there. What should I use to launch it?")
    # answer_question("What is an eigenvalue?")
    # answer_question("What is the name of the Gym leader in Saffron City?")
    # answer_question("What are the names of the three starter pokemon?")
    # answer_question("Explain to me how to win in RBY.")
    # answer_question("What is the main landmark in Aurelia?")
    # answer_question("Does Aurelia have a king?")
    # answer_question("When was the University of Aurelia established?")
    # answer_question("How tall is the Helion Tower?")
    # answer_question("What are the main sectors of Aurelia's economy?")
    # answer_question("Who is the leader of Celadon City Gym?")
    # answer_question("Who does Wartortle evolve into?")
    # answer_question("Is Sabrina bad?")
    # answer_question("Which TMs can paralyze?")
    # answer_question("What is a Charmander?")
    # answer_question("What does focus strike do?")
    # answer_question("What badge do I get if I beat Sabrina?")
    # answer_question("Explain what bulbasaur is, and what does it do?")
    # answer_question("Explain how to use this product")
    answer_question("What is the application like for this product?")