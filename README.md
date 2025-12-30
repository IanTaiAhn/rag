## Lightweight RAG Pipeline (≈3 GB)

This project demonstrates a compact Retrieval‑Augmented Generation (RAG) pipeline built around:

Qwen2.5‑1.5B (≈1.5 GB) for tokenization and generation

MiniLM‑L6‑v2 for both sentence embeddings and cross‑encoder reranking

Despite its small footprint, the pipeline delivers strong retrieval quality and responsive generation on modest hardware.

## Demo Video

<video controls src="rag_demo-1.mp4" title="Title"></video>

### Running the Backend (FastAPI)

You can start the API using either command:

`uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

or:

`uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload`

### Running the Frontend (Vite)

cd into the folder `frontend`

`npm run dev`

#### Before running the frontend

Delete any existing package-lock.json

Then run:

`npm install`

This ensures a clean, reproducible dependency install.

### Required Hugging Face Models

You must manually download the following models and place them in the correct folders.

Refer to rag_models_folder.png in the repo to see exactly which files are required.

Sentence Embeddings  
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main

Cross‑Encoder Reranker  
https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/main

Generator (LLM)  
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/main

Make sure the downloaded model directories match the structure shown in the PNG.

### Environment Setup Recommendations

Create a fresh virtual environment (venv or conda)

Install dependencies one by one to avoid unnecessary packages

A requirements.txt is included, but depending on your system, installing from it directly may pull in extra or incompatible versions

For the frontend, remove package-lock.json and reinstall dependencies to ensure a clean Vite environment

### Hardware tested on

1. Intel i5 (No GPU) — 16 GB RAM
   Overall performance: Surprisingly usable for lightweight RAG workloads.

Retrieval + generation latency: Typically 2–3 minutes for longer queries.

Index creation: Ranges from 10–45 seconds, depending on corpus size and chunking strategy.

Takeaway: Works fine for experimentation and small datasets, but latency becomes noticeable with larger contexts or heavier models.

2. AMD Ryzen 9 5900HS + NVIDIA RTX 3060 Laptop GPU — 16 GB RAM
   Overall performance: Significantly smoother across the board.

Retrieval + generation latency: Much faster, with noticeably reduced wait times for both retrieval and text generation.

Index creation: Substantially quicker thanks to GPU acceleration and stronger CPU single‑core performance.

Takeaway: Ideal for local RAG development. GPU acceleration provides a major boost for embedding generation and model inference.

### Document Size Performance

The pipeline handles documents up to ~35 pages smoothly, which is the largest size tested so far. Performance remains stable during indexing and querying at this scale.

However, when working with significantly larger documents, you may notice increased latency during queries. This slowdown is expected, as larger indexes require more time for retrieval and reranking.
