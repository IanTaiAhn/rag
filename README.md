## Lightweight RAG pipeline (3gb)

#### This project showcases a very light weight RAG pipeline using Qwen2.5(1.5gb) as the tokenizer and generator. Minilm-L6-V2 is being used for sentence embeddings and for the reranker.

### Run fast api using this command

uvicorn main:app --reload --log-level debug

or

uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

### Run frontend using vite

npm run dev

#### Do before

delete package-lock.json if there and run 'npm install' for a fresh one.

### Instructions to use Rag pipeline

#### NOTE\* you will need to manually install models from hugging face to use this application. look at the "rag_models_folder.png" to see which model contents you'll need.

#### Hugging face models needed:

1. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main
2. https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/main
3. https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/main

Refer to the png included in this repo named "rag_models_folder.png" to see which files are needed.

I'd suggest creating a venv and then installing the dependencies needed one by one to avoid unnecessary packages.
A requirements.txt is included but I found that using the requirements.txt isn't the most reliable way to install needed dependencies.
Also, delete the package-lock.json file it exists and run npm install to get the frontend going.
