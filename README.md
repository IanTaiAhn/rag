## Lightweight RAG pipeline (3gb)

#### This project showcases a very light weight RAG pipeline using Qwen2.5(1.5gb) as the tokenizer and generator. Minilm-L6-V2 is being used for sentence embeddings and for the reranker.

### Run fast api using this command

uvicorn main:app --reload --log-level debug
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

### Run frontend using vite

npm run dev


#### Work --> Personal

Install python-multipart onto cuda.
If App2 works you need lucide-react
pip install python-multipart
npm install lucide-react

#### Do before

delete package-lock.json if there and run 'npm install' for a fresh one.