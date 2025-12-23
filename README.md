## Lightweight RAG pipeline (3gb)

#### This project showcases a very light weight RAG pipeline using Qwen2.5(1.5gb) as the tokenizer and generator. Minilm-L6-V2 is being used for sentence embeddings and for the reranker.

##### The main branch contains code to run this locally. Reference the rag_models_folder.png to see which models are needed to run with the correct contents. They were pulled from hugging face.

##### Run this in the terminal using either of these commands.
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Mostly deprecated but kept in repo to remember the good ol' days