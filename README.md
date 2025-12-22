## Lightweight RAG pipeline (3gb)

#### This project showcases a very light weight RAG pipeline using Qwen2.5(1.5gb) as the tokenizer and generator. Minilm-L6-V2 is being used for sentence embeddings and for the reranker.

##### The main branch contains code to run this locally. Reference the rag_models_folder.png to see which models are needed to run with the correct contents. They were pulled from hugging face.

##### rag_api branch is currently being worked on to have this application work with the fast api library. This branch will not work locally due to the pathing changes.