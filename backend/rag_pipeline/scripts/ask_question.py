import rag_pipeline.scripts.build_index as build_index
from rag_pipeline.retrieval.retriever import Retriever
from rag_pipeline.retrieval.reranker import Reranker
from rag_pipeline.generation.prompt import build_prompt
from rag_pipeline.generation.generator import generate_answer


def extract_answer(full_output: str) -> str:
    """
    Extracts the final answer from a model output.
    Splits on the last occurrence of 'Answer:' to avoid accidental collisions.
    """
    if "Answer:" in full_output:
        return full_output.split("Answer:")[-1].strip()
    return full_output.strip()


def ask_question(query: str, index_name="default"):
    """
    Main RAG pipeline entrypoint.
    - Loads index if needed
    - Retrieves top_k chunks
    - Reranks them
    - Builds prompt
    - Calls HF/OpenAI/local generator
    - Returns structured answer + context
    """

    # -------------------------
    # Ensure index is loaded
    # -------------------------
    if (
        build_index.STORE is None or
        build_index.EMBEDDER is None or
        build_index.CURRENT_INDEX != index_name
    ):
        build_index.load_index(index_name)
        build_index.CURRENT_INDEX = index_name

    # -------------------------
    # Retrieve candidates
    # -------------------------
    retriever = Retriever(build_index.EMBEDDER, build_index.STORE, top_k=15)
    candidates = retriever.retrieve(query)

    # -------------------------
    # Rerank using HF/local reranker
    # -------------------------
    reranker = Reranker()
    reranked = reranker.rerank(query, candidates, top_k=1)

    # -------------------------
    # Build prompt for Qwen/OpenAI/HF
    # -------------------------
    prompt = build_prompt(reranked, query)

    # -------------------------
    # Generate answer (HF/OpenAI/local)
    # -------------------------
    full_output = generate_answer(prompt)
    answer = extract_answer(full_output)

    # -------------------------
    # Build context metadata
    # -------------------------
    context_chunks = [
        f"[{c['metadata'].get('doc_id','unknown')}:{c['metadata'].get('chunk_id','?')}] "
        f"{c['metadata'].get('text') or c['metadata'].get('chunk_text') or ''}"
        for c in reranked
    ]

    return {
        "answer": answer,
        "context": context_chunks,
        "raw_output": full_output
    }
