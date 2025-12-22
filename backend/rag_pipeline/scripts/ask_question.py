import backend.rag_pipeline.scripts.build_index as build_index
from backend.rag_pipeline.retrieval.retriever import Retriever
from backend.rag_pipeline.retrieval.reranker import Reranker
from backend.rag_pipeline.generation.prompt import build_prompt
from backend.rag_pipeline.generation.generator import generate_answer

# def ask_question(query: str, index_name="default"):
#     if build_index.STORE is None or build_index.EMBEDDER is None:
#         raise RuntimeError("Index not loaded. Call load_index() first.")

#     retriever = Retriever(build_index.EMBEDDER, build_index.STORE, top_k=15)
#     candidates = retriever.retrieve(query)

#     reranker = Reranker()
#     reranked = reranker.rerank(query, candidates, top_k=1)

#     prompt = build_prompt(reranked, query)
#     # print('prompt from ask_question: ', prompt)
#     # oh. so generate_answer gives me context and question...
#     return generate_answer(prompt)

def extract_answer(full_output: str) -> str:
    # Split on the last occurrence of "Answer:"
    if "Answer:" in full_output:
        return full_output.split("Answer:")[-1].strip()
    return full_output.strip()


def ask_question(query: str, index_name="default"):
    if build_index.STORE is None or build_index.EMBEDDER is None:
        raise RuntimeError("Index not loaded. Call load_index() first.")

    retriever = Retriever(build_index.EMBEDDER, build_index.STORE, top_k=15)
    candidates = retriever.retrieve(query)

    reranker = Reranker()
    reranked = reranker.rerank(query, candidates, top_k=1)

    # Build prompt
    prompt = build_prompt(reranked, query)

    # Generate answer text (full prompt + answer)
    full_output = generate_answer(prompt)

    # Extract only the answer portion
    answer = extract_answer(full_output)

    # Extract context chunks cleanly
    context_chunks = [
        f"[{c['metadata'].get('doc_id','unknown')}:{c['metadata'].get('chunk_id','?')}] "
        f"{c['metadata'].get('text') or c['metadata'].get('chunk_text') or ''}"
        for c in reranked
    ]

    return {
        "answer": answer,
        "context": context_chunks,
        "raw_output": full_output  # optional for debugging
    }


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
    # answer_question("What is the application like for this product?")
    # answer_question("What is the general idea of this document?")
    # answer_question("What sections pertain to corroborated evidence?")
    # answer_question("What does this document say about women?")
    # answer_question("What does this document say about equal employment and disciplinary action?")
    # answer_question("What does this document say about exercise and physical requirements?")
    # answer_question("What does this document say about promotions and raises?")
    # answer_question("What are the new security training requirements?")
    # answer_question("What are the expected fitness standards for generals?")
    # answer_question("Explain to me what Trump thinks about Epstein?")
    ask_question("What are the highlighted portions of this text?")
    