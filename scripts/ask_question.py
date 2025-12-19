# --- FIX IMPORT PATHING ---
import sys, os
SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_PATH))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- END FIX ---

from pathlib import Path

from generation.generator import generate_answer
from generation.prompt import build_prompt
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever

from embeddings.embedder import get_embedder
from embeddings.vectorstore import FaissStore


DATA_DIR = Path("../data/raw_docs")
INDEX_DIR = Path("vectorstore")


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
    answer_question("What are the highlighted portions of this text?")
    