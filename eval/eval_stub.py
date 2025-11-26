# eval/eval_stub.py
# Minimal evaluation harness. Replace judge with LLM pairwise or human labels.

from typing import List


def check_retrieval_gold(hits: List[dict], gold_doc_ids: List[str]):
    # percent of gold docs retrieved
    found = sum(1 for h in hits if h['metadata'].get('doc_id') in gold_doc_ids)
    return found / max(1, len(gold_doc_ids))


if __name__ == '__main__':
    print('Add labeled QA pairs and implement LLM-based judge to compute faithfulness.')