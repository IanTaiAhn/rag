# chunking/chunker.py
import re
from typing import List, Dict


def sentence_split(text: str) -> List[str]:
    # Very simple sentence splitter; replace with nltk if desired
    toks = re.split(r'(?<=[.!?])\s+', text)
    return [t.strip() for t in toks if t.strip()]


def chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> List[Dict]:
    """
    Create overlapping chunks by sentence boundaries. Returns list of dicts:
      {"chunk_id": int, "text": str}
    """
    sents = sentence_split(text)
    chunks = []
    current = []
    curr_len = 0

    def flush(i):
        if not current:
            return None
        chunk_text = " ".join(current)
        return {"chunk_id": i, "text": chunk_text}

    i = 0
    for sent in sents:
        # crude token estimate: words ~= tokens
        tok_est = len(sent.split())
        if curr_len + tok_est <= max_tokens or not current:
            current.append(sent)
            curr_len += tok_est
        else:
            c = flush(i)
            if c:
                chunks.append(c)
                i += 1
            # start new chunk with overlap
            if overlap > 0:
                # keep last sentences that fit into overlap
                keep = []
                keep_len = 0
                while current and keep_len + len(current[-1].split()) <= overlap:
                    keep.insert(0, current.pop())
                    keep_len += len(keep[0].split())
                current = keep
                curr_len = keep_len
            else:
                current = []
                curr_len = 0
            current.append(sent)
            curr_len += tok_est
    # flush final
    c = flush(i)
    if c:
        chunks.append(c)
    return chunks


if __name__ == "__main__":
    t = Path(__file__).parent.parent / "data" / "raw_docs" / "sample.txt"
    if t.exists():
        text = t.read_text()
        ch = chunk_text(text)
        print("Chunks:", len(ch))