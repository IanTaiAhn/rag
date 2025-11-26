# ingestion/text_loader.py
from pathlib import Path


def load_text_file(path: str) -> str:
    return Path(path).read_text()


if __name__ == "__main__":
    p = Path(__file__).parent.parent / "data" / "raw_docs"
    for f in p.glob("*.txt"):
        print(f, len(f.read_text()))