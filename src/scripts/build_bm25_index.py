import re
import pickle
import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi

ROOT_DIR  = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "finance_reuters_cleaned.csv"
OUT_DIR   = ROOT_DIR / "artifacts" / "bm25_store"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE    = 100
CHUNK_OVERLAP = 20


def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if chunk:
            chunks.append(chunk)
    return chunks


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} documents from {DATA_PATH}")

    all_chunks, all_metadata = [], []

    for _, row in df.iterrows():
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        if not text.strip():
            continue
        for chunk in chunk_text(text):
            all_chunks.append(chunk)
            all_metadata.append({"source": str(row.get("title", "unknown"))})

    print(f"Total chunks created: {len(all_chunks)}")

    tokenized = [tokenize(c) for c in all_chunks]
    bm25 = BM25Okapi(tokenized)

    pickle.dump(bm25,         open(OUT_DIR / "bm25_index.pkl",  "wb"))
    pickle.dump(all_chunks,   open(OUT_DIR / "documents.pkl",   "wb"))
    pickle.dump(all_metadata, open(OUT_DIR / "metadata.pkl",    "wb"))

    print(f"BM25 index saved → {OUT_DIR}")


if __name__ == "__main__":
    main()