import os
import pickle
from pathlib import Path
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT_DIR / "data" / "finance_reuters_cleaned.csv"
VECTOR_STORE_DIR = ROOT_DIR / "artifacts" / "vector_store"
TEXT_column="document"

df = pd.read_csv(CSV_PATH)

df=df.dropna(subset=[TEXT_column]).copy()
df[TEXT_column]=df[TEXT_column].astype(str)
df=df[df[TEXT_column].str.strip()!=""]

print(f"Rows loaded: {len(df)}")

def chunk_text(text, chunk_size=120, overlap=30):
    """
    Split text into chunks by words instead of characters.

    chunk_size = number of words per chunk
    overlap = number of words shared between consecutive chunks
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start += chunk_size - overlap

    return chunks

document=[]
metadata=[]

for idx,row in df.iterrows():
    full_text=row[TEXT_column]
    title=row["title"] if "title" in df.columns and pd.notna(row["title"]) else ""
    topics=row["topics"] if "topics" in df.columns and pd.notna(row["topics"]) else ""

    chunks=chunk_text(full_text, chunk_size=500, overlap=100)

    for chunk_id, chunk in enumerate(chunks):
        document.append(chunk)
        metadata.append({
            "row_index": int(idx),
            "chunk_id": int(chunk_id),
            "title": str(title),
            "topics": str(topics)
        })

print(f"Total chunks created: {len(document)}")

model=SentenceTransformer("all-MiniLM-L6-v2")

embeddings=model.encode(document, show_progress_bar=True, convert_to_numpy=True)

dimension=embeddings.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(embeddings)

VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(VECTOR_STORE_DIR / "faiss_index.bin"))

with open(VECTOR_STORE_DIR / "document.pkl", "wb") as f:
    pickle.dump(document, f)

with open(VECTOR_STORE_DIR / "metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("Vector DB built successfully.")
print("Saved files:")
print(f" - {VECTOR_STORE_DIR / 'faiss_index.bin'}")
print(f" - {VECTOR_STORE_DIR / 'document.pkl'}")
print(f" - {VECTOR_STORE_DIR / 'metadata.pkl'}")
