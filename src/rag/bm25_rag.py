import os
import re
import pickle
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from groq import Groq
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
BM25_STORE_DIR = ROOT_DIR / "artifacts" / "bm25_store"


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


class BM25RAG:
    def __init__(
        self,
        index_path: str | Path | None = None,
        doc_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        groq_model: str | None = None,
        k: int = 3,
    ):
        load_dotenv()

        index_path    = index_path    or (BM25_STORE_DIR / "bm25_index.pkl")
        doc_path      = doc_path      or (BM25_STORE_DIR / "documents.pkl")
        metadata_path = metadata_path or (BM25_STORE_DIR / "metadata.pkl")

        with open(index_path, "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)
        with open(doc_path, "rb") as f:
            self.documents: list[str] = pickle.load(f)
        with open(metadata_path, "rb") as f:
            self.metadata: list[dict] = pickle.load(f)

        self.k = k
        self.client = Groq()
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        k = k or self.k
        scores = self.bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            {
                "content":  self.documents[i],
                "metadata": self.metadata[i],
                "score":    float(scores[i]),
            }
            for i in top_idx
        ]

    def generate_answer(self, query: str, context_docs: list[dict]) -> str:
        context_text = "\n\n".join([doc["content"] for doc in context_docs])

        prompt = f"""### Role
You are a highly analytical Financial Research Assistant. Your task is to provide accurate, data-driven answers based strictly on the provided financial news context.

### Context
{context_text}

### User Question
{query}

### Instructions
1. Groundedness: Answer the question using ONLY the provided context. If the context does not contain enough information to answer the question, state clearly: "Based on the provided reports, I do not have enough information to answer this question."
2. Structure: 
   - Use a clear, professional tone.
   - If relevant, quantify data (dates, percentages, currency values) exactly as they appear in the text.

### Answer:"""

        response = self.client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": "You are a helpful financial expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content

    def query(self, query: str) -> dict:
        context_docs = self.retrieve(query)
        answer = self.generate_answer(query, context_docs)
        return {
            "query":   query,
            "answer":  answer,
            "context": context_docs,
        }


if __name__ == "__main__":
    if "GROQ_API_KEY" not in os.environ:
        print("Warning: GROQ_API_KEY not found in environment variables.")

    rag = BM25RAG()

    test_query = "What are the latest trends in the financial market?"
    try:
        result = rag.query(test_query)
        print(f"Query: {result['query']}")
        print(f"Answer: {result['answer']}")
    except Exception as e:
        print(f"Error running query: {e}")