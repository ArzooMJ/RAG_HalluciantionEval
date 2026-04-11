import os
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
 
ROOT_DIR = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIR = ROOT_DIR / "artifacts" / "vector_store"

class SimpleRAG:
    def __init__(
        self,
        index_path: str | Path | None = None,
        doc_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        groq_model: str | None = None,
    ):
        load_dotenv()

        index_path = str(index_path or (VECTOR_STORE_DIR / "faiss_index.bin"))
        doc_path = str(doc_path or (VECTOR_STORE_DIR / "document.pkl"))
        metadata_path = str(metadata_path or (VECTOR_STORE_DIR / "metadata.pkl"))
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load documents and metadata
        with open(doc_path, "rb") as f:
            self.documents = pickle.load(f)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
            
        # Load embedding model
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize Groq client
        # It will automatically look for GROQ_API_KEY environment variable
        self.client = Groq()
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def retrieve(self, query, k=3):
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append({
                    "content": self.documents[idx],
                    "metadata": self.metadata[idx]
                })
        return results

    def generate_answer(self, query, context_docs):
        context_text = "\n\n".join([doc["content"] for doc in context_docs])
        
        prompt = f"""Context:
{context_text}

Question: {query}

Answer only from the provided context. If the context does not contain enough information to answer, say: I do not have enough information."""

        response = self.client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": "You answer questions using the context the user provides."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content

    def query(self, query):
        context_docs = self.retrieve(query)
        answer = self.generate_answer(query, context_docs)
        return {
            "query": query,
            "answer": answer,
            "context": context_docs
        }

if __name__ == "__main__":
    # Note: Ensure GROQ_API_KEY is set in environment
    if "GROQ_API_KEY" not in os.environ:
        print("Warning: GROQ_API_KEY not found in environment variables.")
    
    rag = SimpleRAG()
    
    test_query = "What are the latest trends in the financial market?"
    try:
        result = rag.query(test_query)
        print(f"Query: {result['query']}")
        print(f"Answer: {result['answer']}")
    except Exception as e:
        print(f"Error running query: {e}")