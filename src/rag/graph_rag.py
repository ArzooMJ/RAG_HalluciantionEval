import pickle
import re
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Simple entity extraction
def extract_entities(text):
    return re.findall(r'\b[A-Z][a-zA-Z]+\b', text)

# Graph RAG
class GraphRAG:
    def __init__(self, groq_model=None):
        from pathlib import Path

        ROOT_DIR = Path(__file__).resolve().parents[2]

        graph_path = ROOT_DIR / "artifacts" / "graph_store" / "graph.pkl"
        documents_path = ROOT_DIR / "artifacts" / "graph_store" / "documents.pkl"

        with open(graph_path, "rb") as f:
            self.graph = pickle.load(f)

        with open(documents_path, "rb") as f:
            self.documents = pickle.load(f)

        self.client = Groq()
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    # Retrieve
    def retrieve(self, query, top_k=5):
        query_entities = extract_entities(query)
        results = []

        # Graph retrieval
        for ent in query_entities:
            if ent in self.graph:
                neighbors = list(self.graph.neighbors(ent))

                for n in neighbors:
                    edge_data = self.graph.get_edge_data(ent, n)
                    if edge_data:
                        results.extend(edge_data.get("contexts", []))

        # Fallback
        if len(results) == 0:
            results = [
                doc for doc in self.documents
                if any(word.lower() in doc.lower() for word in query.split())
            ]

        # Remove duplicates
        seen = set()
        unique_docs = []
        for doc in results:
            if doc not in seen:
                seen.add(doc)
                unique_docs.append(doc)

        # Simple ranking
        unique_docs.sort(
            key=lambda d: sum(word.lower() in d.lower() for word in query.split()),
            reverse=True
        )

        return unique_docs[:top_k]

    # Generate answer
    def generate_answer(self, query, docs):
        context = "\n\n".join(docs)

        prompt = f"""### Role
You are a financial assistant.

### Context
{context}

### Question
{query}

### Instructions
Answer using ONLY the context. If insufficient, say you do not have enough information.

### Answer:"""

        response = self.client.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content

    # Main query
    def query(self, query):
        docs = self.retrieve(query)
        answer = self.generate_answer(query, docs)

        return {
            "query": query,
            "answer": answer,
            "context": [{"content": d} for d in docs]
        }

# Test
if __name__ == "__main__":
    rag = GraphRAG()

    query = "What did IBM report about its earnings?"
    result = rag.query(query)

    print("Query:", result["query"])
    print("Answer:", result["answer"])
    print("Retrieved Docs:", len(result["context"]))