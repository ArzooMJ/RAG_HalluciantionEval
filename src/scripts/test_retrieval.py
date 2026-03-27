import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index("vector_store/faiss_index.bin")

# Load stored chunks
with open("vector_store/document.pkl", "rb") as f:
    documents = pickle.load(f)

# Load metadata
with open("vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

query = "Why did oil prices rise?"
query_embedding = model.encode([query])

k = 5
distances, indices = index.search(query_embedding, k)

print("\nTop retrieved chunks:\n")

for rank, idx in enumerate(indices[0], start=1):
    print(f"Result {rank}")
    print("Distance:", distances[0][rank - 1])
    print("Metadata:", metadata[idx])
    print("Text:", documents[idx][:500])
    print("-" * 80)