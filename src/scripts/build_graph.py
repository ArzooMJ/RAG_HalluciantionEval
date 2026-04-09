import pandas as pd
import pickle
import networkx as nx
import re
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "finance_reuters_cleaned.csv"
OUT_DIR = ROOT_DIR / "artifacts" / "graph_store"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Entity extraction
def extract_entities(text):
    return re.findall(r'\b[A-Z][a-zA-Z]+\b', text)

# Build graph
def build_graph(documents):
    G = nx.Graph()

    for doc in documents:
        entities = extract_entities(doc)

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1, e2 = entities[i], entities[j]

                if G.has_edge(e1, e2):
                    G[e1][e2]["contexts"].append(doc)
                else:
                    G.add_edge(e1, e2, contexts=[doc])

    return G

# Main
def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    documents = df["document"].tolist()
    print(f"Loaded {len(documents)} documents")

    print("Building graph...")
    graph = build_graph(documents)

    print("Saving graph...")
    with open(OUT_DIR / "graph.pkl", "wb") as f:
        pickle.dump(graph, f)

    with open(OUT_DIR / "documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    print("Graph build complete!")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()