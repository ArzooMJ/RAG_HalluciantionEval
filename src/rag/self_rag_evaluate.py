import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from self_rag import SelfRAG

ROOT_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT_DIR / "artifacts" / "eval"

def run_self_rag_evaluation():
    # Initialize Self-RAG
    self_rag = SelfRAG()
    
    # Same 10 Dummy Questions for comparison
    dummy_questions = [
        "What are the main factors affecting oil prices according to the reports?",
        "How did the stock market respond to the recent inflation data?",
        "What are the major concerns regarding the banking sector's stability?",
        "How are interest rate hikes impacting consumer spending?",
        "What are the latest updates on the merger and acquisition activities in the tech industry?",
        "How is the global trade tension affecting the manufacturing sector?",
        "What are the projections for the GDP growth in the next quarter?",
        "How are cryptocurrency regulations evolving in major economies?",
        "What are the key highlights from the recent corporate earnings reports?",
        "How is the renewable energy sector performing compared to traditional energy?"
    ]
    
    results = []
    
    print("Running Self-RAG pipeline on dummy questions...")
    for q in tqdm(dummy_questions):
        try:
            res = self_rag.self_rag_query(q)
            results.append({
                "question": q,
                "answer": res["answer"],
                "retrieved_context": [doc["content"] for doc in res["context"]],
                "retries": res.get("retries", 0),
                "is_relevant": res.get("is_relevant", False),
                "is_grounded": res.get("is_grounded", False),
                "is_useful": res.get("is_useful", False)
            })
        except Exception as e:
            print(f"Error processing question '{q}': {e}")
            results.append({
                "question": q,
                "answer": "Error: " + str(e),
                "retrieved_context": [],
                "retries": 0,
                "is_relevant": False,
                "is_grounded": False,
                "is_useful": False
            })
            
    # Save results to a JSON file
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DIR / "self_rag_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Also save as a Markdown table for the report
    with open(EVAL_DIR / "self_rag_evaluation_results.md", "w") as f:
        f.write("# Self-RAG Evaluation Results\n\n")
        f.write("| Question | Answer | Relevant | Grounded | Useful | Retries |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for res in results:
            clean_answer = res["answer"].replace("\n", " ")
            f.write(f"| {res['question']} | {clean_answer} | {res['is_relevant']} | {res['is_grounded']} | {res['is_useful']} | {res['retries']} |\n")
            
    print(
        "Self-RAG Evaluation complete. Results saved to "
        f"{EVAL_DIR / 'self_rag_evaluation_results.json'} and {EVAL_DIR / 'self_rag_evaluation_results.md'}"
    )

if __name__ == "__main__":
    run_self_rag_evaluation()