import json
import pandas as pd
from pathlib import Path

from baseline import SimpleRAG
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT_DIR / "artifacts" / "eval"


def run_evaluation():
    # Initialize RAG
    rag = SimpleRAG()
    
    # 10 Dummy Questions based on financial news context
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
    
    print("Running RAG pipeline on dummy questions...")
    for q in tqdm(dummy_questions):
        try:
            res = rag.query(q)
            results.append({
                "question": q,
                "answer": res["answer"],
                "retrieved_context": [doc["content"] for doc in res["context"]]
            })
        except Exception as e:
            print(f"Error processing question '{q}': {e}")
            results.append({
                "question": q,
                "answer": "Error: " + str(e),
                "retrieved_context": []
            })
            
    # Save results to a JSON file
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Also save as a Markdown table for the report
    with open(EVAL_DIR / "evaluation_results.md", "w") as f:
        f.write("# RAG Evaluation Results\n\n")
        f.write("| Question | Answer |\n")
        f.write("| --- | --- |\n")
        for res in results:
            clean_answer = res["answer"].replace("\n", " ")
            f.write(f"| {res['question']} | {clean_answer} |\n")
            
    print(
        "Evaluation complete. Results saved to "
        f"{EVAL_DIR / 'evaluation_results.json'} and {EVAL_DIR / 'evaluation_results.md'}"
    )

if __name__ == "__main__":
    run_evaluation()