import json
from pathlib import Path
from tqdm import tqdm

from graph_rag import GraphRAG


# -----------------------------
# SIMPLE GRADER (LLM-based)
# -----------------------------
class Evaluator:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def _ask(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip().upper()

    def grade_grounded(self, context, answer):
        prompt = f"""
Context:
{context}

Answer:
{answer}

Is the answer fully supported by the context? Answer YES or NO.
"""
        return "YES" in self._ask(prompt)

    def grade_relevant(self, query, context):
        prompt = f"""
Query:
{query}

Context:
{context}

Is the context relevant to the query? Answer YES or NO.
"""
        return "YES" in self._ask(prompt)

    def grade_useful(self, query, answer):
        prompt = f"""
Query:
{query}

Answer:
{answer}

Does the answer address the query? Answer YES or NO.
"""
        return "YES" in self._ask(prompt)


# -----------------------------
# MAIN EVALUATION
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT_DIR / "artifacts" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def run_evaluation():
    rag = GraphRAG()

    evaluator = Evaluator(rag.client, rag.groq_model)

    test_questions = [
        "What did IBM report about earnings?",
        "What were the financial results of Swire Pacific Ltd?",
        "How did LYPHOMED INC perform?",
        "Which companies reported strong earnings?",
        "What are the latest trends in the financial market?",
        "What is the impact of cryptocurrency regulation?",
        "How are interest rates affecting the economy?",
        "What are the key highlights from corporate earnings?",
    ]

    results = []

    print("Running Graph RAG evaluation...\n")

    grounded_count = 0
    relevant_count = 0
    useful_count = 0

    for q in tqdm(test_questions):
        try:
            res = rag.query(q)

            context_text = "\n\n".join([d["content"] for d in res["context"]])

            is_grounded = evaluator.grade_grounded(context_text, res["answer"])
            is_relevant = evaluator.grade_relevant(q, context_text)
            is_useful = evaluator.grade_useful(q, res["answer"])

            grounded_count += int(is_grounded)
            relevant_count += int(is_relevant)
            useful_count += int(is_useful)

            results.append({
                "question": q,
                "answer": res["answer"],
                "grounded": is_grounded,
                "relevant": is_relevant,
                "useful": is_useful,
                "retrieved_docs": len(res["context"])
            })

        except Exception as e:
            results.append({
                "question": q,
                "answer": f"ERROR: {str(e)}",
                "grounded": False,
                "relevant": False,
                "useful": False,
                "retrieved_docs": 0
            })

    # -----------------------------
    # METRICS
    # -----------------------------
    total = len(results)

    summary = {
        "total_questions": total,
        "grounded_rate": grounded_count / total,
        "relevance_rate": relevant_count / total,
        "usefulness_rate": useful_count / total,
        "hallucination_rate": 1 - (grounded_count / total)
    }

    # -----------------------------
    # SAVE JSON
    # -----------------------------
    with open(EVAL_DIR / "graph_evaluation_results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=4)

    # -----------------------------
    # SAVE MARKDOWN (for report)
    # -----------------------------
    with open(EVAL_DIR / "graph_evaluation_results.md", "w") as f:
        f.write("# Graph RAG Evaluation Results\n\n")

        f.write("## Summary\n")
        for k, v in summary.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Detailed Results\n")
        f.write("| Question | Grounded | Relevant | Useful |\n")
        f.write("|---|---|---|---|\n")

        for r in results:
            f.write(f"| {r['question']} | {r['grounded']} | {r['relevant']} | {r['useful']} |\n")

    print("\n Evaluation Complete!")
    print("Summary:", summary)


if __name__ == "__main__":
    run_evaluation()