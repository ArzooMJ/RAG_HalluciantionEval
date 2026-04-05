import os
import json
from pathlib import Path
from typing import List, Dict, Any
from src.rag.baseline import SimpleRAG

class SelfRAG(SimpleRAG):
    """
    Self-RAG implementation that performs retrieval, verification, and correction.
    It extends SimpleRAG and adds logic for:
    1. Retrieval Grader: Is the retrieved context relevant to the query?
    2. Hallucination Grader: Is the generated answer grounded in the context?
    3. Answer Grader: Does the answer actually address the user's question?
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def grade_retrieval(self, query: str, context: str) -> bool:
        """Determines if the retrieved context is relevant to the query."""
        prompt = f"""### Role
You are a Relevance Grader. Your task is to assess whether the provided context is relevant to the user's question.

### Context
{context}

### User Question
{query}

### Instructions
- Respond with only 'YES' or 'NO'.
- 'YES' if the context contains information that can help answer the question.
- 'NO' if the context is completely irrelevant.

### Relevance (YES/NO):"""
        
        response = self.client.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        score = response.choices[0].message.content.strip().upper()
        return "YES" in score

    def grade_hallucination(self, context: str, answer: str) -> bool:
        """Determines if the answer is grounded in the provided context."""
        prompt = f"""### Role
You are a Hallucination Grader. Your task is to assess whether the generated answer is grounded in and supported by the provided context.

### Context
{context}

### Generated Answer
{answer}

### Instructions
- Respond with only 'YES' or 'NO'.
- 'YES' if the answer is fully supported by the context (no hallucinations).
- 'NO' if the answer contains information not present in the context.

### Grounded (YES/NO):"""
        
        response = self.client.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        score = response.choices[0].message.content.strip().upper()
        return "YES" in score

    def grade_answer(self, query: str, answer: str) -> bool:
        """Determines if the answer actually addresses the query."""
        prompt = f"""### Role
You are an Answer Grader. Your task is to assess whether the generated answer effectively addresses the user's question.

### User Question
{query}

### Generated Answer
{answer}

### Instructions
- Respond with only 'YES' or 'NO'.
- 'YES' if the answer directly addresses the question.
- 'NO' if the answer is irrelevant or fails to answer the question.

### Useful (YES/NO):"""
        
        response = self.client.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        score = response.choices[0].message.content.strip().upper()
        return "YES" in score

    def self_rag_query(self, query: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Performs Self-RAG: Retrieve -> Grade Retrieval -> Generate -> Grade Hallucination -> Grade Answer.
        """
        retries = 0
        context_docs = self.retrieve(query)
        
        while retries <= max_retries:
            context_text = "\n\n".join([doc["content"] for doc in context_docs])
            
            # 1. Grade Retrieval
            is_relevant = self.grade_retrieval(query, context_text)
            
            if not is_relevant and retries < max_retries:
                # If not relevant, try retrieving more/different docs (simplified here by just increasing k)
                retries += 1
                context_docs = self.retrieve(query, k=3 + retries)
                continue
            
            # 2. Generate Answer
            answer = self.generate_answer(query, context_docs)
            
            # 3. Grade Hallucination
            is_grounded = self.grade_hallucination(context_text, answer)
            
            if not is_grounded and retries < max_retries:
                # If hallucinating, try to regenerate with a stricter prompt or more context
                retries += 1
                continue
                
            # 4. Grade Answer usefulness
            is_useful = self.grade_answer(query, answer)
            
            if not is_useful and retries < max_retries:
                retries += 1
                continue
            
            # If we reach here, we either passed all checks or exhausted retries
            return {
                "query": query,
                "answer": answer,
                "context": context_docs,
                "retries": retries,
                "is_relevant": is_relevant,
                "is_grounded": is_grounded,
                "is_useful": is_useful
            }

        return {
            "query": query,
            "answer": "Failed to generate a verified answer after multiple attempts.",
            "context": context_docs,
            "retries": retries
        }

if __name__ == "__main__":
    if "GROQ_API_KEY" not in os.environ:
        print("Warning: GROQ_API_KEY not found in environment variables.")
    
    self_rag = SelfRAG()
    test_query = "What are the latest trends in the financial market?"
    try:
        result = self_rag.self_rag_query(test_query)
        print(f"Query: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"Verification - Relevant: {result.get('is_relevant')}, Grounded: {result.get('is_grounded')}, Useful: {result.get('is_useful')}")
    except Exception as e:
        print(f"Error running Self-RAG query: {e}")