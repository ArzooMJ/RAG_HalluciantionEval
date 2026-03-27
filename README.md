# RAG Hallucination Eval

This repo builds a small Retrieval-Augmented Generation (RAG) pipeline over a cleaned Reuters finance dataset, then runs a lightweight evaluation by asking a set of fixed questions and saving the generated answers.

## Directory structure

```
RAG_HalluciantionEval/
  src/
    rag/
      pipeline.py           # SimpleRAG: retrieve + answer with Groq
      evaluate_rag.py       # Runs dummy-question evaluation, saves results
    scripts/
      build_vector_db.py    # Builds FAISS index + pickles into artifacts/
      clean_dataset.py      # Cleans raw CSV into cleaned CSV
      dataset.py            # Parses Reuters SGM into CSV
      test_retrieval.py     # Quick retrieval sanity check

  data/
    finance_reuters_cleaned.csv  # Cleaned dataset used for embeddings

  artifacts/
    vector_store/
      faiss_index.bin
      document.pkl
      metadata.pkl
    eval/
      evaluation_results.json
      evaluation_results.md

  requirements.txt
  .env                     # local only (not committed)
```

## Setup

### 1) Create a virtual environment

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure Groq

Create a `.env` file at the repo root:

```bash
GROQ_API_KEY=your_key_here
```

## What’s happening (high level)

- **Index build** (`src/scripts/build_vector_db.py`)
  - Loads `data/finance_reuters_cleaned.csv`
  - Chunks each document into overlapping word chunks
  - Embeds chunks with `sentence-transformers` (`all-MiniLM-L6-v2`)
  - Writes a FAISS index + chunk text/metadata pickles to `artifacts/vector_store/`

- **RAG query** (`src/rag/pipeline.py`)
  - Loads the FAISS index + pickles from `artifacts/vector_store/`
  - Retrieves top-\(k\) chunks for a query
  - Calls Groq Chat Completions to generate an answer using the retrieved context

- **Evaluation** (`src/rag/evaluate_rag.py`)
  - Runs a fixed set of “dummy” finance questions through `SimpleRAG`
  - Saves results under `artifacts/eval/` as JSON + a small Markdown report

## Run it

All commands below assume you start from the repo root.

### 1) Build the vector store

```bash
python src/scripts/build_vector_db.py
```

Outputs:
- `artifacts/vector_store/faiss_index.bin`
- `artifacts/vector_store/document.pkl`
- `artifacts/vector_store/metadata.pkl`

### 2) Run evaluation

```bash
python src/rag/evaluate_rag.py
```

Outputs:
- `artifacts/eval/evaluation_results.json`
- `artifacts/eval/evaluation_results.md`

## Notes / common issues

- **Missing API key**: ensure `GROQ_API_KEY` is set in `.env` (repo root) or your shell env.
- **Vector store not found**: run the vector build step first so `artifacts/vector_store/` exists.

