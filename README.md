# RAG Hallucination Eval

This repository builds retrieval-augmented generation (RAG) over a cleaned Reuters finance corpus. It includes a **dense vector baseline** (FAISS + sentence embeddings), **BM25** lexical retrieval, and **Self-RAG** (retrieval + LLM-based relevance, grounding, and usefulness checks). A **Streamlit** app runs the three approaches on the same question and shows answers side by side, with Self-RAG verification metrics and short comparison hints.

## Directory structure

```
RAG_HalluciantionEval/
  app.py                      # Streamlit UI: compare Vector RAG, Self-RAG, BM25

  src/
    rag/
      baseline.py             # SimpleRAG: FAISS retrieve + Groq generation
      self_rag.py             # SelfRAG: extends baseline with grading / retries
      bm25_rag.py             # BM25RAG: BM25 retrieve + Groq generation
      evaluate_rag.py         # Batch eval: baseline on dummy questions
      self_rag_evaluate.py    # Batch eval: Self-RAG
      bm25_evaluate.py        # Batch eval: BM25

    scripts/
      build_vector_db.py      # Builds FAISS index → artifacts/vector_store/
      build_bm25_index.py     # Builds BM25 index → artifacts/bm25_store/
      clean_dataset.py        # Cleans raw CSV → cleaned CSV
      dataset.py              # Parses Reuters SGM → CSV
      test_retrieval.py       # Quick check against vector store

  data/
    finance_reuters_cleaned.csv   # Corpus used by index scripts (expected path)

  artifacts/
    vector_store/             # faiss_index.bin, document.pkl, metadata.pkl
    bm25_store/               # bm25_index.pkl, documents.pkl, metadata.pkl
    eval/                     # optional batch evaluation outputs

  requirements.txt
  .env                        # local only (not committed)
```

## Setup

### 1) Virtual environment

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2) Dependencies

```bash
pip install -r requirements.txt
```

On GPU-only setups you may swap **`faiss-cpu`** for a suitable **`faiss`** build; otherwise the file as listed targets CPU.

### 3) Groq API key

Create a `.env` file at the repo root:

```bash
GROQ_API_KEY=your_key_here
```

Optional: override the chat model (default is `llama-3.1-8b-instant`):

```bash
GROQ_MODEL=llama-3.1-8b-instant
```

### 4) Data

Index scripts expect:

`data/finance_reuters_cleaned.csv`

If you are starting from raw Reuters SGM files or a raw CSV, use `src/scripts/dataset.py` and `src/scripts/clean_dataset.py` as needed, and place or symlink the final cleaned file at `data/finance_reuters_cleaned.csv`.

---

## Building indexes (what to run, in order)

### Baseline & Self-RAG: vector database

**Vector RAG** (`SimpleRAG` in `baseline.py`) and **Self-RAG** (`self_rag.py`) both use the same FAISS store under `artifacts/vector_store/`. Build it first:

```bash
python src/scripts/build_vector_db.py
```

This reads `data/finance_reuters_cleaned.csv`, chunks text (see script for column `document` / chunking), embeds with `sentence-transformers` (`all-MiniLM-L6-v2`), and writes:

- `artifacts/vector_store/faiss_index.bin`
- `artifacts/vector_store/document.pkl`
- `artifacts/vector_store/metadata.pkl`

Without this step, baseline and Self-RAG will fail when loading the index.

### BM25: lexical index

**BM25 RAG** uses a separate on-disk index. After you have `data/finance_reuters_cleaned.csv`, build it with:

```bash
python src/scripts/build_bm25_index.py
```

This writes:

- `artifacts/bm25_store/bm25_index.pkl`
- `artifacts/bm25_store/documents.pkl`
- `artifacts/bm25_store/metadata.pkl`

Chunking and tokenization are defined in that script (BM25 uses the `text` column).

### Optional checks

- `python src/scripts/test_retrieval.py` — quick sanity check on the vector store.

---

## Streamlit app: compare models side by side

With **both** the vector store and BM25 artifacts built, and `GROQ_API_KEY` set, run from the **repo root**:

```bash
streamlit run app.py
```

The UI lets you enter a question and compare, in three columns:

1. **Vector RAG** — dense retrieval + generation  
2. **Self-RAG** — same retriever as baseline, plus relevance / grounding / usefulness grading and retries  
3. **BM25 RAG** — lexical retrieval + generation  

Each column shows the answer and expandable retrieved chunks (BM25 shows scores). Below that, Self-RAG verification metrics and short notes compare answers across methods.

---

## Optional: batch evaluation scripts

These run fixed dummy finance questions and write results under `artifacts/eval/` (see each script for exact filenames). They assume the corresponding indexes already exist.

- Baseline: run from `src/rag/` (imports use `baseline`):  
  `cd src/rag && python evaluate_rag.py`
- Self-RAG: `cd src/rag && python self_rag_evaluate.py`
- BM25: `cd src/rag && python bm25_evaluate.py`
