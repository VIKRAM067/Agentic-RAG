# AgenticRAG 🤖
> A production-grade, self-correcting Agentic RAG system powered by LangGraph

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-green?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-orange?style=flat-square)](https://console.groq.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple?style=flat-square)](https://www.trychroma.com)

---

## What is AgenticRAG?

Most RAG systems follow a fixed pipeline: retrieve → answer. **AgenticRAG is different.**

A LangGraph agent makes intelligent decisions at every step — routing queries to the best retrieval strategy, grading its own answers, and self-correcting with HyDE when quality falls below threshold. It doesn't just retrieve; it *thinks*.

---

## Basic RAG vs AgenticRAG

| Feature | Basic RAG | AgenticRAG |
|---|---|---|
| Retrieval Strategy | Fixed | Agent routes per query |
| Search Type | Dense only | Hybrid (BM25 + Dense + RRF) |
| Quality Check | ❌ None | ✅ Judge LLM grades every answer |
| Retries | ❌ One shot | ✅ Self-corrects with HyDE |
| Reranking | ❌ None | ✅ Cross-encoder reranker |
| Evaluation | ❌ None | ✅ RAGAS metrics |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   LangGraph Agent                    │
│                                                      │
│  ┌─────────┐    ┌──────────┐    ┌──────────────┐   │
│  │  Route  │───▶│ Retrieve │───▶│   Generate   │   │
│  └─────────┘    └──────────┘    └──────┬───────┘   │
│  decides:       BM25 + Dense           │            │
│  hybrid /       + RRF Fusion           ▼            │
│  semantic /     + Reranker      ┌──────────────┐   │
│  keyword /                      │    Grade     │   │
│  direct                         └──────┬───────┘   │
│                                        │            │
│                              score ≥ 0.7?           │
│                              YES ──▶ Finalize       │
│                              NO  ──▶ Retry (HyDE)  │
└─────────────────────────────────────────────────────┘
    │
    ▼
Answer + Sources + Quality Score
```

---

## Core Concepts

### 🔀 Hybrid Search
Combines two retrieval methods for maximum coverage:
- **Dense search** — semantic embeddings via ChromaDB (great for conceptual questions)
- **BM25 sparse search** — keyword matching (great for exact terms, formulas, names)
- **RRF Fusion** — merges both ranked lists, boosting documents that appear in both

### 🎯 Cross-Encoder Reranking
Two-pass retrieval for better precision:
1. Fast bi-encoder retrieves top 5 candidates
2. Slow cross-encoder scores each candidate against the query — much more accurate
3. Returns top 3 most relevant chunks

### 💡 HyDE (Hypothetical Document Embeddings)
On retries, instead of embedding the raw query, the agent generates a hypothetical ideal answer and embeds *that*. Answer-shaped vectors match document chunks far better than question-shaped vectors.

### 🔄 Self-Correcting Agent Loop
The agent grades its own output using a judge LLM. If the quality score falls below threshold (default: `0.7`), it retries with a different retrieval strategy — up to `MAX_RETRIES` times.

### 📊 RAGAS Evaluation
Automated quality metrics on every run:
- **Faithfulness** — is the answer grounded in retrieved context?
- **Answer Relevancy** — does it address the question?
- **Context Precision** — were retrieved chunks actually useful?
- **Context Recall** — did retrieval find all needed information?

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Sparse Search | rank-bm25 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM Framework | LangChain |
| API | FastAPI |
| Evaluation | RAGAS |

---

## Project Structure

```
Agentic-RAG/
├── apps/
│   ├── core/
│   │   ├── config.py          # Central settings via pydantic-settings
│   │   └── prompts.py         # All LangChain prompt templates
│   ├── services/
│   │   ├── embeddings.py      # SentenceTransformer embedding manager
│   │   ├── vector_store.py    # ChromaDB wrapper
│   │   ├── ingestion.py       # PDF loading + chunking pipeline
│   │   ├── retriever.py       # Hybrid retriever (BM25 + Dense + RRF)
│   │   ├── reranker.py        # Cross-encoder reranker
│   │   └── chain.py           # LCEL RAG chain
│   ├── agent/
│   │   └── graph.py           # LangGraph state machine + all nodes
│   ├── api/
│   │   ├── schemas.py         # Pydantic request/response models
│   │   └── routes/
│   │       ├── ingest.py      # POST /ingest
│   │       ├── query.py       # POST /query
│   │       └── evaluate.py    # POST /evaluate
│   ├── evaluation/
│   │   └── ragas_eval.py      # RAGAS evaluation pipeline
│   └── main.py                # FastAPI entry point
├── data/
│   ├── uploads/               # Uploaded PDFs
│   └── vector_store/          # ChromaDB persistence
├── frontend.html              # Single-file frontend
├── .env.example               # Environment variable template
└── requirements.txt
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/VIKRAM067/Agentic-RAG.git
cd Agentic-RAG
python -m venv .RAGvenv
source .RAGvenv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 3. Run

```bash
uvicorn apps.main:app --reload
```

### 4. Open the frontend

Open `frontend.html` in your browser or visit:
```
http://localhost:8000/docs
```

---

## API Reference

### `POST /ingest`
Upload a PDF for indexing.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@your_document.pdf"
```

```json
{
  "status": "success",
  "filename": "your_document.pdf",
  "pages": 12,
  "chunks": 87
}
```

---

### `POST /query`
Ask a question about your uploaded documents.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "what is gradient descent?"}'
```

```json
{
  "answer": "Gradient descent is an optimization algorithm... [Source: lecture.pdf, Page: 3]",
  "sources": [{"source": "lecture.pdf", "page": 3, "score": 0.921}],
  "grade": {"score": 0.87, "reason": "answer is faithful and relevant"},
  "retry_count": 0
}
```

---

### `POST /evaluate`
Run RAGAS evaluation on your system.

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "questions": ["what is gradient descent?"],
    "ground_truths": ["Gradient descent is an optimization algorithm that minimizes loss..."]
  }'
```

```json
{
  "faithfulness": 0.91,
  "answer_relevancy": 0.88,
  "context_precision": 0.79,
  "context_recall": 0.83
}
```

---

## Configuration

All settings configurable via `.env`:

```env
# LLM
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.3

# Retrieval
TOP_K=5
BM25_WEIGHT=0.4
DENSE_WEIGHT=0.6

# Reranker
RERANKER_TOP_N=3

# Agent
MAX_RETRIES=2
GRADE_THRESHOLD=0.7
```

---

## Author

**Vikram** — Junior AI Engineer
Building production-grade AI systems.
Open to AI Engineering roles.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Vikram-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/vikramv2002/)