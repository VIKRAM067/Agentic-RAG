# AgenticRAG 🤖
> A production-grade, self-correcting Agentic RAG system powered by LangGraph

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-green?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-orange?style=flat-square)](https://console.groq.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple?style=flat-square)](https://www.trychroma.com)
[![SQLite](https://img.shields.io/badge/SQLite-Memory-blue?style=flat-square&logo=sqlite)](https://sqlite.org)

---

## What is AgenticRAG?

Most RAG systems follow a fixed pipeline: retrieve → answer. **AgenticRAG is different.**

A LangGraph agent makes intelligent decisions at every step — routing queries to the best retrieval strategy, grading its own answers, and self-correcting with HyDE when quality falls below threshold. It remembers your conversations, isolates sessions per user, and knows which documents you've uploaded.

It doesn't just retrieve. It *thinks*.

---

## What's New (v2)

| Feature | Details |
|---|---|
| 🧠 Persistent Memory | Full conversation history stored in SQLite per session |
| 🗂 Session Management | Create, rename, switch, and delete sessions from the UI |
| 📂 File-Aware Routing | Ask "what documents do you have?" — agent fetches from ChromaDB |
| 🎯 Document Filtering | Say "search in X.pdf" — retrieval filters to that file only |
| 🔧 Bug Fixes | Fixed `return` inside loop bugs in ingestion, chain, and vector store |

---

## Basic RAG vs AgenticRAG

| Feature | Basic RAG | AgenticRAG |
|---|---|---|
| Retrieval Strategy | Fixed | Agent routes per query |
| Search Type | Dense only | Hybrid (BM25 + Dense + RRF) |
| Quality Check | ❌ None | ✅ Judge LLM grades every answer |
| Retries | ❌ One shot | ✅ Self-corrects with HyDE |
| Reranking | ❌ None | ✅ Cross-encoder reranker |
| Conversation Memory | ❌ None | ✅ SQLite per-session history |
| Session Management | ❌ None | ✅ Full UI — create, rename, delete |
| Document Filtering | ❌ None | ✅ Query specific PDFs by name |
| Evaluation | ❌ None | ✅ RAGAS metrics |

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│                     LangGraph Agent                       │
│                                                          │
│  ┌─────────┐    ┌──────────┐    ┌──────────────┐        │
│  │  Route  │───▶│ Retrieve │───▶│   Generate   │        │
│  └─────────┘    └──────────┘    └──────┬───────┘        │
│  hybrid /       BM25 + Dense           │                 │
│  semantic /     + RRF Fusion           ▼                 │
│  keyword /      + Reranker      ┌──────────────┐        │
│  direct /       + File Filter   │    Grade     │        │
│  files                          └──────┬───────┘        │
│     │                                  │                 │
│     ▼                        score ≥ 0.7?                │
│  ┌────────┐                  YES ──▶ Finalize            │
│  │ Files  │                  NO  ──▶ Retry (HyDE)        │
│  └────────┘                                              │
└──────────────────────────────────────────────────────────┘
    │
    ▼
Answer + Sources + Quality Score
    │
    ▼
┌──────────────────┐
│  Memory (SQLite) │  ← saves every turn per session
└──────────────────┘
```

---

## Core Concepts

### 🔀 Hybrid Search
Combines two retrieval methods:
- **Dense search** — semantic embeddings via ChromaDB (great for conceptual questions)
- **BM25 sparse search** — keyword matching (great for exact terms, names, formulas)
- **RRF Fusion** — merges both ranked lists, boosting docs that appear in both

### 🎯 Cross-Encoder Reranking
Two-pass retrieval for better precision:
1. Fast bi-encoder retrieves top N candidates
2. Slow cross-encoder scores each candidate against the query
3. Returns top K most relevant chunks

### 💡 HyDE (Hypothetical Document Embeddings)
On retries, the agent generates a hypothetical ideal answer and embeds *that* instead of the raw query. Answer-shaped vectors match document chunks far better than question-shaped vectors.

### 🔄 Self-Correcting Agent Loop
The agent grades its own output using a judge LLM. If quality falls below threshold (`GRADE_THRESHOLD=0.7`), it retries with HyDE — up to `MAX_RETRIES` times.

### 🧠 Persistent Conversation Memory
Every query and response is saved to SQLite, keyed by `session_id`. On each request, the last 20 messages are injected into the prompt as `chat_history`, giving the LLM full conversational context.

### 🗂 Session Isolation
Each conversation gets a unique `session_id` generated on the frontend. Sessions are fully isolated — rename them, switch between them, or delete them without affecting others.

### 📂 File-Aware Agent
A dedicated `files` route lets users ask "what documents do you have?" and get a live list from ChromaDB. Queries like "search in research_paper.pdf" automatically filter retrieval to that specific document.

### 📊 RAGAS Evaluation
Automated quality metrics:
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
| Memory | SQLite (via Python built-in sqlite3) |
| API | FastAPI |
| Evaluation | RAGAS |

---

## Project Structure

```
Agentic-RAG/
├── apps/
│   ├── core/
│   │   ├── config.py              # Central settings via pydantic-settings
│   │   └── prompts.py             # All LangChain prompt templates
│   ├── services/
│   │   ├── embeddings.py          # SentenceTransformer embedding manager
│   │   ├── vector_store.py        # ChromaDB wrapper with filter support
│   │   ├── ingestion.py           # PDF loading + chunking pipeline
│   │   ├── retriever.py           # Hybrid retriever (BM25 + Dense + RRF)
│   │   ├── reranker.py            # Cross-encoder reranker
│   │   ├── chain.py               # LCEL RAG chain
│   │   └── memory.py              # SQLite conversation memory manager
│   ├── agent/
│   │   └── graph.py               # LangGraph state machine + all nodes
│   ├── api/
│   │   ├── schemas.py             # Pydantic request/response models
│   │   └── routes/
│   │       ├── ingest.py          # POST /ingest
│   │       ├── query.py           # POST /query
│   │       ├── sessions.py        # GET/PUT/DELETE /sessions
│   │       └── evaluate.py        # POST /evaluate
│   ├── evaluation/
│   │   └── ragas_eval.py          # RAGAS evaluation pipeline
│   └── main.py                    # FastAPI entry point
├── data/
│   ├── uploads/                   # Uploaded PDFs
│   ├── vector_store/              # ChromaDB persistence
│   └── chat_memory.db             # SQLite conversation history
├── frontend.html                  # Single-file dark terminal UI
├── .env.example                   # Environment variable template
├── .gitignore
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

Open `frontend.html` directly in your browser.

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

### `POST /query`
Ask a question. Pass a `session_id` to enable conversation memory.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "what is gradient descent?", "session_id": "sess_abc123"}'
```

```json
{
  "answer": "Gradient descent is an optimization algorithm... [Source: lecture.pdf, Page: 3]",
  "sources": [{"source": "lecture.pdf", "page": 3, "score": 0.921}],
  "grade": {"score": 0.87, "reason": "answer is faithful and relevant"},
  "retry_count": 0,
  "session_id": "sess_abc123"
}
```

### `GET /sessions`
List all sessions.

### `PUT /sessions/{session_id}`
Rename a session.

```json
{ "name": "AI Research Chat" }
```

### `DELETE /sessions/{session_id}`
Delete a single session and its history.

### `DELETE /sessions`
Delete all sessions.

### `POST /evaluate`
Run RAGAS evaluation.

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "questions": ["what is gradient descent?"],
    "ground_truths": ["Gradient descent minimizes loss by following the negative gradient..."]
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

All settings in `.env`:

```env
# LLM
GROQ_API_KEY=...


```

---

## Author

**Vikram** — AI Engineer
Building production-grade AI systems from scratch.
Open to AI Engineering roles.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Vikram-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/vikramv2002/)