from pydantic import BaseModel
from typing import List, Optional


# ── Ingest ─────────────────────────────────────────────
class IngestResponse(BaseModel):
    status: str
    filename: str
    pages: int
    chunks: int


# ── Query ──────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    session_id: str
    chat_history: list = []


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    grade: dict
    retry_count: int
    session_id: str


# ── Evaluate ───────────────────────────────────────────
class EvaluateRequest(BaseModel):
    questions: List[str]
    ground_truths: List[str]


class EvaluateResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
