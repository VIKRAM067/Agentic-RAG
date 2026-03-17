from fastapi import APIRouter
from apps.api.schemas import EvaluateRequest, EvaluateResponse
from apps.evaluation.ragas_eval import evaluate_rag

router = APIRouter()


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    scores = evaluate_rag(request.questions, request.ground_truths)
    return EvaluateResponse(**scores)
