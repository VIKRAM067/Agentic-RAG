from fastapi import APIRouter
from apps.api.schemas import QueryRequest, QueryResponse
from apps.agent.graph import agent

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):

    # invoke the agent with initial state
    result = agent.invoke(
        {
            "question": request.question,
            "strategy": "hybrid",
            "documents": [],
            "context": "",
            "answer": "",
            "grade": {},
            "retry_count": 0,
            "final_answer": "",
        }
    )

    # extract sources from documents
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "score": doc.metadata.get("reranker_score", 0),
        }
        for doc in result["documents"]
    ]

    return QueryResponse(
        answer=result["final_answer"],
        sources=sources,
        grade=result["grade"],
        retry_count=result["retry_count"],
    )
