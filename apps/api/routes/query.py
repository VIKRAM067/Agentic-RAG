from fastapi import APIRouter
from apps.api.schemas import QueryRequest, QueryResponse
from apps.agent.graph import agent
from apps.services.memory import memory_manager
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):

    # create or get session
    memory_manager.get_or_create_session(request.session_id)

    # fetch history and convert to LangChain message objects
    raw_history = memory_manager.get_history(request.session_id, last_n=20)
    formatted_history = []
    for msg in raw_history:
        if msg["role"] == "user":
            formatted_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_history.append(AIMessage(content=msg["content"]))

    # invoke the agent with initial state
    result = agent.invoke(
        {
            "question": request.question,
            "strategy": "hybrid",
            "chat_history": formatted_history,
            "documents": [],
            "context": "",
            "answer": "",
            "grade": {},
            "retry_count": 0,
            "final_answer": "",
            "filter_source": None,
        }
    )

    # save messages to memory
    memory_manager.add_message(request.session_id, "user", request.question)
    memory_manager.add_message(request.session_id, "assistant", result["final_answer"])

    # extract sources from documents
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "score": doc.metadata.get("reranker_score", 0),
        }
        for doc in (result["documents"] or [])
    ]

    return QueryResponse(
        answer=result["final_answer"],
        sources=sources,
        grade=result["grade"],
        retry_count=result["retry_count"],
        session_id=request.session_id,
    )
