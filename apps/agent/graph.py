from langgraph.graph import StateGraph, END
from langsmith import traceable
from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from apps.core.config import settings
from apps.core.prompts import router_prompt
from apps.services.retriever import hybrid_retriever
from apps.services.reranker import reranker
from apps.services.chain import get_llm, format_context, grade_answer, rag_chain
from apps.services.vector_store import vector_store
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
import json

import os

print("LANGCHAIN_TRACING_V2 =", os.getenv("LANGCHAIN_TRACING_V2"))
print("LANGCHAIN_PROJECT =", os.getenv("LANGCHAIN_PROJECT"))
print("LANGCHAIN_API_KEY exists =", bool(os.getenv("LANGCHAIN_API_KEY")))


class AgentState(TypedDict):
    question: str
    strategy: str
    chat_history: List[dict]
    documents: List[Document]
    context: str
    answer: str
    grade: dict
    retry_count: int
    final_answer: str
    filter_source: Optional[str]


def extract_filename(question: str) -> Optional[str]:
    question_lower = question.lower()
    all_docs = vector_store.collection.get(include=["metadatas"])
    filenames = set(m.get("source", "") for m in all_docs["metadatas"])
    for filename in filenames:
        name_without_ext = filename.lower().replace(".pdf", "").replace("_", " ")
        if name_without_ext in question_lower or filename.lower() in question_lower:
            return filename
    return None  # ← outside loop


@traceable(name="route_Node")  # for trace visibility in LangSmith
def route_Node(state: AgentState) -> AgentState:
    llm = get_llm()
    router_chain = router_prompt | llm | StrOutputParser()
    result = router_chain.invoke({"question": state["question"]})
    try:
        parsed = json.loads(result)
        strategy = parsed.get("strategy", "hybrid")
    except json.JSONDecodeError:
        strategy = "hybrid"

    filter_source = extract_filename(state["question"])
    return {**state, "strategy": strategy, "filter_source": filter_source}


@traceable(name="direct_node")  # for trace visibility in LangSmith
def direct_node(state: AgentState) -> AgentState:
    llm = get_llm()
    recent_history = state.get("chat_history", [])

    messages = [
        SystemMessage(
            content="""You are a helpful assistant with memory of this conversation.
CRITICAL RULES:
- The LAST message in history always overrides earlier messages
- If the user updated their name, use the LATEST name only
- If the user corrected any information, always use the MOST RECENT version
- Never contradict what was most recently established"""
        ),
        *recent_history,
        HumanMessage(content=state["question"]),
    ]

    answer = llm.invoke(messages).content
    return {
        **state,
        "answer": answer,
        "context": "",
        "documents": [],
        "grade": {"score": 1.0, "reason": "direct answer"},
    }


@traceable(name="files_node")  # for trace visibility in LangSmith
def files_node(state: AgentState) -> AgentState:
    all_docs = vector_store.collection.get(include=["metadatas"])
    filenames = list(set(m.get("source", "unknown") for m in all_docs["metadatas"]))

    if filenames:
        file_list = "\n".join(f"- {f}" for f in filenames)
        answer = f"Here are the documents I have access to:\n{file_list}"
    else:
        answer = "No documents are currently indexed."

    return {  # ← outside both if/else blocks
        **state,
        "answer": answer,
        "context": "",
        "documents": [],
        "grade": {"score": 1.0, "reason": "files list"},
    }


@traceable(name="retrieve_Node")  # for trace visibility in LangSmith
def retrieve_Node(state: AgentState) -> AgentState:
    hybrid_retriever.strategy = state["strategy"]
    hybrid_retriever.filter_source = state.get("filter_source")

    docs = hybrid_retriever.invoke(state["question"])
    reranked_docs = reranker.rerank(state["question"], docs)
    context = format_context(reranked_docs)

    return {**state, "documents": reranked_docs, "context": context}


@traceable(name="generate_Node")  # for trace visibility
def generate_Node(state: AgentState) -> AgentState:
    answer = rag_chain.invoke(
        {
            "question": state["question"],
            "context": state["context"],
            "chat_history": state.get("chat_history", []),
        }
    )
    return {**state, "answer": answer}


@traceable(name="grade_Node")
def grade_Node(state: AgentState) -> AgentState:
    grade = grade_answer(state["question"], state["context"], state["answer"])
    return {**state, "grade": grade}


@traceable(name="finalize_Node")
def finalize_node(state: AgentState) -> AgentState:
    return {**state, "final_answer": state["answer"]}


@traceable(name="retry_Node")
def retry_node(state: AgentState) -> AgentState:
    retry_count = state.get("retry_count", 0) + 1
    return {**state, "retry_count": retry_count}


def should_retry_node(state: AgentState) -> str:
    grade = state.get("grade", {})
    retry_count = state.get("retry_count", 0)

    if (
        grade.get("score", 0) < settings.grade_threshold
        and retry_count < settings.max_retries
    ):
        return "retry"
    else:
        return "finalize"


def build_agent():
    workflow = StateGraph(AgentState)

    # add all nodes
    workflow.add_node("route", route_Node)
    workflow.add_node("retrieve", retrieve_Node)
    workflow.add_node("generate", generate_Node)
    workflow.add_node("grade", grade_Node)
    workflow.add_node("retry", retry_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("direct", direct_node)
    workflow.add_node("files", files_node)

    # entry point
    workflow.set_entry_point("route")

    # fixed edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "grade")
    workflow.add_edge("retry", "retrieve")
    workflow.add_edge("direct", "finalize")
    workflow.add_edge("files", "finalize")
    workflow.add_edge("finalize", END)

    # conditional edge after route
    workflow.add_conditional_edges(
        "route",
        lambda s: (
            "direct"
            if s["strategy"] == "direct"
            else "files" if s["strategy"] == "files" else "retrieve"
        ),
        {"direct": "direct", "files": "files", "retrieve": "retrieve"},
    )

    # conditional edge — retry or finalize?
    workflow.add_conditional_edges(
        "grade", should_retry_node, {"retry": "retry", "finalize": "finalize"}
    )

    return workflow.compile()


agent = build_agent()
