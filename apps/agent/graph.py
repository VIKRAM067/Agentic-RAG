from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.documents import Document
from apps.core.config import settings
from apps.core.prompts import router_prompt, rag_prompt
from apps.services.retriever import hybrid_retriever
from apps.services.reranker import reranker
from apps.services.chain import get_llm, format_context, grade_answer, rag_chain
from langchain_core.output_parsers import StrOutputParser
import json


class AgentState(TypedDict):
    question: str
    strategy: str
    documents: List[Document]
    context: str
    answer: str
    grade: dict
    retry_count: int
    final_answer: str


def route_Node(state: AgentState) -> str:
    llm = get_llm()
    router_chain = router_prompt | llm | StrOutputParser()
    result = router_chain.invoke({"question": state["question"]})
    try:
        parsed = json.loads(result)
        strategy = parsed.get("strategy", "hybrid")
    except json.JSONDecodeError:
        strategy = "hybrid"  # fallback if parsing fails

    return {**state, "strategy": strategy}


def retrieve_Node(state: AgentState) -> AgentState:
    hybrid_retriever.strategy = state["strategy"]

    docs = hybrid_retriever.invoke(state["question"])

    reranked_docs = reranker.rerank(state["question"], docs)

    context = format_context(reranked_docs)

    return {**state, "documents": reranked_docs, "context": context}


def generate_Node(state: AgentState) -> AgentState:
    answer = rag_chain.invoke(
        {"question": state["question"], "context": state["context"]}
    )
    return {**state, "answer": answer}


def grade_Node(state: AgentState) -> AgentState:
    grade = grade_answer(state["question"], state["context"], state["answer"])
    return {**state, "grade": grade}


def finalize_node(state: AgentState) -> AgentState:
    return {**state, "final_answer": state["answer"]}


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

    # entry point
    workflow.set_entry_point("route")

    # fixed edges
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "grade")
    workflow.add_edge("retry", "retrieve")
    workflow.add_edge("finalize", END)

    # conditional edge — retry or finalize?
    workflow.add_conditional_edges(
        "grade", should_retry_node, {"retry": "retry", "finalize": "finalize"}
    )

    return workflow.compile()


agent = build_agent()
